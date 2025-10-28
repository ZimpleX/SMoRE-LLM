from __future__ import annotations

from peft.tuners.tuners_utils import BaseTunerLayer
import math
from typing import Optional, List
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from model_moe.model_utils import ParamStatsSMoRE
from model_moe.moe import BaseMoELayer
from model_moe.moe import MoEMlpOutput


@dataclass(frozen=True)
class SMoREInitMethod:
    NORMAL: str="normal"
    B_ZERO: str="b_zero"
    BW_ZERO: str="bw_zero"
    W_FINAL_ZERO: str="w_final_zero"
    EQ_1_LAYER: str="eq_1_layer"



class SMoREFFN(BaseMoELayer, BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        "moe_gate", 
        # "moe_gate_keys", 
        "moe_downproj_x", 
        "moe_A", 
        "moe_B", 
        "moe_W",
        "moe_W_final",
        "moe_A_bias",
        "moe_B_bias",
        "moe_W_bias",
        "moe_W_final_bias",
        # learnable activation
        "moe_expert_act_param",
        # "gate_act_param",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "hidden_act", "num_experts", "num_active", "expert_dims", "gate_type", "lora_alpha", "balance_loss_weight",
        "init_method", "is_shared_expert_pool", "activation_position",
    )

    def __init__(
        self, 
        base_layer, 
        adapter_name: str,
        r: List[int],  # dim of residual [r]anks for all sub-layers; 0 ~ L-1
        s: List[int],  # [s]ize of all sub-layers (# residual terms): 0 ~ L-1
        f: List[int],  # sampling [f]anout for all sub-layers: 0 ~ L-1
        d_g: List[int],  # dim of gates for all sub-layers: 0 ~ L-1
        # lora_alpha: List[float],
        dim_downproj_x: int,  # downproj 4k to small value for input into gate MLP
        expert_act_fn: str,
        gate_act_fn: str,
        gate_type: str, # dense, noisy_topk, switch
        gate_arch: str, # mlp, linear, attn, recurrent
        init_method: str=SMoREInitMethod.NORMAL,
        dim_hid_gate_mlp: Optional[int]=None,
        # additional configs in rebuttal
        lora_alpha: Optional[List[float]]=None,
        # infra param
        layer_forward_mode: str="subgraph",
        balance_loss_weight: float=1e-2,
        # v2 design: shared expert pool for all layers
        is_shared_expert_pool: bool=False,
        is_selective_final_proj: bool=False,
        activation_position: str="all",
    ):
        super().__init__(base_layer)
        assert activation_position in ("all", "child", "child_v2")
        self.activation_position = activation_position
        if lora_alpha is None:
            self.lora_scaling = [1.0 for _ in r]  # default for submitted version
            self.lora_alpha = [sc * ri for sc, ri in zip(self.lora_scaling, r)]
        else:
            assert len(lora_alpha) == len(s)
            self.lora_scaling = [ai / ri for ai, ri in zip(lora_alpha, r)]
            self.lora_alpha = lora_alpha
        print(self.lora_scaling)
        # -------
        self.param_stats = ParamStatsSMoRE()
        # -------
        self.balance_loss_weight = balance_loss_weight
        self.adapter_name = adapter_name
        self.expert_act_fn = expert_act_fn
        assert isinstance(base_layer, torch.nn.Linear)
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.dim = self.in_features
        # 
        self._setup_per_layer_arch(r, s, f, is_shared_expert_pool)
        self._setup_layer_params(adapter_name, is_shared_expert_pool)
        L = self.num_sublayers
        # gates
        # batch x token x 4096 -> batch x token x 32
        # sub-layer 1:
        #    batch x token x 32 -> batch x token x 8 -> select 2
        # sub-layer 2:
        #    batch x token x 2 x (32 + 32) -> select 2 x 2
        # L x s x dim
        self.d_g = d_g  # index 0 ~ L-1: key vec associated with each residual term
        assert len(self.d_g) == L
        if dim_downproj_x is None or dim_downproj_x <= 0:
            dim_downproj_x = self.dim
        self.moe_downproj_x = nn.ModuleDict(
            {adapter_name: nn.Linear(in_features=self.dim, out_features=dim_downproj_x)}
        )
        self.param_stats.gate_param += self.param_stats.accum_param(adapter_name, self.moe_downproj_x)
        self.param_stats.gate_flops += self.dim * dim_downproj_x
        dim_in_gate_mlp = [None] * L
        parent_fanout = [None] * L
        ancestor_dim = 0
        for l in range(L - 1, -1, -1): # L-1 ~ 0
            if gate_arch == "bottom_up":
                dim_in_gate_mlp[l] = dim_downproj_x if l == 0 else self.d[l]
            else:
                dim_in_gate_mlp[l] = dim_downproj_x + ancestor_dim
            ancestor_dim += d_g[l]
            if l == L-1:
                parent_fanout[l] = []
            else:
                parent_fanout[l] = parent_fanout[l + 1] + [self.f[l + 1]]
        self.moe_gate = nn.ModuleDict({adapter_name: nn.ModuleList([])})
        if gate_type == "noisy_topk":
            full_gate_type = "SMoREGate"
        elif gate_type == "dense":
            full_gate_type = "SMoREDenseGate"
        elif gate_type == "switch":
            full_gate_type = "SMoRESwitchGate"
        
        self.is_bottom_up_routing = (gate_arch == "bottom_up")
        for l in range(L):
            kwargs = {
                "dim_in": dim_in_gate_mlp[l],
                "dim_hid": d_g[l],
                "num_experts": self.s[l],
                "num_active": self.f[l],
                "act_fn": gate_act_fn,
                "gate_type": full_gate_type,
                "gate_arch": gate_arch,
                "balance_loss_weight": balance_loss_weight,
                # for bottom-up routing                
                "parent_fanout": parent_fanout[l],
            }
            _gate = self._create_gate(**kwargs)
            _gate.set_sublayer_id(l)
            self.moe_gate[adapter_name].append(_gate)
            self.param_stats.gate_flops += _gate.get_inference_flops()
        self.param_stats.gate_param += self.param_stats.accum_param(adapter_name, self.moe_gate)
        self.param_stats.trainable_param = self.param_stats.expert_param + self.param_stats.gate_param
        self.param_stats.int()
        self.init_method = init_method
        self.reset_parameters(adapter_name, init_method)

        # ---
        self.device = device
        self.dtype = dtype
        self._move_adapter_to_device_of_base_layer(adapter_name, device=device, dtype=dtype)
        self.set_adapter(self.active_adapters)

        # infra optimization
        assert layer_forward_mode in ["subgraph", "tree", "loop"] or is_shared_expert_pool # should always try to use "subgraph" mode
        self.layer_forward_mode = layer_forward_mode
        if not is_shared_expert_pool:
            if layer_forward_mode == "subgraph":
                self._forward_layer = self._forward_layer_mode_2
            elif layer_forward_mode == "tree":
                self._forward_layer = self._forward_layer_mode_1
            elif layer_forward_mode == "loop":
                self._forward_layer = self._forward_layer_mode_0
        else:
            self._forward_layer = self.forward_shared_expert_pool
    
    def _setup_per_layer_arch(
        self, r: List[int], s: List[int], f: List[int], is_shared_expert_pool: bool
    ):
        self.is_shared_expert_pool = is_shared_expert_pool
        self.r = r
        self.s = s
        self.f = f
        assert len(self.r) == len(self.s) == len(self.f), "num sub-layers mismatch!!"
        L = self.num_sublayers
        if not is_shared_expert_pool:
            # assert self.f[0] is None or self.f[0] <= 0, "f[0] will be ignored!"
            # recursion to get hidden dim for sub-layer
            d = [0] + [None] * L  # d_0 = 0
            for l in range(1, L + 1):
                d[l] = d[l - 1] + s[l - 1] * r[l - 1]
            # d[L] = self.out_features  # output dim must equal input dim to allow stack of transformer layers
            self.d = d
        else:
            assert len(set(self.r)) == len(set(self.s)) == 1, \
                f"shared expert pool across layers. inconsistent expert spec {r=}\t{s=}"
            self.d = [0] + [self.s[0] * self.r[0]] * L

    def _setup_layer_params(self, adapter_name: str, is_shared_expert_pool: bool):
        L = self.num_sublayers
        r, s, d = self.r, self.s, self.d
        # construct weight params
        self.is_expert_bias = True
        for pn in ("moe_A", "moe_B", "moe_W"):
            suffixes = ("", "_bias") if self.is_expert_bias else ("",)
            for suffix in suffixes:
                setattr(self, pn + suffix, nn.ModuleDict({adapter_name: nn.ParameterList([])}))
        self.moe_W_final = nn.ParameterDict({adapter_name: nn.Parameter(torch.empty((self.out_features, d[L])))})
        if self.is_expert_bias:
            self.moe_W_final_bias = nn.ParameterDict({adapter_name: nn.Parameter(torch.empty(self.out_features))})
        else:
            self.moe_A_bias = self.moe_B_bias = self.moe_W_bias = nn.ModuleDict({adapter_name: None})
            self.moe_W_final_bias = nn.ParameterDict({adapter_name: None})
        
        def _append_empty_param(pt, shape):
            pt[adapter_name].append(nn.Parameter(torch.empty(shape)))
        def _append_shared_param(pt, shape):
            assert shape == pt[adapter_name][0].shape
            pt[adapter_name].append(pt[adapter_name][0])
        
        # set up W
        self.moe_W[adapter_name].append(None)
        if self.is_expert_bias:
            self.moe_W_bias[adapter_name].append(None)
        for l in range(1, L):
            init_dict = {"moe_W": (d[l + 1], d[l])}
            if self.is_expert_bias:
                init_dict["moe_W_bias"] = d[l + 1]
            for k, v in init_dict.items():
                _append_empty_param(getattr(self, k), v)
        # set up A, B
        for l in range(L):
            init_dict = {"moe_A": (s[l], r[l], self.dim), "moe_B": (s[l], d[l + 1], r[l])}
            if self.is_expert_bias:
                init_dict["moe_A_bias"] = (s[l], r[l])
                init_dict["moe_B_bias"] = (s[l], d[l + 1])
            _f = (
                _append_empty_param
                if l == 0 or not is_shared_expert_pool
                else _append_shared_param
            )
            for k, v in init_dict.items():
                _f(getattr(self, k), v)

        # update param stats
        num_active_experts_l = reduce(lambda x, y: x * y, self.f)
        if not is_shared_expert_pool:
            for l in range(L):
                num_act_experts_dup = num_active_experts_l
                num_act_experts_uniq = min(num_active_experts_l, s[l])
                # active param: just add A, B now. Handle W outside loop
                _cnt_AB = self.dim * r[l] + r[l] * d[l + 1]
                self.param_stats.active_param_wo_proj += num_act_experts_uniq * _cnt_AB
                self.param_stats.expert_flops_wo_proj += (
                    num_act_experts_uniq * _cnt_AB
                    + num_act_experts_dup * (d[l + 1] * d[l])
                )
                num_active_experts_l /= self.f[l]
                if self.is_expert_bias:
                    self.param_stats.active_param_wo_proj += num_act_experts_uniq * (r[l] + d[l + 1])
        else:
            total_act_experts = 0
            for l in range(L):
                total_act_experts += num_active_experts_l
                self.param_stats.expert_flops_wo_proj += num_active_experts_l * (d[l + 1] * d[l])  # W
                num_active_experts_l /= self.f[l]
            total_act_experts = min(total_act_experts, self.s[0])
            _cnt_AB = self.dim * r[0] + r[0] * d[-1]
            self.param_stats.expert_flops_wo_proj += total_act_experts * _cnt_AB
            self.param_stats.active_param_wo_proj += total_act_experts * (_cnt_AB + r[0] + d[-1])
        self.param_stats.expert_param_wo_proj = self.param_stats.accum_param(
            adapter_name, 
            self.moe_A, 
            self.moe_B, 
            self.moe_W,
            self.moe_A_bias,
            self.moe_B_bias,
            self.moe_W_bias,
        )
        self.param_stats.active_param_wo_proj += self.param_stats.accum_param(
            adapter_name, self.moe_W, self.moe_W_bias,
        )
        self.param_stats.active_param = (
            self.param_stats.active_param_wo_proj
            + self.param_stats.accum_param(adapter_name, self.moe_W_final, self.moe_W_final_bias)
        )
        self.param_stats.expert_flops = self.param_stats.expert_flops_wo_proj + (self.out_features * d[L]) # TODO adapt based on is_selective_final_proj
        self.param_stats.expert_param = (
            self.param_stats.expert_param_wo_proj 
            + self.param_stats.accum_param(adapter_name, self.moe_W_final, self.moe_W_final_bias)
        )
        # init activation
        if self.expert_act_fn.startswith("prelu"):
            pstruct = nn.ParameterDict(
                {adapter_name: nn.ParameterList([nn.Parameter(torch.empty(1)) for _ in range(L)])}
            )
            for stat_name in (
                "trainable_param",
                "active_param",
                "active_param_wo_proj",
                "expert_param",
                "expert_param_wo_proj",
            ):
                _val = getattr(self.param_stats, stat_name)
                setattr(self.param_stats, stat_name, _val + L)
        else:
            pstruct = nn.ParameterDict({adapter_name: None})
        self.moe_expert_act_param = pstruct
    
    def _prelu_act(self, input: torch.Tensor, adapter_name: str, layer_idx: int):
        a = self.moe_expert_act_param[adapter_name][layer_idx].flatten()
        return torch.clamp(input, min=0) + a * torch.clamp(input, max=0)

    def activation(
        self, 
        input1: torch.Tensor, 
        input2: Optional[torch.Tensor], 
        adapter_name: str, 
        layer_idx: int,
        W: Optional[torch.Tensor],
        W_bias: Optional[torch.Tensor],
        score: torch.Tensor,
    ):
        # _x2 = Wl @ x_prev + Wl_bias
        if self.expert_act_fn == "identity":
            if input2 is None:
                return (score * input1).sum(-3)
            elif self.activation_position == "all":
                return (score * (input1 + W @ input2 + W_bias)).sum(-3)
            elif self.activation_position == "child":
                return (score * input1 + W @ input2 + W_bias).sum(-3)
            elif self.activation_position == "child_v2":
                return (score * (input1 + W @ input2 + W_bias)).sum(-3)
        elif self.expert_act_fn in ("tanh", "relu") or self.expert_act_fn.startswith("prelu"):
            if self.expert_act_fn == "tanh":
                _fn = F.tanh
            elif self.expert_act_fn == "relu":
                _fn = F.relu
            elif self.expert_act_fn.startswith("prelu"):
                _fn = partial(self._prelu_act, adapter_name=adapter_name, layer_idx=layer_idx)
            else:
                raise NotImplementedError
            if input2 is None:
                if self.activation_position == "all":
                    return (score * _fn(input1)).sum(-3)
                else:
                    assert self.activation_position in ["child", "child_v2"]
                    return (score * input1 + 0 * _fn(torch.zeros(1, device=input1.device, dtype=input1.dtype))).sum(-3)
            else:
                if self.activation_position == "all":
                    return (score * _fn(input1 + W @ input2 + W_bias)).sum(-3)
                elif self.activation_position == "child":
                    return (score * input1 + W @ _fn(input2) + W_bias).sum(-3)
                else:
                    assert self.activation_position == "child_v2"
                    return (score * (input1 + W @ _fn(input2) + W_bias)).sum(-3)
        else:
            raise NotImplementedError
    
    @classmethod
    def _get_uniq_params(cls, param_list):
        """ handle the case for param sharing in is_shared_expert_pool """
        return {id(t): t for t in param_list}.values()
        
    def reset_parameters(self, adapter_name: str, init_method: str):
        # ("moe_gate", "moe_downproj_x", "moe_A", "moe_B", "moe_W")
        for Al in self._get_uniq_params(self.moe_A[adapter_name]):
            for Als in Al:
                # NOTE kaiming_uniform_ cares about weight vs. weight.t()
                # Here Als has shape (out_dim, in_dim)
                nn.init.kaiming_uniform_(Als, a=math.sqrt(5))
        for Bl in self._get_uniq_params(self.moe_B[adapter_name]):  # when is_shared_expert_pool, there's only 1 Bl
            _s, _d, _r = Bl.shape
            if self.is_shared_expert_pool:
                assert _s == self.s[0] and _r == self.r[0] and _d == self.d[-1]
            for i, Bls in enumerate(Bl):  # s dimension
                # consider B, W all zero for non-final layers?
                if init_method in (SMoREInitMethod.B_ZERO, SMoREInitMethod.BW_ZERO):
                    nn.init.zeros_(Bls)
                elif init_method in (SMoREInitMethod.NORMAL, SMoREInitMethod.W_FINAL_ZERO):
                    nn.init.kaiming_uniform_(Bls, a=math.sqrt(5))
                elif init_method in (SMoREInitMethod.EQ_1_LAYER,):
                    with torch.no_grad():  # s, s*r, r
                        Bls[:] = 0
                        Bls[i * _r : (i + 1) * _r, :] = torch.eye(_r, device=Bls.device, dtype=Bls.dtype)
                else:
                    raise NotImplementedError
        for Wl in self.moe_W[adapter_name]:  # NOTE: no param sharing for W
            if Wl is not None:
                if init_method in (SMoREInitMethod.BW_ZERO,):
                    nn.init.zeros_(Wl)
                elif init_method in (SMoREInitMethod.EQ_1_LAYER, ):
                    assert len(Wl.shape) == 2
                    if self.is_shared_expert_pool:
                        assert Wl.shape[0] == Wl.shape[1]
                    with torch.no_grad():
                        Wl[:] = 0
                        # P_{d_L \times d_{\ell+1}} according to Eq 25
                        Wl[-Wl.shape[-1]:, :] = torch.eye(Wl.shape[-1], device=Wl.device, dtype=Wl.dtype)
                else:
                    nn.init.kaiming_uniform_(Wl, a=math.sqrt(5))
        if self.is_expert_bias:
            for bias_l in self._get_uniq_params(self.moe_A_bias[adapter_name]):
                nn.init.zeros_(bias_l)
            for bias_l in self._get_uniq_params(self.moe_B_bias[adapter_name]):
                nn.init.zeros_(bias_l)
            for bias_l in self.moe_W_bias[adapter_name]:
                if bias_l is not None:
                    nn.init.zeros_(bias_l)
        # final proj weight
        if init_method in (SMoREInitMethod.W_FINAL_ZERO, SMoREInitMethod.EQ_1_LAYER):
            nn.init.zeros_(self.moe_W_final[adapter_name])
        elif init_method in (SMoREInitMethod.B_ZERO, SMoREInitMethod.BW_ZERO, SMoREInitMethod.NORMAL):
            nn.init.kaiming_uniform_(self.moe_W_final[adapter_name], a=math.sqrt(5))
        else:
            raise NotImplementedError
        nn.init.zeros_(self.moe_W_final_bias[adapter_name])

        if self.expert_act_fn.startswith("prelu"):
            _, init = self.expert_act_fn.split("|")
            for pinit in self.moe_expert_act_param[adapter_name]:
                nn.init.constant_(pinit, float(init))
        # moe_gate has already been initialized
        # moe_gate_keys / moe_downproj_x
        
    def _unittest_init_eq_1_layer(self, adapter_name: str):
        """
        unit-test case: 2-layer smore exactly reduces to 1-layer
        hypothetical expert selection:
        # layer 2: 3, 1
        # layer 1: [3, 2], [0, 1]
        """
        _A = self.moe_A[adapter_name]
        _B = self.moe_B[adapter_name]
        _W = self.moe_W[adapter_name]
        # _WL = self.moe_W_final[adapter_name]
        _x = torch.rand(_A[0].shape[-1])
        _y10 = _B[0][3]@(_A[0][3]@_x) + _B[0][2]@(_A[0][2]@_x)
        _y11 = _B[0][0]@(_A[0][0]@_x) + _B[0][1]@(_A[0][1]@_x)
        _z20 = _B[1][3]@_A[1][3]@_x + _W[1]@_y10
        _z21 = _B[1][1]@_A[1][1]@_x + _W[1]@_y11
        y = (_z20 + _z21)
        if self.is_shared_expert_pool:
            y = y.reshape(4, -1)
            y_compare = {
                0: _A[0][0]@_x,
                1: _A[0][1]@_x * 2,
                2: _A[0][2]@_x,
                3: _A[0][3]@_x * 2,
            }
        else:
            y = y.reshape(8, -1)
            y_compare = {
                1: _A[1][1]@_x,
                3: _A[1][3]@_x,
                4: _A[0][0]@_x,
                5: _A[0][1]@_x,
                6: _A[0][2]@_x,
                7: _A[0][3]@_x
            }
        for k, v in y_compare.items():
            assert torch.allclose(y[k], v)

    @property
    def num_sublayers(self):
        return len(self.r)
    
    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device, dtype):
        """
        Simplified implementation based on BaseTuner:_move_adapter_to_device_of_base_layer
        """
        for adapter_layer_name in self.adapter_layer_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            assert adapter_layer is not None, f"SMoRE should contain {adapter_layer_name}"
            if adapter_layer[adapter_name] is not None:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
    
    def _forward_layer_mode_0(
        self, 
        adapter_name: str, 
        total_layers: int, 
        layer_idx: int, 
        x: torch.Tensor, 
        x_prev: torch.Tensor,
        id_exp: torch.Tensor, 
        score_exp: torch.Tensor, # = topK_score[l]
    ):
        """ loop """
        if layer_idx != total_layers - 1:
            return self._forward_layer_mode_1(  # tree
                adapter_name=adapter_name,
                total_layers=total_layers,
                layer_idx=layer_idx,
                x=x,
                x_prev=x_prev,
                id_exp=id_exp,
                score_exp=score_exp,
            )
        moe_A = self.moe_A[adapter_name][layer_idx]
        moe_B = self.moe_B[adapter_name][layer_idx]
        moe_W = self.moe_W[adapter_name][layer_idx]
        assert moe_W is None and self.expert_act_fn == "identity", \
            "forward_mode_0 currently only available for mixlora 1-layer model"
        moe_A_bias = self.moe_A_bias[adapter_name][layer_idx]
        moe_B_bias = self.moe_B_bias[adapter_name][layer_idx]
        scaling = self.lora_scaling[layer_idx]
        # Al = moe_A[id_exp]  # (batch * seq) x f1 x f0 x r0 x 4k
        # Bl = moe_B[id_exp]  # (batch * seq) x f1 x f0 x d1 x r0
        assert len(id_exp.shape) == 2
        ret = 0
        f0 = id_exp.shape[-1]  # id_exp: (batch * seq) x f0
        for ie in range(f0):
            fi = id_exp[:, ie]
            score_i = score_exp[:, ie]
            Ai = moe_A[fi]  # (batch * seq) x r x 4k
            Bi = moe_B[fi]  # (batch * seq) x d x r
            Ai_bias = 0 if moe_A_bias is None else moe_A_bias[fi][..., None]
            Bi_bias = 0 if moe_B_bias is None else moe_B_bias[fi][..., None]
            ret += (Bi @ (Ai @ x + Ai_bias) + Bi_bias) * scaling * score_i[:, None, None]
        return ret

    def _forward_layer_mode_1(
        self, 
        adapter_name: str, 
        total_layers: int, 
        layer_idx: int, 
        x: torch.Tensor, 
        x_prev: torch.Tensor,
        id_exp: torch.Tensor, 
        score_exp: torch.Tensor, # = topK_score[l]
    ):
        """ tree """
        moe_A = self.moe_A[adapter_name][layer_idx]
        moe_B = self.moe_B[adapter_name][layer_idx]
        moe_W = self.moe_W[adapter_name][layer_idx]
        moe_A_bias = self.moe_A_bias[adapter_name][layer_idx]
        moe_B_bias = self.moe_B_bias[adapter_name][layer_idx]
        moe_W_bias = self.moe_W_bias[adapter_name][layer_idx]
        scaling = self.lora_scaling[layer_idx]
        Al = moe_A[id_exp]  # (batch * seq) x f1 x f0 x r0 x 4k
        Bl = moe_B[id_exp]  # (batch * seq) x f1 x f0 x d1 x r0
        Wl = moe_W  # d1 x d0
        Al_bias = 0 if moe_A_bias is None else moe_A_bias[id_exp][..., None]
        Bl_bias = 0 if moe_B_bias is None else moe_B_bias[id_exp][..., None]
        Wl_bias = 0 if moe_W_bias is None or moe_W_bias is None else moe_W_bias[..., None]
        _x = x
        for _ in range(total_layers - layer_idx):
            _x = _x.unsqueeze(-3)
        score_exp = score_exp[..., None, None]
        _x1 = (Bl @ (Al @ _x + Al_bias) + Bl_bias) * scaling
        if layer_idx == 0:
            x_prev = None
        return self.activation(_x1, x_prev, adapter_name, layer_idx, Wl, Wl_bias, score_exp)
        # return (score_exp * _act).sum(-3)  # (batch * seq) x f1 x f0 x d1 x 1

    def _forward_layer_mode_2(
        self, 
        adapter_name: str, 
        total_layers: int, 
        layer_idx: int, 
        x: torch.Tensor, 
        x_prev: torch.Tensor,
        id_exp: torch.Tensor, 
        score_exp: torch.Tensor, # = topK_score[l]
    ):
        """ subgraph """
        moe_A = self.moe_A[adapter_name][layer_idx]  # s, r, d
        moe_B = self.moe_B[adapter_name][layer_idx]  # s, d', r
        moe_W = self.moe_W[adapter_name][layer_idx]  # d', d
        moe_A_bias = self.moe_A_bias[adapter_name][layer_idx]
        moe_B_bias = self.moe_B_bias[adapter_name][layer_idx]
        moe_W_bias = self.moe_W_bias[adapter_name][layer_idx]
        scaling = self.lora_scaling[layer_idx]
        # bias
        Al_bias = 0 if moe_A_bias is None else moe_A_bias[None, ..., None]
        Bl_bias = 0 if moe_B_bias is None else moe_B_bias[None, ..., None]
        Wl_bias = 0 if moe_W_bias is None or moe_W_bias is None else moe_W_bias[None, ..., None]
        # all_lora_out: batch*seq, sl, dl, 1
        all_lora_out = ((moe_B[None, ...] @ (moe_A[None, ...] @ x[:, None, ...] + Al_bias)) + Bl_bias) * scaling
        # id_exp: batch*seq, f', f
        # _x1: batch*seq, f', f, dl, 1
        id_exp_reshape = id_exp.flatten(1)[..., None, None]
        id_exp_reshape = id_exp_reshape.expand(*id_exp_reshape.shape[:-2], *all_lora_out.shape[-2:])
        _x1 = torch.gather(all_lora_out, 1, id_exp_reshape).unflatten(1, id_exp.shape[1:])
        if layer_idx == 0:
            x_prev = None
        return self.activation(_x1, x_prev, adapter_name, layer_idx, moe_W, Wl_bias, score_exp[..., None, None])
        # return (score_exp[..., None, None] * _act).sum(-3)  # (batch * seq) x f1 x f0 x d1 x 1
    
    def forward_shared_expert_pool(
        self, 
        adapter_name: str, 
        x: torch.Tensor, 
        id_exp_all: List[torch.Tensor], 
        score_exp_all: List[torch.Tensor], # = topK_score[l]
    ):
        """ 
        shared expert pool: precompute all selected experts
        id_exp: (seq_len, f, f)
        """
        L = len(id_exp_all)
        xl = None
        all_experts = self.s[0]
        for l in range(L):
            all_fanout = math.prod(id_exp_all[l].shape[1:])
            if all_fanout >= 0.5 * all_experts:
                _fn_forward = self._forward_layer_mode_2
            else:
                _fn_forward = self._forward_layer_mode_1
            xl = _fn_forward(
                adapter_name=adapter_name, 
                total_layers=L, 
                layer_idx=l, 
                x=x, 
                x_prev=xl,
                id_exp=id_exp_all[l], 
                score_exp=score_exp_all[l],
            )
        return xl

    def forward(self, x, attention_mask_flat=None, act_at_last: bool=False) -> MoEMlpOutput:
        y_base = self.base_layer(x.squeeze(-1))
        torch_result_dtype = y_base.dtype
        original_shape = x.shape[:-1]
        adapter_name = self.adapter_name
        x = x.reshape(-1, self.in_features)  # (batch * seq) x in_features
        # x = x.to(self.moe_downproj_x[adapter_name].weight.dtype)
        x = x.to(self.moe_B["default"][-1].dtype)
        L = self.num_sublayers
        moe_gate = self.moe_gate[adapter_name]
        if self.moe_downproj_x[adapter_name] is not None:
            x_parent = self.moe_downproj_x[adapter_name](x)
        else:
            x_parent = x
        balance_loss = 0
        if not self.is_bottom_up_routing:
            topK_id = [None] * L
            topK_score = [None] * L
            load_all = [None] * L
            importance_all = [None] * L
            for l in range(L - 1, -1, -1):
                gate_outputs = moe_gate[l](x_parent, attention_mask_flat=attention_mask_flat)
                if gate_outputs["balance_loss"] is not None:
                    balance_loss += gate_outputs["balance_loss"]
                load_all[l] = gate_outputs["load"]
                importance_all[l] = gate_outputs["importance"]
                topK_id[l] = gate_outputs["topK_indices"]
                topK_score[l] = gate_outputs["topK_scores"]
                selected_keys = gate_outputs["selected_keys"]  # (batch * seq) x f2 x f1 x dim
                if l > 0:  # no need to prepare gate input in sub-layer 0
                    x_parent = x_parent[..., None, :]
                    x_parent_shape = list(x_parent.shape)
                    x_parent_shape[-2] = selected_keys.shape[-2]
                    x_parent = torch.cat((x_parent.expand(*x_parent_shape), selected_keys), dim=-1)
        else:
            topK_id = None
            topK_score = None
            load_all = None
            importance_all = None
        x = x[..., None]
        xl = None  # layer 0 doesn't need xl
        if not self.is_shared_expert_pool:
            for l in range(L):
                if self.is_bottom_up_routing:
                    raise NotImplementedError
                else:
                    _topK_id = topK_id[l]
                    _topK_score = topK_score[l]
                xl = self._forward_layer(
                    adapter_name=adapter_name, 
                    total_layers=L, 
                    layer_idx=l, 
                    x=x, 
                    x_prev=xl,
                    id_exp=_topK_id, 
                    score_exp=_topK_score,
                )
        else:
            xl = self.forward_shared_expert_pool(
                adapter_name=adapter_name, 
                x=x, 
                id_exp_all=topK_id,
                score_exp_all=topK_score,
            )
        y = xl
        # * get unique expert indices for all layers. 
        # * each selected final proj: 8*32 -> 32 -> 4096
        # * sum final 4096 over all unique active experts
        if self.moe_W_final[adapter_name] is not None:  # for compatibility with mixlora design
            y = (self.moe_W_final[adapter_name][None, ...] @ y).squeeze(-1)
        if self.moe_W_final_bias[adapter_name] is not None:
            y += self.moe_W_final_bias[adapter_name][None, :]
        out_shape = original_shape + (self.out_features,)
        y = y.to(torch_result_dtype).reshape(out_shape) + y_base.reshape(out_shape)
        if not isinstance(balance_loss, torch.Tensor):
            balance_loss = None
            load_all = [torch.tensor(0.0)]
            importance_all = [torch.tensor(0.0)]
        return MoEMlpOutput(
            hidden_states=y,
            balance_loss=balance_loss if isinstance(balance_loss, torch.Tensor) else None,
            num_dropped_tokens=torch.tensor(-1.0),
            gate_load=load_all,
            gate_importance=importance_all,
        )
