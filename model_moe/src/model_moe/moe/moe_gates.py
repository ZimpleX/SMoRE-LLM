import warnings

import torch
from torch import nn
from torch.distributions.normal import Normal
from typing import Optional, List
import torch.nn.functional as F
import math

valid_gate_type = ("linear", "mlp")


def cv_squared(x, eps=1e-10):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
    x: a `Tensor`.
    Returns:
    a `Scalar`.s
    """
    # if only num_experts = 1
    if x.shape[0] == 1:
        return torch.tensor(0.0, device=x.device)
    return x.float().var() / (x.float().mean() ** 2 + eps)


class SMoREGateCommon(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        num_experts: int,
        num_active: int,
        act_fn,
        gate_arch: str,
        dim_out: Optional[int]=None,
        #
        parent_fanout: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        assert gate_arch in ("mlp", "linear", "attention", "bottom_up") or gate_arch.startswith("mixlora"), f"Unknown S'MoRE gate type: {gate_arch}"
        self.gate_arch = gate_arch
        self.sublayer_id = -1
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.act_fn = act_fn
        self.parent_fanout = parent_fanout
        if dim_out is None:
            self.dim_out = self.dim_hid
        else:
            self.dim_out = dim_out

        if self.gate_arch not in ("bottom_up",):
            if self.act_fn == "identity":
                _act = nn.Identity()
            elif self.act_fn == "tanh":
                _act = nn.Tanh()
            elif self.act_fn == "relu":
                _act = nn.ReLU()
            elif self.act_fn.startswith("prelu"):
                _, init = self.act_fn.split("|")
                _act = nn.PReLU(init=float(init))
            else:
                raise NotImplementedError
            if gate_arch == "mlp":
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.dim_in, self.dim_hid, bias=True),
                    _act,
                    torch.nn.Linear(self.dim_hid, self.dim_out, bias=True),
                )
                # key vector for gate-level dot product
                self.keys = nn.Parameter(torch.empty([self.num_experts, self.dim_out]))
            else:
                self.mlp = torch.nn.Sequential(torch.nn.Identity())
                self.keys = nn.Parameter(torch.empty([self.num_experts, self.dim_in]))
        elif self.gate_arch == "bottom_up":
            self.mlp = torch.nn.Sequential(torch.nn.Identity())
            self.keys = nn.Parameter(torch.empty([*self.parent_fanout, self.num_experts, self.dim_in]))
        else:
            raise NotImplementedError
    
    def reset_parameters(self):
        if self.gate_arch in ["linear", "bottom_up"]:
            nn.init.normal_(self.keys, mean=0.0, std=0.02)
            # raise NotImplementedError
            # bound = 1 / math.sqrt(self.keys.shape[1]) # should be sqrt token dim
            # nn.init.uniform_(self.keys, -bound, bound)
        else:
            nn.init.uniform_(self.keys)
    
    def set_sublayer_id(self, sublayer_id: int):
        self.sublayer_id = sublayer_id
    
    def get_inference_flops(self):
        counter = 0
        for p in self.mlp:
            if isinstance(p, nn.Linear):
                counter += p.weight.numel()
        # dot product for self.keys
        counter += math.prod(self.keys.shape)
        return counter


class SMoRESwitchGate(SMoREGateCommon):
    """
    Ref to implementation of MixLoRA
    """

    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        num_experts: int,
        num_active: int,
        act_fn,
        dim_out: Optional[int]=None,
        gate_arch: str="mlp",
        balance_loss_weight: float=1e-2,
        # reviewer question
        parent_fanout: Optional[List[int]] = None,
    ):
        super().__init__(
            dim_in=dim_in,
            dim_hid=dim_hid,
            num_experts=num_experts,
            num_active=num_active,
            act_fn=act_fn,
            gate_arch=gate_arch,
            dim_out=dim_out,
            parent_fanout=parent_fanout,
        )
        self.balance_loss_weight = balance_loss_weight
        self.reset_parameters()
    
    def forward(self, x, attention_mask_flat=None):
        logits_gate = self.mlp(x) @ self.keys.t()
        score_gate = F.softmax(logits_gate, dim=-1)
        top_k_scores, top_k_indices = score_gate.topk(self.num_active, dim=-1)
        if self.num_active > 1:
            top_k_scores /= top_k_scores.sum(dim=-1, keepdim=True)
        if attention_mask_flat is not None:
            assert len(attention_mask_flat.shape) == 2, "attention_mask_flat should have shape (batch_size, seq_len)"
            attention_mask_flat = attention_mask_flat.flatten()
        else:
            attention_mask_flat = torch.ones(x.shape[0], device=x.device, dtype=torch.int)
        num_valid_tokens = attention_mask_flat.sum() * math.prod(score_gate.shape[1: -1])
        for i in range(len(score_gate.shape) - 1):
            attention_mask_flat = attention_mask_flat.unsqueeze(-1)
        tokens_per_expert = (
            F.one_hot(top_k_indices, self.num_experts) * attention_mask_flat[..., None]
        ).flatten(0, -2).sum(dim=0) / num_valid_tokens
        score_per_expert = (score_gate * attention_mask_flat).flatten(0, -2).sum(dim=0) / num_valid_tokens
        balance_loss = self.balance_loss_weight * self.num_experts * (tokens_per_expert * score_per_expert).sum()
        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": tokens_per_expert * num_valid_tokens,
            "importance": torch.tensor(0.0),
            #
            "selected_keys": self.keys[top_k_indices],
        }
        

class SMoREDenseGate(SMoREGateCommon):
    """
    No load-balance. All experts are activated

    Similar to HydraLoRA
    """

    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        num_experts: int,
        num_active: int,
        act_fn,
        dim_out: Optional[int]=None,
        gate_arch: str="mlp",
    ):
        super().__init__(
            dim_in=dim_in,
            dim_hid=dim_hid,
            num_experts=num_experts,
            num_active=num_active,
            act_fn=act_fn,
            gate_arch=gate_arch,
            dim_out=dim_out,
        )
        self.reset_parameters()
    
    def forward(self, x, attention_mask_flat=None):
        """
        attention_mask_flat is unused, since we don't compute balance loss here
        """
        logits_gate = self.mlp(x) @ self.keys.t()
        score_gate = F.softmax(logits_gate, dim=-1)
        num_experts = score_gate.shape[-1]
        top_k_indices = torch.arange(num_experts, device=x.device)
        for i in range(len(score_gate.shape) - 1):
            top_k_indices = top_k_indices.unsqueeze(0)
        top_k_indices = top_k_indices.expand(*x.shape[:-1], num_experts)
        return {
            "topK_indices": top_k_indices,
            "topK_scores": score_gate,
            "balance_loss": None,
            "load": None,
            "importance": None,
            #
            "selected_keys": self.keys[top_k_indices],
        }


class SMoREGate(SMoREGateCommon):
    """
    Noisy sparse-k gate
    """

    def __init__(
        self, 
        dim_in: int,
        dim_hid: int,
        num_experts: int,
        num_active: int,
        act_fn,
        use_softmax=True,
        use_balance=True,
        balance_loss_weight=1e-2,
        add_noise=True,
        noise_epsilon=1e-2,
        dim_out: Optional[int]=None,
        gate_arch: str="mlp",
    ):
        super().__init__(
            dim_in=dim_in,
            dim_hid=dim_hid,
            num_experts=num_experts,
            num_active=num_active,
            act_fn=act_fn,
            gate_arch=gate_arch,
            dim_out=dim_out,
        )
        # -----
        self.use_softmax = use_softmax
        if self.num_active == 1:
            self.use_softmax = False
        self.softmax = nn.Softmax(-1)

        self.use_balance = use_balance
        self.balance_loss_weight = balance_loss_weight

        # add_noise
        self.add_noise = add_noise
        self.noise_epsilon = noise_epsilon
        self.warned = False
        if self.add_noise:
            self.weight_noise = nn.Linear(dim_in, num_experts, bias=False)
            self.weight_noise.weight.data = torch.zeros(
                (num_experts, dim_in),
                requires_grad=True,
                device=self.weight_noise.weight.data.device,
                dtype=self.weight_noise.weight.data.dtype,
            )
            self.mean = 0.0
            self.std = 1.0
            self.normal = Normal(self.mean, self.std)
            self.softplus = nn.Softplus()
        ######
        # TODO
        # 1. add temperature for gating score
        # 2. offset init key to be pos & neg / init all gate weights to 0 (Appendix A of ICLR 2017)
        self.reset_parameters()

    def forward(self, x, attention_mask_flat=None):
        q = F.normalize(self.mlp(x), p=2, dim=-1)
        k = F.normalize(self.keys, p=2, dim=-1)
        logits_gate = q@k.t()

        if self.training and self.add_noise:
            noise_mm = self.weight_noise(x)
            noise_control = self.softplus(noise_mm) + self.noise_epsilon
            logits_noise = torch.randn_like(logits_gate) * noise_control
            logits = logits_gate + logits_noise
        else:
            logits = logits_gate

        top_logits, top_indices = logits.topk(min(self.num_active + 1, self.num_experts), dim=-1)  # top k+1
        # top_k_logits: (batch * seq) x f2 x f1
        top_k_logits = top_logits[..., :self.num_active]
        top_k_indices = top_indices[..., :self.num_active]
        top_k_scores = self.softmax(top_k_logits.to(torch.float32)) if self.use_softmax else top_k_logits
        top_k_scores = top_k_scores.to(logits.dtype)

        # compute importance
        # e.g., 2L model, processed layer 2, and currently at layer 1:
        # logits: (batch * seq) x f2 x s1
        zeros = torch.zeros_like(logits, requires_grad=True, device=logits.device)
        scores_filtered = zeros.scatter(dim=-1, index=top_k_indices, src=top_k_scores)  # (batch_size, topk) scattered to (batch_size, num_experts)
        # Multiply scores_filtered with attention_mask here -- eos weights should not contribute to experts' load balance
        if attention_mask_flat is not None:
            assert len(attention_mask_flat.shape) == 2, "attention_mask_flat should have shape (batch_size, seq_len)"
            attention_mask_flat = attention_mask_flat.flatten()
            for i in range(len(scores_filtered.shape) - 1):
                attention_mask_flat = attention_mask_flat.unsqueeze(-1)
        else:
            attention_mask_flat = 1
        importance = (scores_filtered * attention_mask_flat).flatten(start_dim=0, end_dim=-2).sum(0)  # shape(num_experts): sum of expert i's score for the whole batch

        # compute load
        if self.training:
            if self.add_noise:
                assert self.num_active < self.num_experts
                threshold_if_in = top_logits[..., [-1]]
                is_in = torch.gt(logits, threshold_if_in)
                threshold_if_out = top_logits[..., [-2]]
                # prob_if_in & prob_if_out: for each expert i, re-sample noise. What's the prob of i being within top-K?
                # We need: noise_control * (logits_gate + X) >= threshold_if_in where X is random noise
                # So Prob(X >= (threshold_if_in - logits_gate) / noise_control) 
                # = Prob(X <= (logits_gate - threshold_if_in) / noise_control) 
                # = cdf((logits_gate - threshold_if_in) / noise_control)
                prob_if_in = self.normal.cdf((logits_gate - threshold_if_in) / noise_control)
                prob_if_out = self.normal.cdf((logits_gate - threshold_if_out) / noise_control)
                prob = torch.where(is_in, prob_if_in, prob_if_out)
                # as above, eos weights should not contribute to experts' load balance
                load = (prob * attention_mask_flat).flatten(start_dim=0, end_dim=-2).sum(0)
            else:
                load = ((scores_filtered * attention_mask_flat) > 0).flatten(start_dim=0, end_dim=-2).sum(0)
                if not self.add_noise and not self.warned:
                    warnings.warn('Gradient-trackable implementation for load calculation is only available when "add_noise=True". '
                                  'Training without noise will block the gradient from "load" path and lead to inconsistency in optimization objectives.')
                    self.warned = True
        else:
            load = ((scores_filtered * attention_mask_flat) > 0).flatten(start_dim=0, end_dim=-2).sum(0)

        if self.use_balance:
            balance_loss = cv_squared(importance) + cv_squared(load)
            balance_loss *= self.balance_loss_weight
        else:
            balance_loss = torch.tensor(-100.0, device=x.device)

        return {
            "topK_indices": top_k_indices,
            "topK_scores": top_k_scores,
            "balance_loss": balance_loss,
            "load": load,
            "importance": importance,
            #
            "selected_keys": k[top_k_indices],
        }


