from dataclasses import dataclass, field, asdict
from torch import nn

@dataclass
class ParamStatsSMoRE:
    trainable_param: int = field(default=0)
    # trainable_param_by_layer = expert_param_by_layer + gate_param_by_layer
    expert_param: int = field(default=0)
    expert_param_wo_proj: int = field(default=0)
    gate_param: int = field(default=0)
    # change w. expert activation pattern. Profile stats during training. can do either
    # 1. profile stats during training (based on activation pattern)
    # 2. compute upper bound
    active_param: int = field(default=0)
    active_param_wo_proj: int = field(default=0)
    # 
    expert_flops: int = field(default=0)  # just count multiplication
    expert_flops_wo_proj: int = field(default=0)
    gate_flops: int = field(default=0)

    @classmethod
    def accum_param(cls, adapter_name, *params):
        """
        Will not double count under parameter sharing
        """
        counter = 0
        for p in params:
            pa = p[adapter_name]
            counter += sum([])
            if isinstance(pa, nn.ParameterList):
                for pai in {id(t): t for t in pa}.values():  # orig: pa
                    if pai is not None:
                        counter += pai.numel()
            elif isinstance(pa, nn.Parameter):
                counter += pa.numel()
            elif isinstance(pa, nn.Module):
                counter += sum([v.numel() for v in pa.parameters() if v.requires_grad])
        return counter

    def int(self):
        self.trainable_param = int(self.trainable_param)
        self.expert_param = int(self.expert_param)
        self.expert_param_wo_proj = int(self.expert_param_wo_proj)
        self.gate_param = int(self.gate_param)
        self.active_param = int(self.active_param)
        self.active_param_wo_proj = int(self.active_param_wo_proj)
        self.expert_flops = int(self.expert_flops)
        self.expert_flops_wo_proj = int(self.expert_flops_wo_proj)
        self.gate_flops = int(self.gate_flops)

    @classmethod
    def agg_stats(cls, *stats):
        all_stats = {
            "trainable_param": 0,
            "expert_param": 0,
            "expert_param_wo_proj": 0,
            "gate_param": 0,
            "active_param": 0,
            "active_param_wo_proj": 0,
            "expert_flops": 0,
            "expert_flops_wo_proj": 0,
            "gate_flops": 0,
        }
        for stat in stats:
            for n in all_stats:
                all_stats[n] += getattr(stat, n)
        return cls(**all_stats)

    def asdict(self, unit="G"):
        scale_down = 1
        if unit is None:
            unit = ""
        if unit.lower() == "g":
            scale_down = 1e9
        elif unit.lower() == "m":
            scale_down = 1e6
        elif unit.lower() == "k":
            scale_down = 1e3
        ret = asdict(self)
        if scale_down == 1:
            return ret
        else:
            return {f"{k}_{unit.upper()}": v / scale_down for k, v in ret.items()}
