
import torch.nn as nn
import torch.nn.functional as F
import torch
from cutile.ops.matmul import launch_matmul
from cutile.ops.silu_and_mul import launch_silu_and_mul


class MyQwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=torch.float16)
        assert config.hidden_act in ["silu"], "Unsupported activation function"
        self.register_buffer("fused_gate_up_weight", None)
        # self.act_fn = ACT2FN[config.hidden_act]

    def _initialize_fused_weights(self):
        """Initialize the fused weight from individual gate_proj and up_proj weights."""
        with torch.no_grad():
            # Concatenate gate_proj.weight and up_proj.weight along output dimension
            # gate_proj.weight: [intermediate_size, hidden_size]
            # up_proj.weight: [intermediate_size, hidden_size]
            # fused_weight: [2 * intermediate_size, hidden_size]
            self.fused_gate_up_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)

    def init_weights(self):
        self.gate_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)


    def forward(self, x):
        if self.fused_gate_up_weight is None:
            self._initialize_fused_weights()
        batch_size = x.size(0)
        xv = x.view(-1, x.size(-1))
        linear = torch.zeros((xv.size(0), 2*self.intermediate_size), device=x.device, dtype=torch.float16)
        launch_matmul(xv, self.fused_gate_up_weight, linear, transb=True, act=0)
        v0 = torch.zeros((xv.size(0), self.intermediate_size), device=x.device, dtype=torch.float16)
        launch_silu_and_mul(linear, v0)
        finalout = torch.zeros((xv.size(0), self.hidden_size), device=x.device, dtype=torch.float16)
        launch_matmul(v0, self.down_proj.weight, finalout, transb=True, act=0)
        return finalout.view(batch_size, -1, self.hidden_size)
        # down_proj = self.down_proj(v0.view(batch_size, -1, self.intermediate_size))
        # return down_proj


if __name__ == "__main__":
    import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
    # Simple test
    class Config:
        hidden_size = 128
        intermediate_size = 64
        hidden_act = "silu"

    config = Config()
    mlp = MyQwen2MLP(config).cuda()
    x = torch.rand((8, 16, 128), device='cuda', dtype=torch.float16)
    output = mlp(x)
    mlp2 = qwen2_mod.Qwen2MLP(config).cuda()
    mlp2.gate_proj = mlp.gate_proj
    mlp2.up_proj = mlp.up_proj
    mlp2.down_proj = mlp.down_proj
    output2 = mlp2(x)
    torch.testing.assert_close(output, output2, rtol=1e-5, atol=1e-4)
    print("✓ MyQwen2MLP test passed!")