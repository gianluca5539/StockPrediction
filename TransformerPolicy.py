import torch
from GTrXL import StableTransformerXL


class TransformerPolicy(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        n_transformer_layers=6,
        n_attn_heads=12,
        d_head_inner=256,
        d_ff_inner=2048,
    ):
        """
        Initialize a Policy with Transformer Encoder (GTrXL) as backbone.
        :param state_dim: (int) dimension of state
        :param act_dim: (int) dimension of action
        :param n_transformer_layers: (int) number of transformer layers
        :param n_attn_heads: (int) number of attention heads
        :param d_head_inner: (int) dimension of inner layer in attention head
        :param d_ff_inner: (int) dimension of inner layer in feed forward network
        NOTE - I/P Shape : [seq_len, batch_size, state_dim]
        """
        super(TransformerPolicy, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.transformer = StableTransformerXL(
            d_input=state_dim,
            n_layers=n_transformer_layers,
            action_space=act_dim,
            n_heads=n_attn_heads,
            d_head_inner=d_head_inner,
            d_ff_inner=d_ff_inner,
        )

        self.memory = None  # for storing the memory of transformer

    def forward(self, state):
        """Forward pass of the policy."""
        trans_state = self.transformer(state, self.memory)
        trans_state, self.memory = trans_state["logits"], trans_state["memory"]

        if self.act_dim != 1:
            # apply softmax to get probabilities
            layer_out = torch.softmax(trans_state, dim=-1)
        else:
            # apply tanh to get action
            layer_out = torch.tanh(trans_state)

        return layer_out


if __name__ == "__main__":
    states = torch.randn(20, 5, 80)  # seq_size, batch_size, dim - better if dim % 2 == 0
    print("=> Testing Policy")
    policy = TransformerPolicy(state_dim=states.shape[-1], act_dim=3)
    for i in range(10):
        act = policy(states)
        print("Action" + str(act))
