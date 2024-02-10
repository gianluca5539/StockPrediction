import torch


class SelfAttentionPooling(torch.nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = torch.nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (seq_length, batch_size, d_model)

        attention_weight:
            att_w : size (batch_size, sequence_length, 1)

        return:
            utter_rep: size (batch_size, d_model)
        """
        # transpose from (seq_length, batch_size, d_model) to (batch_size, seq_length, d_model)
        batch_rep = batch_rep.transpose(0, 1)
        softmax = torch.nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim=1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        """Positional Embedding for self-attention.
        + Arguments
            - dim: int, dimension of the input."""
        super(PositionalEmbedding, self).__init__()

        self.dim = dim  # dimension of the input
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))  # inverse frequency (10000^(2i/d)
        self.register_buffer("inv_freq", inv_freq)  # register as buffer (not to be trained)

    def forward(self, positions):
        """Forward pass, the same as the original paper."""
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)  # shape: (seq, dim)
        return pos_emb[:, None, :]


class PositionwiseFF(torch.nn.Module):
    def __init__(self, d_input, d_inner, dropout):
        """Position-wise Feed-Forward Network.
        + Arguments
            - d_input: int, dimension of the input.
            - d_inner: int, dimension of the inner layer.
            - dropout: float, dropout rate.
        """
        super(PositionwiseFF, self).__init__()

        self.d_input = d_input  # dimension of the input
        self.d_inner = d_inner  # dimension of the inner layer
        self.dropout = dropout  # dropout rate
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_input, d_inner),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_inner, d_input),
            torch.nn.Dropout(dropout),
        )  # feed-forward network (2 layers, ReLU activation, 2 dropouts)

    def forward(self, input_):
        """Forward pass."""
        ff_out = self.ff(input_)
        return ff_out


class GatingMechanism(torch.nn.Module):
    def __init__(self, d_input, bg=0.1):
        """Gating Mechanism for the Transformer-XL architecture.
        + Arguments
            - d_input: int, dimension of the input.
            - bg: float, bias for the gating mechanism.
        """
        super(GatingMechanism, self).__init__()

        # Initializing linear layers for different gating components (check comments here)
        # Weight matrix for reset gate
        self.Wr = torch.nn.Linear(d_input, d_input)
        # Weight matrix for reset gate
        self.Ur = torch.nn.Linear(d_input, d_input)
        # Weight matrix for update gate
        self.Wz = torch.nn.Linear(d_input, d_input)
        # Weight matrix for update gate
        self.Uz = torch.nn.Linear(d_input, d_input)
        # Weight matrix for new memory gate
        self.Wg = torch.nn.Linear(d_input, d_input)
        # Weight matrix for new memory gate
        self.Ug = torch.nn.Linear(d_input, d_input)

        # Initializing bias for different gating components
        self.bg = bg

        # Initializing activation functions
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))  # reset gate
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)  # update gate
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # new memory gate
        g = torch.mul(1 - z, x) + torch.mul(z, h)  # gated output
        return g


class MultiHeadAttentionXL(torch.nn.Module):
    def __init__(self, d_input, d_inner, n_heads=4, dropout=0.1, dropouta=0.0):
        """Multi-Head Attention for Transformer-XL.
        + Arguments
            - d_input: int, dimension of the input.
            - d_inner: int, dimension of the inner layer.
            - n_heads: int, number of heads.
            - dropout: float, dropout rate.
            - dropouta: float, dropout rate for attention weights.
        """
        super(MultiHeadAttentionXL, self).__init__()

        self.d_input = d_input  # dimension of the input
        self.d_inner = d_inner  # dimension of the inner layer
        self.n_heads = n_heads  # number of heads

        # Linear transformation for keys & values for all heads at once for efficiency.
        # 2 for keys & values.
        self.linear_kv = torch.nn.Linear(d_input, (d_inner * n_heads * 2), bias=False)
        # for queries (will not be concatenated with memorized states so separate).
        self.linear_q = torch.nn.Linear(d_input, d_inner * n_heads, bias=False)

        self.linear_p = torch.nn.Linear(d_input, d_inner * n_heads, bias=False)  # for positional encoding

        self.scale = 1 / (d_inner**0.5)  # for scaled dot product attention
        # dropout for attention weights
        self.dropa = torch.nn.Dropout(dropouta)

        self.lout = torch.nn.Linear(d_inner * n_heads, d_input, bias=False)  # output linear layer
        self.dropo = torch.nn.Dropout(dropout)  # dropout for output

    def _rel_shift(self, x):
        """IDK what this does.
        + Arguments
            - x: torch.FloatTensor, shape - (seq, bs, n_heads, d_inner) = (20, 5, 4, 8)
        """
        # x shape: [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        return (
            torch.cat([zero_pad, x], dim=1)  # concatenate with zero padding
            # remove the first row
            .view(x.size(1) + 1, x.size(0), *x.size()[2:])[1:].view_as(x)  # reshape to original shape
        )

    def forward(self, input_, pos_embs, memory, u, v, mask=None):
        """
        + pos_embs: positional embeddings passed separately to handle relative positions.
        + Arguments
            - input: torch.FloatTensor, shape - (seq, bs, self.d_input) = (20, 5, 8)
            - pos_embs: torch.FloatTensor, shape - (seq + prev_seq, bs, self.d_input) = (40, 1, 8)
            - memory: torch.FloatTensor, shape - (prev_seq, b, d_in) = (20, 5, 8)
            - u: torch.FloatTensor, shape - (num_heads, inner_dim) = (3 x )
            - v: torch.FloatTensor, shape - (num_heads, inner_dim)
            - mask: torch.FloatTensor, Optional = (20, 40, 1)

        + Returns
            - output: torch.FloatTensor, shape - (seq, bs, self.d_input)

        + Symbols representing shape of the tensors
            - cs: current sequence length, b: batch, H: no. of heads
            - d: inner dimension, ps: previous sequence length
        """
        cur_seq = input_.shape[0]  # current sequence length
        prev_seq = memory.shape[0]  # previous sequence length
        H, d = self.n_heads, self.d_inner  # no. of heads, inner dimension
        # concat memory across sequence dimension
        # input_with_memory = [seq + prev_seq x B x d_input] = [40 x 5 x 8]
        input_with_memory = torch.cat([memory, input_], dim=0)

        # k_tfmd, v_tfmd = [seq + prev_seq x B x n_heads.d_head_inner], [seq + prev_seq x B x n_heads.d_head_inner]
        # tfmd = transformed
        k_tfmd, v_tfmd = torch.chunk(
            self.linear_kv(input_with_memory),
            2,
            dim=-1,
        )
        # q_tfmd = [seq x B x n_heads.d_head_inner] = [20 x 5 x 96]
        q_tfmd = self.linear_q(input_)

        _, bs, _ = q_tfmd.shape  # batch size
        assert bs == k_tfmd.shape[1]  # batch size of q_tfmd and k_tfmd (so also v_tfmd) should be same

        # content_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        # attn = attention
        content_attn = torch.einsum(  # einsum is Einstein summation
            "ibhd,jbhd->ijbh",  # this means multiply the last two dimensions of the first tensor with the first two dimensions of the second tensor
            (
                (q_tfmd.view(cur_seq, bs, H, d) + u),
                k_tfmd.view(cur_seq + prev_seq, bs, H, d),
            ),
        )

        # p_tfmd: [seq + prev_seq x 1 x n_heads.d_head_inner] = [40 x 1 x 96]
        # positional embeddings are same for all batches so we add an extra dimension
        p_tfmd = self.linear_p(pos_embs)
        # position_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        position_attn = torch.einsum(
            "ibhd,jhd->ijbh",  # this means multiply the last two dimensions of the first tensor with the first two dimensions of the second tensor
            (
                (q_tfmd.view(cur_seq, bs, H, d) + v),
                p_tfmd.view(cur_seq + prev_seq, H, d),
            ),
        )

        position_attn = self._rel_shift(position_attn)  # shift position attention
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = content_attn + position_attn  # add content and position attention together

        if mask is not None and mask.any().item():  # if mask is not None and mask is not all False
            # fills float('-inf') where mask is True.
            attn = attn.masked_fill(mask[..., None], -float("inf"))
        # rescale to prevent values from exploding.
        # normalize across the value sequence dimension.
        attn = torch.softmax(attn * self.scale, dim=1)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = self.dropa(attn)

        # attn_weighted_values = [curr x B x n_heads.d_inner] = [20 x 5 x 96]
        attn_weighted_values = (
            torch.einsum(
                "ijbh,jbhd->ibhd",
                (
                    attn,  # (cs, cs + ps, b, H)
                    v_tfmd.view(cur_seq + prev_seq, bs, H, d),  # (cs + ps, b, H, d)
                ),
            )  # (cs, b, H, d)
            .contiguous()  # we need to change the memory layout to make `view` work
            .view(cur_seq, bs, H * d)
        )  # (cs, b, H * d)

        # output = [curr x B x d_input] = [20 x 5 x 8]
        output = self.dropo(self.lout(attn_weighted_values))
        return output


class StableTransformerEncoderLayerXL(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_input: int,
        d_head_inner: int,
        d_ff_inner: int,
        dropout: float,
        gating=True,
        dropouta=0.0,
    ) -> None:
        """Transformer Encoder Layer with Gating Mechanism.
        + Parameters
            - n_heads: int, number of heads
            - d_input: int, input dimension
            - d_head_inner: int, inner dimension of each head
            - d_ff_inner: int, inner dimension of feedforward layer
            - dropout: float, dropout rate
            - gating: bool, whether to use gating mechanism
            - dropouta: float, dropout rate for attention
        """
        super(StableTransformerEncoderLayerXL, self).__init__()

        self.gating = gating
        self.gate1 = GatingMechanism(d_input)
        self.gate2 = GatingMechanism(d_input)
        self.mha = MultiHeadAttentionXL(
            d_input,
            d_head_inner,
            n_heads=n_heads,
            dropout=dropout,
            dropouta=dropouta,
        )
        self.ff = PositionwiseFF(d_input, d_ff_inner, dropout)
        self.norm1 = torch.nn.LayerNorm(d_input)
        self.norm2 = torch.nn.LayerNorm(d_input)

    def forward(
        self,
        input_: torch.Tensor,
        pos_embs: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        mask=None,
        mems=None,
    ) -> torch.Tensor:
        """Forward pass of Transformer Encoder Layer.
        + Parameters
            - input_: torch.Tensor, input tensor
            - pos_embs: torch.Tensor, positional embeddings
            - u: torch.Tensor, u vector
            - v: torch.Tensor, v vector
            - mask: torch.Tensor, mask tensor
            - mems: torch.Tensor, memory tensor
        + Returns
            - src: torch.Tensor, output tensor
        """
        src2 = self.norm1(input_)  # layer normalization
        src2 = self.mha(src2, pos_embs, mems, u, v, mask=mask)  # multi-head attention
        src = self.gate1(input_, src2) if self.gating else input_ + src2  # gating mechanism
        src2 = self.ff(self.norm2(src))  # layer normalization + feedforward
        src = self.gate2(src, src2) if self.gating else src + src2  # gating mechanism
        return src


class StableTransformerXL(torch.nn.Module):
    def __init__(
        self,
        d_input: int,
        n_layers: int,
        n_heads: int,
        d_head_inner: int,
        d_ff_inner: int,
        dropout=0.1,
        dropouta=0.0,
        mem_len=100,
        action_space=3,
    ):
        """Stable Transformer XL.
        + Parameters
            - d_input: int, input dimension
            - n_layers: int, number of layers
            - n_heads: int, number of heads
            - d_head_inner: int, inner dimension of each head
            - d_ff_inner: int, inner dimension of feedforward layer
            - dropout: float, dropout rate
            - dropouta: float, dropout rate for attention weights
            - mem_len: int, memory length
        """

        super(StableTransformerXL, self).__init__()

        (self.n_layers, self.n_heads, self.d_input, self.d_head_inner, self.d_ff_inner,) = (
            n_layers,
            n_heads,
            d_input,
            d_head_inner,
            d_ff_inner,
        )  # model parameters

        # positional embeddings for input
        self.pos_embs = PositionalEmbedding(d_input)
        self.drop = torch.nn.Dropout(dropout)  # dropout layer
        self.mem_len = mem_len  # memory length
        self.layers = torch.nn.ModuleList(  # list of Transformer Encoder Layers
            [
                StableTransformerEncoderLayerXL(  # Transformer Encoder Layer
                    n_heads,
                    d_input,
                    d_head_inner=d_head_inner,
                    d_ff_inner=d_ff_inner,
                    dropout=dropout,
                    dropouta=dropouta,
                )
                for _ in range(n_layers)
            ]
        )

        # linear layers to go from [batch_size,d_inner] to [batch_size,action_space]
        self.linear1 = torch.nn.Linear(d_input, 32)
        # TODO dropout?
        self.linear2 = torch.nn.Linear(32, action_space)

        #! u and v are global parameters: maybe changing these to per-head parameters might help performance?
        self.u, self.v = (
            # [n_heads x d_head_inner] = [3 x 32]
            torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head_inner)),  # u
            torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head_inner)),  # v
        )

        # initialize selfattentionpooling
        self.attention_pooling = SelfAttentionPooling(d_input)

    def init_memory(self, device=torch.device("cpu")):
        """Initialize memory.
        + Parameters
            - device: torch.device, device to use
        + Returns
            - memory: List[torch.FloatTensor], list of memory tensors
        """
        return [
            torch.empty(0, dtype=torch.float).to(device)  # empty tensor for memory initialization
            for _ in range(self.n_layers + 1)  # for each layer
        ]

    def update_memory(self, previous_memory, hidden_states):
        """Update memory.
        + Parameters
            - previous_memory: List[torch.FloatTensor], list of memory tensors
            - hidden_states: List[torch.FloatTensor], list of hidden state tensors
        + Returns
            - new_memory: List[torch.FloatTensor], list of updated memory tensors
        """

        assert len(hidden_states) == len(previous_memory)  # check length
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)
        # mem_len, seq_len = 3, hidden_states[0].size(0)
        # print(mem_len, seq_len)

        with torch.no_grad():  # no gradient calculation
            new_memory = []  # new memory
            end_idx = mem_len + seq_len  # end index
            beg_idx = max(0, end_idx - self.mem_len)  # begin index
            for m, h in zip(previous_memory, hidden_states):  # for each layer
                # concatenate memory and hidden state
                cat = torch.cat([m, h], dim=0)
                # append to new memory
                new_memory.append(cat[beg_idx:end_idx].detach())
        return new_memory

    def forward(self, inputs, memory=None):
        """Forward pass of Stable Transformer XL.
        + Parameters
            - inputs: torch.Tensor, input tensor
            - memory: List[torch.FloatTensor], list of memory tensors
        + Returns
            - outputs: torch.Tensor, output tensor
            - memory: List[torch.FloatTensor], list of updated memory tensors
        """
        if memory is None:  # if memory is not given
            memory = self.init_memory(inputs.device)  # initialize memory
        assert len(memory) == len(self.layers) + 1  # check length is correct

        # current sequence length and batch size
        cur_seq, bs = inputs.shape[:2]
        prev_seq = memory[0].size(0)  # previous sequence length

        # dec_attn_mask = [curr x curr + prev x 1] = [20 x 40 x 1]
        dec_attn_mask = (  # attention mask for decoder
            torch.triu(
                torch.ones((cur_seq, cur_seq + prev_seq)),
                diagonal=1 + prev_seq,
            )
            .bool()[..., None]
            .to(inputs.device)
        )

        pos_ips = torch.arange(cur_seq + prev_seq - 1, -1, -1.0, dtype=torch.float).to(inputs.device)
        # pos_embs = [curr + prev x 1 x d_input] = [40 x 1 x 8]
        pos_embs = self.drop(self.pos_embs(pos_ips))  # positional embeddings
        if self.d_input % 2 != 0:  # if input dimension is odd
            pos_embs = pos_embs[:, :, :-1]  # remove last dimension

        hidden_states = [inputs]
        layer_out = inputs  # layer output
        # for each layer and memory tensor
        for mem, layer in zip(memory, self.layers):
            # layer_out = [curr x B x d_inner] = [20 x 5 x 8]
            layer_out = layer(  # Transformer Encoder Layer
                layer_out,
                pos_embs,
                self.u,
                self.v,
                mask=dec_attn_mask,
                mems=mem,
            )
            hidden_states.append(layer_out)  # append to hidden states

        # Memory is treated as a const., don't propagate through it
        # new_memory = [[T x B x d_inner] x 4]
        memory = self.update_memory(memory, hidden_states)  # update memory

        # perform pooling to transform layer_out from [seq_len x batch_size x d_input] to [batch_size x d_input]
        # layer_out = layer_out.mean(dim=0)  # mean pooling
        layer_out = self.attention_pooling(layer_out)

        # use linear layers to go from [batch_size,d_inner] to [batch_size,action_space]
        layer_out = self.linear1(layer_out)
        layer_out = torch.tanh(layer_out)
        layer_out = self.linear2(layer_out)

        return {
            "logits": layer_out,
            "memory": memory,
        }  # return logits and memory as dict
