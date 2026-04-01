import torch
from torch import nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.w = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=True,
            # device=device,
            # dtype=dtype,
        )

        nn.init.trunc_normal_(self.w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.w, x, "out in,... in -> ... out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # (vocab_size, d_model)
        self.embedding_matrix = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim), requires_grad=True
        )

        nn.init.trunc_normal_(self.embedding_matrix)

    # x:(batch_size, sequence_length) -> (batch_size, sequence_length,d_model)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.gain = nn.Parameter(requires_grad=True)
        self.d_model = d_model
        self.eps = eps

    # (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dypte = x.dtype
        x = x.to(dtype=torch.float32)

        result = (
            x
            / torch.sqrt(
                torch.sum(x * x, dim=-1, keepdim=True) / self.d_model + self.eps
            )
            * self.gain
        )

        return result.to(dtype=in_dypte)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_ff, d_model)
        self.w2 = Linear(d_model, d_ff)
        self.w3 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2((SiLU(self.w1(x)) * self.w3(x)))


def SiLU(x):
    return x * torch.sigmoid(x)


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        inv_freq = 1 / theta ** (torch.arange(0, d_k, 2, device=device) / d_k)
        pos = torch.arange(max_seq_len, device=device)
        angles = einsum(pos, inv_freq, "i,j->i j")

        self.register_buffer("sin", angles.sin(), persistent=False)
        self.register_buffer("cos", angles.cos(), persistent=False)

    # x (..., seq_len, d_k), token_positions (..., seq_len)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        x_even_rotated = x1 * cos - x2 * sin
        x_odd_rotated = x1 * sin + x2 * cos

        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1).flatten(-2)

        return x_rotated


def softmax(x: torch.Tensor, dim: int):
    exp_x = torch.exp(x - torch.max(x))
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


# q,k (batch_size, ..., seq_len, d_k),v (batch_size, ..., seq_len, d_v),mask(seq_len,seq_len).->(batch_size,..., d_v)
def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask
):
    return einsum(
        softmax(
            (
                einsum(q, k, "... n d_k, ... m d_k -> ... n m") / q.size(-1) ** 0.5
            ).masked_fill(~mask, float("-inf")),
            -1,
        ),
        v,
        "... n m,... m d_v -> ... n d_v",
    )


class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        d_k = d_v = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        q = rearrange(self.q_proj(x), "b s (h d_k) -> b h s d_k", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b s (h d_k) -> b h s d_k", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b s (h d_v) -> b h s d_v", h=self.num_heads)

        seq_len = q.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len)).to(dtype=torch.bool))

        o = scaled_dot_product_attention(q, k, v, causal_mask)

        return self.o_proj(rearrange(o, "b h s d_v -> b s (h d_v)"))


class multihead_self_attention_with_rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int):
        super().__init__()

        d_k = d_v = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

        self.rope = RoPE(theta, d_k, max_seq_len)

    def forward(
        self, x: torch.Tensor, token_positions
    ) -> torch.Tensor:

        q = rearrange(self.q_proj(x), "b s (h d_k) -> b h s d_k", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b s (h d_k) -> b h s d_k", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b s (h d_v) -> b h s d_v", h=self.num_heads)

        seq_len = q.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len)).to(dtype=torch.bool))

        # token_positions = torch.arange(seq_len)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        o = scaled_dot_product_attention(q, k, v, causal_mask)

        return self.o_proj(rearrange(o, "b h s d_v -> b s (h d_v)"))
