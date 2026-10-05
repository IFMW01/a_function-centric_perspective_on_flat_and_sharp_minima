import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn

# https://github.com/ShivamRajSharma/Vision-Transformer/blob/master/ImageTransformer.py


class SelfAttention(nn.Module):
    def __init__(self, embedding_dims, heads, dropout):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embedding_dims = embedding_dims
        self.head_dims = int(embedding_dims / heads)

        self.key = nn.Linear(self.head_dims, self.head_dims)
        self.query = nn.Linear(self.head_dims, self.head_dims)
        self.value = nn.Linear(self.head_dims, self.head_dims)

        self.fc = nn.Linear(self.head_dims * self.heads, self.embedding_dims)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        Batch = query.shape[0]

        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(Batch, query_len, self.heads, self.head_dims)
        key = key.reshape(Batch, key_len, self.heads, self.head_dims)
        value = value.reshape(Batch, value_len, self.heads, self.head_dims)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        attention_score = torch.einsum("bqhd,bkhd->bhqk", [query, key])

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float("-1e20"))

        attention_score = attention_score / ((self.head_dims) ** (1 / 2))
        attention_score = torch.softmax(attention_score, dim=-1)

        out = torch.einsum("bhqv,bvhd->bqhd", [attention_score, value]).reshape(
            Batch, query_len, self.heads * self.head_dims
        )

        out = self.dropout(self.fc(out))

        return out


class TransformerBlock(nn.Module):
    def __init__(
        self, embedding_dims, heads, dropout, forward_expansion, layer_norm_eps
    ):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.attention = SelfAttention(embedding_dims, heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dims, embedding_dims * forward_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dims * forward_expansion, embedding_dims),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        norm = self.layer_norm1(x)
        attention_block = self.attention(norm, norm, norm, mask)
        add = x + attention_block
        norm = self.layer_norm2(add)
        feed_forward = self.feed_forward(norm)
        out = feed_forward + add
        return out


class ViT(nn.Module):
    def __init__(
        self,
        patch_height,
        patch_width,
        max_len,
        embedding_dims,
        heads,
        forward_expansion,
        num_layers,
        dropout,
        layer_norm_eps,
        num_classes,
    ):
        super(ViT, self).__init__()

        self.vit_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dims, heads, dropout, forward_expansion, layer_norm_eps
                )
                for _ in range(num_layers)
            ]
        )
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dims))
        self.patch_embeddings = nn.Linear(embedding_dims, embedding_dims)
        self.postional_embedding = nn.Parameter(
            torch.zeros(1, max_len + 1, embedding_dims)
        )
        self.to_cls_token = nn.Identity()
        self.preclassifier = nn.Sequential(
            nn.LayerNorm(embedding_dims),
            nn.Linear(embedding_dims, num_classes * 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(num_classes * 4, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images, return_feat=False):
        patches = images.unfold(2, self.patch_height, self.patch_width).unfold(
            3, self.patch_height, self.patch_width
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(
            patches.shape[0],
            patches.shape[1],
            patches.shape[2],
            patches.shape[3] * patches.shape[4] * patches.shape[5],
        )
        patches = patches.view(patches.shape[0], -1, patches.shape[-1])

        x = self.cls_embedding.expand(patches.shape[0], -1, -1)
        patch_embeddings = self.patch_embeddings(patches)
        x = torch.cat((x, patch_embeddings), dim=1) + self.postional_embedding
        x = self.dropout(x)
        mask = None
        for block in self.vit_blocks:
            x = block(x, mask)
        x = self.to_cls_token(x[:, 0])
        features = self.preclassifier(x)
        out = self.classifier(features)

        if return_feat:
            return out, features.squeeze()
        return out


# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head**-0.5)))

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels=3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        return self.to_patch_tokens(x_with_shifts)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, (
            "Image dimensions must be divisible by the patch size."
        )

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, (
            "pool type must be either cls (cls token) or mean (mean pooling)"
        )

        self.to_patch_embedding = SPT(dim=dim, patch_size=patch_size, channels=channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, img, return_feat=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)

        features = self.mlp_head(x)
        out = self.classifier(features)

        if return_feat:
            return out, features.squeeze()
        return out
