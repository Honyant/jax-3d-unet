import math
from typing import List, Tuple
import jax
import jax.numpy as jnp
from jax import jit
from equinox import Module, nn, static_field, filter_jit

class Embedding(Module):
    __doc__ = '\n    Takes an image and returns a list of patches of size patch_size x patch_size x patch_size\n    dimensions must be divisible by patch_size\n    '
    patch_size: int
    in_channels: int
    in_dim: int
    patchifier: nn.Conv
    shape = static_field()
    shape: List[int]

    def __init__(self, patch_size, in_channels, in_dim, key):
        self.patchifier = nn.Conv(num_spatial_dims=3, in_channels=in_channels,
          out_channels=(in_channels * patch_size ** 3),
          kernel_size=patch_size,
          stride=patch_size,
          groups=in_channels,
          key=key)
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.shape = [self.in_channels * self.patch_size ** 3, (self.in_dim // self.patch_size) ** 3]

    @jit
    def __call__(self, x):
        x = self.patchifier(x)
        return jnp.reshape(x, self.shape)


class TransformerBlock(Module):
    __doc__ = '\n    Standard block of Vision Transformer\n    '
    attention: nn.MultiheadAttention
    feed_forward: nn.Sequential
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    linear1: nn.Linear
    linear2: nn.Linear

    def __init__(self, num_heads, hidden_size, query_size, key):
        subkeys = jax.random.split(key, 4)
        self.attention = nn.MultiheadAttention(num_heads=num_heads, query_size=query_size, key=(subkeys[0]))
        self.feed_forward = nn.MLP(query_size, query_size, hidden_size, depth=1, key=(subkeys[1]))
        self.norm1 = nn.LayerNorm(None, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(None, elementwise_affine=False)
        self.linear1 = jax.vmap(nn.Linear(query_size, hidden_size, key=(subkeys[2])))
        self.linear2 = jax.vmap(nn.Linear(hidden_size, query_size, key=(subkeys[3])))

    @filter_jit
    def __call__(self, x):
        y = self.norm1(x)
        x = self.attention(y, y, y) + y
        y = self.norm2(x)
        y = jax.nn.relu(self.linear1(y))
        x = jax.nn.relu(self.linear2(y)) + x
        return x


class VisionTransformer(Module):
    __doc__ = '\n    Vision Transformer\n    '
    blocks: List[TransformerBlock]
    embedding: Embedding
    mlp: nn.MLP

    def __init__(self, num_heads, hidden_size, query_size, patch_size, in_channels, in_dim, num_blocks, out_size, key):
        subkeys = jax.random.split(key, num_blocks)
        self.embedding = Embedding(patch_size, in_channels, in_dim, key=key)
        self.blocks = [TransformerBlock(num_heads, hidden_size, query_size, key=subkey) for subkey in subkeys]
        self.mlp = nn.MLP(query_size, out_size, hidden_size, depth=1, key=key)

    @filter_jit
    def __call__(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        else:
            return self.mlp(x[0])
