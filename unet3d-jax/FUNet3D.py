from typing import List
import jax
from jax import jit, numpy as jnp
from equinox import Module, nn, static_field
from utils import *
# written on 6/23/22 by anthony wang
class Down(Module):
    down: nn.Conv
    norm1: nn.GroupNorm
    conv2: nn.Conv
    norm2: nn.GroupNorm
    "Downscale, then single convolution"

    def __init__(self, in_channels, out_channels, key):
        keys = jax.random.split(key, 2)
        self.down = nn.Conv(3, in_channels, out_channels, 3, 2, 1, key=keys[0])
        self.conv2 = nn.Conv(3, out_channels, out_channels, 3, 1, 1, key=keys[1])
        self.norm1 = nn.GroupNorm(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(out_channels, out_channels)

    def __call__(self, x):
        x1 = x
        x = self.norm1(self.down(x))
        return jax.nn.leaky_relu(self.norm2(self.conv2(x))), x1


class Up(Module):
    up: nn.ConvTranspose
    norm2: nn.GroupNorm
    conv2: nn.Conv
    norm3: nn.GroupNorm
    conv3: nn.Conv
    "Upscale, then double convolution"

    def __init__(self, in_channels, out_channels, key):
        keys = jax.random.split(key, 3)
        self.up = nn.ConvTranspose(3, in_channels, out_channels, 2, 2, key=keys[0])
        self.conv2 = nn.Conv(3, out_channels * 2, out_channels, 3, 1, 1, key=keys[1])
        self.conv3 = nn.Conv(3, out_channels, out_channels, 3, 1, 1, key=keys[2])
        self.norm2 = nn.GroupNorm(out_channels, out_channels)
        self.norm3 = nn.GroupNorm(out_channels, out_channels)

    def __call__(self, x1, x2):
        x = jnp.concatenate([self.up(x1), x2], axis=0)
        x = self.norm2(self.conv2(x))
        return jax.nn.leaky_relu(self.norm3(self.conv3(x)))


class FUNet3D(Module):
    down: List[Module]
    up: List[Module]
    conv1: nn.Conv
    norm1: nn.GroupNorm
    conv2: nn.Conv
    norm2: nn.GroupNorm
    convOuts: List[Module]
    out_depth: int = static_field()
    """Dims consists of the # of channels for input, then layers sorted by depth from least to greatest, then ouput
    Out depth is the number of layers that give an output"""

    def __init__(self, dims, out_depth, key):
        keys = jax.random.split(key, 3)
        in_channels = dims.pop(0)
        out_channels = dims.pop()
        self.conv1 = nn.Conv(3, in_channels, dims[0], 3, 1, 1, key=keys[0])
        self.norm1 = nn.GroupNorm(dims[0],dims[0])
        self.conv2 = nn.Conv(3, dims[0], dims[0], 3, 1, 1, key=keys[1])
        self.norm2 = nn.GroupNorm(dims[0],dims[0])
        keys = jax.random.split(keys[0], len(dims * 2) - 2)
        self.down = []
        self.up = []
        self.convOuts = []
        for i in range(len(dims) - 1):
            self.down.append(Down(dims[i], dims[i + 1], keys[i]))
        for i in range(len(dims) - 1):
            self.up.append(Up(dims[i + 1], dims[i], keys[i + len(dims)]))
        self.up.reverse()
        keys = jax.random.split(keys[0], out_depth)
        for i in range(out_depth):
            self.convOuts.append(nn.Conv(3, dims[i], out_channels, 1, 1, key=keys[i]))
        self.convOuts.reverse()
        self.out_depth = out_depth

    @jit
    def __call__(self, x):
        x = jax.nn.leaky_relu(self.norm1(self.conv1(x)))
        x = jax.nn.leaky_relu(self.norm2(self.conv2(x)))
        stack = []
        for i in range(len(self.down)):
            x, x1 = self.down[i](x)
            stack.append(x1)
        out = []
        for i in range(len(self.up)):
            if i >= len(self.up) - self.out_depth:
                x = self.up[i](x, stack.pop())
                out.append(jax.nn.sigmoid(self.convOuts[i + self.out_depth - len(self.up)](x)))
            else:
                x = self.up[i](x, stack.pop())
        return out

if __name__ == "__main__":
    def loss_fn2(model, xs, ys):
        inp, labels = xs, ys
        logits = model(inp)
        loss = dice_loss(logits[0], labels)
        return loss
    arr=[8,16,32,48,128,256]
    arr.insert(0,2)
    arr.append(3)
    key=jax.random.PRNGKey(0)
    model=FUNet3D(arr,3,key)
    x=jax.random.normal(key,(2,256,256,256))
    y=jax.random.normal(key,(3,256,256,256))
    import time
    start=time.time()
    print(loss_fn(model,x,y))
    loss, grads = jax.vmap(jax.value_and_grad(loss_fn)(model, x, y))
    print(time.time()-start)
    start=time.time()
    times=[]
    key=jax.random.PRNGKey(0)
    keys=jax.random.split(key,10)
    for i in range(10):
        x=jax.random.normal(keys[i],(2,128,128,128))
        y=jax.random.normal(keys[i],(3,128,128,128))
        loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    print("average loop time:",(time.time()-start)/10)