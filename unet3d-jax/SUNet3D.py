from typing import List

import jax
import jax.numpy as jnp
from jax import jit
from equinox import Module, nn

#written on 6/19/22 by anthony wang

class DoubleConvBlock(Module):
    depth_conv: nn.Conv
    point_conv: nn.Conv
    seq: nn.Sequential

    def __init__(
        self,
        first_kernel_size,
        in_size,
        hidden_size,
        out_size,
        key):
        keys=jax.random.split(key,3)

        self.depth_conv=nn.Conv(num_spatial_dims=3,
            in_channels=in_size,
            out_channels=hidden_size,
            kernel_size=first_kernel_size,
            padding=first_kernel_size//2,
            key=keys[0])

        self.point_conv=nn.Conv(num_spatial_dims=3,
            in_channels=hidden_size,
            out_channels=out_size,
            kernel_size=1,
            key=keys[2])

        self.seq=nn.Sequential([self.depth_conv,
            nn.LayerNorm(None,elementwise_affine=False),#TODO: check if this is ok
            self.point_conv])
        
    def __call__(self, x):

        return jax.nn.leaky_relu(self.seq(x))

class DownConvBlock(Module):
    block: DoubleConvBlock
    dilated_conv: nn.Conv
    """
    Double convblock followed by a dilated conv
    """
    def __init__(self,
        first_kernel_size,
        in_size,
        hidden_size_1,
        hidden_size_2,
        out_size,
        key):
        keys=jax.random.split(key,2)


        self.block = DoubleConvBlock(first_kernel_size,
            in_size,
            hidden_size_1,
            hidden_size_2,
            key=keys[0])

        self.dilated_conv = nn.Conv(num_spatial_dims=3,
            in_channels=hidden_size_2,
            out_channels=out_size,
            kernel_size=3,
            stride=2,
            padding=1,
            key=keys[1])
        
    def __call__(self, x):
        x = self.block(x)
        return self.dilated_conv(x), x

class UpConvBlock(Module):
    block: DoubleConvBlock
    transposed_conv: nn.ConvTranspose
    reduce_point_conv: nn.Conv
    """
    Transposed conv, merge with skip connection, then a double conv block
    """
    def __init__(self,
        first_kernel_size,
        in_size,
        hidden_size_1,
        hidden_size_2,
        out_size,
        key):
        keys=jax.random.split(key,3)

        self.block = DoubleConvBlock(first_kernel_size,
            hidden_size_2,#double bcs of concat but we do a depthwise conv so its ok
            hidden_size_2,
            out_size,
            keys[0])
        
        self.transposed_conv = nn.ConvTranspose(num_spatial_dims=3,
            in_channels=in_size,
            out_channels=hidden_size_1,
            kernel_size=2,
            stride=2,
            key=keys[1])

        self.reduce_point_conv = nn.Conv(num_spatial_dims=3,#reduce from hidden_size_1*2 to hidden_size_2
            in_channels=hidden_size_1*2,
            out_channels=hidden_size_2,
            kernel_size=1,
            key=keys[2])

    def __call__(self, x, y):
        return self.block(
            self.reduce_point_conv(
            jnp.concatenate([self.transposed_conv(x), y], axis=0)))



class SUNet3D(Module):
    downLayers: List[Module]
    upLayers: List[Module]
    double_conv: Module
    """
    3DUNet with more customizability for the layers
    """
    def __init__(self, config, config2, output, key):
        self.downLayers = []
        self.upLayers = []
        keys = jax.random.split(key, len(config*2)+1)
        for i in range(len(config)):
            self.downLayers.append(DownConvBlock(5,*config[i], keys[i]))
        for i in range(len(config)):
            #if last layer to have a different size
            if i==0:
                layerConfig=list(reversed(config[i]))
                layerConfig=layerConfig[:-1]
                layerConfig.append(output)
                self.upLayers.append(UpConvBlock(3,*layerConfig, keys[i+len(config)]))
            else:
                self.upLayers.append(UpConvBlock(3,*list(reversed(config[i])), keys[i+len(config)]))
        self.upLayers.reverse()
        self.double_conv=DoubleConvBlock(3,*config2, keys[-1])
   
    def __call__(self, x):
        stack=[]
        for i in range(len(self.downLayers)):
            x, y = self.downLayers[i](x)
            stack.append(y)
        x=self.double_conv(x)
        for i in range(len(self.upLayers)):
            x = self.upLayers[i](x, stack.pop())
        return jax.nn.sigmoid(x)