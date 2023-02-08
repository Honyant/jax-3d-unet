import math
from typing import List

import jax
import jax.numpy as jnp
from jax import jit
from equinox import Module, nn, filter_jit

#written on 6/14/22 by anthony wang


def make_resolution_pyramid(x, num_tiers):
    tiers=[]
    tiers.append(x)
    for i in range(num_tiers - 1):
        #downscale to half resolution
        x=(x)
        tiers.append(x)
    return tiers

class ConvBlock(Module):

    seq: nn.Sequential
    point_conv2: nn.Conv #gotta do this bcs idk how to use this lib (how to use gelu in sequential)
    def __init__(
        self,
        first_kernel_size,
        in_size,
        hidden_size,
        out_size,
        key):

        depth_conv=nn.Conv(num_spatial_dims=3,
            in_channels=in_size,
            out_channels=in_size,
            kernel_size=first_kernel_size,
            groups=in_size,
            padding=first_kernel_size//2,
            key=key)

        point_conv=nn.Conv(num_spatial_dims=3,
            in_channels=in_size,
            out_channels=hidden_size,
            kernel_size=1,
            key=key)

        self.point_conv2=nn.Conv(num_spatial_dims=3,
            in_channels=hidden_size,
            out_channels=out_size,
            kernel_size=1,
            key=key)

        self.seq=nn.Sequential([depth_conv,
            nn.LayerNorm(None,elementwise_affine=False),#TODO: check if this is ok
            point_conv])
        
    def __call__(self, x):
        return self.point_conv2(
            jax.nn.gelu(
                self.seq(x)))+x



class TierBlock(Module):
    """
    A block of the TierNet with operations on every resolution
    input of n tiers of progressively halved resolution of 3d images
    output of n tiers of progressively halved resolution of 3d images
    """
    tier_convs: List[Module]
    point_convs: List[nn.Conv]
    image_dim: int
    def __init__(self, 
        image_dim,
        num_tiers, 
        first_kernel_size, 
        in_size, 
        hidden_size, 
        out_size, 
        key):

        self.image_dim=image_dim

        self.tier_convs=[]
        for i in range(num_tiers):
            self.tier_convs.append(ConvBlock(first_kernel_size,
                in_size,
                hidden_size,
                out_size//num_tiers,
                key))#TODO: i broke some shit so fix it later
            
        self.point_convs=[]
        for i in range(num_tiers):
            self.point_convs.append(nn.Conv(num_spatial_dims=3,
                in_channels=out_size,#we concat them
                out_channels=out_size,
                kernel_size=1,
                key=key))

        

    def __call__(self, x):
        out=[]
        num_tiers=len(self.tier_convs)
        #x is the pyramid of resolutions
        for i in range(num_tiers):
            out.append(self.tier_convs[i](x[i]))
        for i in range(num_tiers):
            components=[]
            for j in range(num_tiers):
                #skip if we are at the same resolution
                if i==j:
                    components.append(out[j])
                    continue
                #resize to the same resolution with jax.image.resize   "linear"
                components.append(jax.image.resize(out[j],out[i].shape, method="nearest"))
            #concatenate the components
            out[i]=jnp.concatenate(components,axis=0)
            #apply the point convolution
            out[i]=self.point_convs[i](out[i])
        return out


class TierNet(Module):
    """
    The TierNet architecture
    """
    tier_blocks: List[TierBlock]
    point_conv: nn.Conv
    point_conv_start: nn.Conv
    pooling: nn.AvgPool3D
    tier_num: int
    def __init__(self,
        start_channels,
        num_blocks,
        image_dim,
        num_tiers,
        first_kernel_size,
        in_size,
        hidden_size,
        out_size,
        key):


            self.tier_num=num_tiers
            self.tier_blocks=[TierBlock(image_dim,
                    num_tiers,
                    first_kernel_size,
                    in_size,
                    hidden_size,
                    out_size,
                    key) for i in range(num_blocks)]
            
            self.point_conv=nn.Conv(num_spatial_dims=3,
                    in_channels=out_size*num_tiers,#we concat them
                    out_channels=out_size,
                    kernel_size=1,
                    key=key)

            self.point_conv_start=nn.Conv(num_spatial_dims=3,
                    in_channels=start_channels,
                    out_channels=in_size,
                    kernel_size=1,
                    key=key)

            self.pooling=nn.AvgPool3D(kernel_size=3, stride=2, padding=1)
    #@filter_jit
    def __call__(self, x):
        x=self.point_conv_start(x)

        tiers=[]
        tiers.append(x)
        for i in range(self.tier_num - 1):
            #downscale to half resolution
            x=self.pooling(x)
            tiers.append(x)
        #start with the first convolution
        x=tiers
        for i in range(len(self.tier_blocks)-1):
            x=self.tier_blocks[i](x)
        #concatenate the components
        #here we do the final concatenation and point convolution into the final output
        components=[]
        for i in range(self.tier_num):
            components.append(jax.image.resize(x[i],x[0].shape, method="linear"))
        
        x=jnp.concatenate(components,axis=0)
        x=self.point_conv(x)
        x=jax.nn.sigmoid(x)
        return x