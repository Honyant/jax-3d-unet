from jax import numpy as jnp
from jax import random as jrand
import numpy as np
import jax
import glob
import nibabel as nib
from FUNet3D import *
import functools
from typing import Tuple

#flip the 3dimage with a 50% chance for axis 1, 2, and 3
def random_flip(image, label, rng):
    flips=[]
    if rng.rand() > .5:
        flips.append(1)
    if rng.rand() > .5:
        flips.append(2)
    if rng.rand() > .5:
        flips.append(3)
    flips=tuple(flips)
    if flips != ():
        image = jnp.flip(image, flips)
        label = jnp.flip(label, flips)
    return image, label

#3,128,128,128
def random_rotate(image, label, rng):
    angle = rng.randint(0,4)
    axes = tuple(np.random.choice([1,2,3], 2, replace=False))
    image = jnp.rot90(image, angle, axes)
    label = jnp.rot90(label, angle, axes)
    return image, label

def random_noise(image, rng, key):
    std=rng.normal()*0.1
    image = image + jrand.normal(key,image.shape)*std
    return image

def random_scale(image, rng):
    scale = rng.normal()*0.2+1
    image = image*jnp.clip(scale, 0.85, 1.15)
    return image

def random_translate(image, label, rng):
    x = rng.randint(-10,10)
    y = rng.randint(-10,10)
    z = rng.randint(-10,10)
    image = jnp.roll(image, x, axis=1)
    image = jnp.roll(image, y, axis=2)
    image = jnp.roll(image, z, axis=3)
    label = jnp.roll(label, x, axis=1)
    label = jnp.roll(label, y, axis=2)
    label = jnp.roll(label, z, axis=3)
    return image, label


def random_transform(image, label, rng):
    key=jrand.PRNGKey(rng.randint(0, 2**32))
    probs=[1,1,0.7,0.7]
    r = rng.rand()
    if r < probs[0]:
        image, label = random_flip(image, label, rng)
    r = rng.rand()
    if r < probs[1]:
        image, label = random_rotate(image, label, rng)
    r = rng.rand()
    if r < probs[2]:
        image = random_noise(image, rng, key)
    r = rng.rand()
    if r < probs[3]:
        image = random_scale(image, rng)
    return [image, label]




def dice_loss(input, target):
    input, target = input[1:], target[1:]
    input = input.flatten()
    target = target.flatten()
    intersection = jnp.sum(input * target)
    return 1 - (2.0 * intersection + 1) / (jnp.sum(input) + jnp.sum(target) + 1)


def get_files(dir):
    files = []
    for filename in glob.glob(dir + "*.nii.gz"):
        files.append(filename)
    files.sort()
    ctFiles=[f for f in files if "ct" in f]
    ptFiles=[f for f in files if "pt" in f]
    maskFiles=[f for f in files if "mask" in f]
    return ctFiles, ptFiles, maskFiles


def load_data(files):
    x = []
    y = []
    for filename in files:
        ct = nib.load(filename[0]).get_fdata()
        pt = nib.load(filename[1]).get_fdata()
        mask = nib.load(filename[2]).get_fdata()
        #stack ct and pt with concat
        x.append(np.array([ct, pt]))
        y.append(np.array(jax.nn.one_hot(mask, 3, axis=0)))
    return np.array(x), np.array(y)

def loss_fn(model, xs, ys):
    inp, labels = xs, ys
    logits = model(inp)
    loss = (dice_loss(logits[2], labels) 
    + 0.5 * dice_loss(logits[1], jax.image.resize(labels, logits[1].shape, "bilinear")) 
    + 0.25 * dice_loss(logits[0], jax.image.resize(labels, logits[0].shape, "bilinear")))/1.75
    loss += sum(
        l2_loss(w, alpha=0.009) 
        for w in jax.tree_leaves(model)
    )
    return loss

def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()
