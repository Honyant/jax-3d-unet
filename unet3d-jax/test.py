from FUNet3D import *
import pickle
from utils import *
import jax
import numpy as np
import matplotlib.pyplot as plt
from config import *

rng = np.random.RandomState(0)
key = jax.random.PRNGKey(0)

# load everything into memory
ctFiles, ptFiles, maskFiles = get_files(normal_data_dir)
files = list(zip(ctFiles, ptFiles, maskFiles))

cctFiles, cptFiles, cmaskFiles = get_files(coarse_data_dir)
cfiles = list(zip(cctFiles, cptFiles, cmaskFiles))

data=load_data(files[:10])
data_coarse=load_data(cfiles[:10])

#print max and min for both [0]

print("Max normal:", np.max(data[0][0]))
print("Min normal:", np.min(data[0][0]))
print("Max coarse:", np.max(data_coarse[0][0]))
print("Min coarse:", np.min(data_coarse[0][0]))


#do the same but average over the batch

dd=[]
for i in range(10):
        ex=[]
        ex.append(np.max(data[0][i]))
        ex.append(np.min(data[0][i]))
        ex.append(np.max(data_coarse[0][i]))
        ex.append(np.min(data_coarse[0][i]))
        dd.append(ex)

#calc avg
print("Avg Max normal:", np.mean(np.array(dd)[:,0]))
print("Avg Min normal:", np.mean(np.array(dd)[:,1]))
print("Avg Max coarse:", np.mean(np.array(dd)[:,2]))
print("Avg Min coarse:", np.mean(np.array(dd)[:,3]))


#print(n.shape)


def dice_loss(input, target):
        input, target = input[1:], target[1:]
        input = input.flatten()
        target = target.flatten()
        intersection = jnp.sum(input * target)
        return 1 - (2. * intersection + 0.0001) / (jnp.sum(input) + jnp.sum(target) +  0.0001)

def test_dice_loss():
        model=FUNet3D([2,16,32,64,256,512,1028,3], 3, key)
        out=model(x)[2]
        n=np.argmax(out,axis=0)
        dice_loss(out,y)
        plt.hist(np.array(out[2][0]).flatten(),bins=100)


