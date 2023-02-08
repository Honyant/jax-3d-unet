from calendar import EPOCH
import functools
from random import random
from typing import Tuple
from config import *
import optax
from vit3d import *
import glob
import pickle
import equinox as eqx
from optax import adam, clip_by_global_norm, chain, apply_every, scale_by_schedule, warmup_cosine_decay_schedule
import numpy as np
from alive_progress import alive_bar
import wandb

BATCH_SIZE = 8
EPOCHS = 100
GRADIENT_ACCUMULATE_EVERY = 8
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 0.5
VALIDATE_EVERY = 100
SAMPLE_EVERY  = 500
WARMUP_LENGTH = 1000
DECAY_LENGTH = 10000
BASE_LR = 1e-6
MAX_LR = 1e-2
NUM_BATCHES=DECAY_LENGTH
DEVICES = 8

def mse_loss(input, target):
    input, target = input[1:], target[1:]
    input = input.flatten()
    target = target.flatten()
    intersection = jnp.sum(input * target)
    return 1 - (2. * intersection + 1) / (jnp.sum(input) + jnp.sum(target) + 1)

def get_files(dir):
    files = []
    for filename in glob.glob(dir+"*.npz"):
        files.append(filename)
    files.sort()
    return files


def load_data(files):
    x=[]
    y=[]
    for filename in files:
        data = jnp.load(filename)
        x.append(np.array([(np.clip(data["CT"],10,200)-10)/190,data["PT"]/10]))
        y.append(np.array(jax.nn.one_hot(data["label"],3,axis=0)))
    return np.array(x), np.array(y)

def generate_config(sizes):
    config=[]
    for i in range(len(sizes)-1):
        config.append([sizes[i],sizes[i+1],sizes[i+1],sizes[i+1]])
    return config, [sizes[-1],sizes[-1],sizes[-1]]


def split(arr):
  """Splits the first axis of `arr` evenly across the number of devices."""
  return np.transpose(arr.reshape(DEVICES, arr.shape[0] // DEVICES, *arr.shape[1:]), (1, 0) + tuple(range(2, arr.ndim + 1)))

#load everything into memory
files=get_files(processed_dir)
#shuffle
rng = np.random.RandomState(69)
rng.shuffle(files)
train_count=int((len(files)*0.8)//BATCH_SIZE)
train_files, val_files = files[:train_count*BATCH_SIZE], files[train_count*BATCH_SIZE:]

x, y = load_data(train_files)
#reshape into batches
x_val, y_val = load_data(val_files)


STEPS=x.shape[0]

sizes=[2,32,72,120,196,256,384,512]
key=jax.random.PRNGKey(69)
print("compiling")
config, config2 = generate_config(sizes)
model=VisionTransformer(config, config2, 3, key)
replicated_params = jax.tree_map(lambda x: jnp.array([x] * DEVICES), model)


#@eqx.filter_value_and_grad
def loss_fn(model, xs, ys):
    inp, labels = xs, ys
    logits = model(inp)
    return mse_loss(logits, labels)

schedule=warmup_cosine_decay_schedule(BASE_LR, MAX_LR, WARMUP_LENGTH, DECAY_LENGTH, BASE_LR)


@functools.partial(jax.pmap, axis_name='num_devices')
def update(params: VisionTransformer, xs: jnp.ndarray, ys: jnp.ndarray, lr: jnp.ndarray) -> Tuple[VisionTransformer, jnp.ndarray]:
  """Performs one SGD update step on params using the given data."""
  loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)
  grads = jax.lax.pmean(grads, axis_name='num_devices')
  loss = jax.lax.pmean(loss, axis_name='num_devices')
  new_params = jax.tree_map(
      lambda param, g: param - g * lr, params, grads)
  return new_params, loss


wandb.init(project="my-test-project")
wandb.config = {
  "learning_rate": BASE_LR,
  "epochs": NUM_BATCHES*BATCH_SIZE/STEPS,
  "batch_size": BATCH_SIZE,
}

counter=0

order=np.arange(STEPS)
with alive_bar(NUM_BATCHES, title='Training (batches)') as bar:
    for i in range(NUM_BATCHES):
        LEARNING_RATE = schedule(counter)
        idx = order[counter%STEPS]
        xs = x[idx]
        ys = y[idx]


        replicated_params, loss = update(replicated_params, xs, ys, jnp.array([LEARNING_RATE]* DEVICES))
        print(loss[0])
        counter+=1
        
        
        #validation
        if i%VALIDATE_EVERY==0:
            #get 8 random validation samples
            idx = np.random.choice(range(x_val.shape[0]),8,replace=False)
            xs = x_val[idx]
            throwaway, val_loss = update(replicated_params, xs, ys, jnp.array([LEARNING_RATE]* DEVICES))
            wandb.log({"loss": loss[0], "learning_rate": LEARNING_RATE, "val_loss": val_loss[0]})
        else:
            wandb.log({"loss": loss[0], "learning_rate": LEARNING_RATE})
        bar()

model = jax.device_get(jax.tree_map(lambda x: x[0], replicated_params))

#run on validation set
val_loss=sum([loss_fn(model, x_val[i], y_val[i]) for i in range(len(x_val))])/len(x_val)
print("Validation loss:",val_loss)
wandb.log({"val_loss": val_loss})


pickle.dump(model, open(model_dir+"model.pkl", "wb" ))