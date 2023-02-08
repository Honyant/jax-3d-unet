import pickle as pkl
import os
from optax import warmup_cosine_decay_schedule
import numpy as np
from alive_progress import alive_bar
import wandb
from utils import *
from FUNet3D import *
from jax.experimental.compilation_cache import compilation_cache as cc
from config import *


cc.initialize_cache(cache_path)

dev = os.environ["dev"]


BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
SAMPLE_EVERY = 500
WARMUP_LENGTH = 2e3#2e3 if int(dev) < 4 else 8e2
DECAY_LENGTH = 4e4#2e4 if int(dev) < 4 else 8e3
BASE_LR = 1e-6
MAX_LR  = [2e-2,4e-2,8e-2,1.6e-1,0.025,0.05,0.1,0.2]

MAX_LR = MAX_LR[int(dev)]
NUM_BATCHES = int(DECAY_LENGTH)
DEVICES = 8
DECAY = 1e-4

rng = np.random.RandomState(int(dev))
key = jax.random.PRNGKey(int(dev))

x, y, x_val, y_val = pkl.load(open(DATA_DIR+"data.pkl", "rb"))

print("epochs:", int((BATCH_SIZE * NUM_BATCHES)//x.shape[0]))
sizes = [2,16,32,64,256,512,1028,3] if int(dev) < 4 else [2,2,4,8,32,64,128,3]
print("Config:", sizes)
model = FUNet3D(sizes, 3, key)
replicated_params = jax.tree_map(lambda x: jnp.array([x] * DEVICES), model)
schedule = warmup_cosine_decay_schedule(
    BASE_LR, MAX_LR, WARMUP_LENGTH, DECAY_LENGTH, BASE_LR)

wandb.init(project="my-test-project")
wandb.config = {
    "batch size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning rate": LEARNING_RATE,
    "validate every": VALIDATE_EVERY,
    "warmup length": WARMUP_LENGTH,
    "decay length": DECAY_LENGTH,
    "base lr": BASE_LR,
    "max lr": MAX_LR,
    "num batches": NUM_BATCHES,
    "devices": DEVICES,
    "decay": DECAY
}

def get_transformed_data(idx):
    xs = [jnp.array(x[id]) for id in idx]
    ys = [jnp.array(y[id]) for id in idx]
    output = [random_transform(img, lbl, np.random.RandomState(
        rng.randint(0, 2**32))) for img, lbl in zip(xs, ys)]
    xs = [img[0] for img in output]
    ys = [img[1] for img in output]
    return xs, ys


@functools.partial(jax.pmap, axis_name="num_devices")
def update(params: FUNet3D, xs: jnp.ndarray, ys: jnp.ndarray, lr: jnp.ndarray) -> Tuple[FUNet3D, jnp.ndarray]:
    """Performs one SGD update step on params using the given data."""
    loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)
    grads = jax.lax.pmean(grads, axis_name="num_devices")
    loss = jax.lax.pmean(loss, axis_name="num_devices")
    new_params = jax.tree_map(lambda param, g: param - g * lr, params, grads)
    return new_params, loss



with alive_bar(NUM_BATCHES, title="Training (batches)") as bar:
    for i in range(NUM_BATCHES):
        LEARNING_RATE = schedule(i)
        idx = rng.choice(range(x.shape[0]), 8, replace=False)
        xs, ys = get_transformed_data(idx)
        replicated_params, loss = update(replicated_params, jnp.array(
            xs), jnp.array(ys), jnp.array([LEARNING_RATE] * DEVICES))
        # validation
        if i % VALIDATE_EVERY == 0:
            # get 8 random validation samples
            idx = rng.choice(range(x_val.shape[0]), 8, replace=False)
            throwaway, val_loss = update(replicated_params, jnp.array(
                x_val[idx]), jnp.array(y_val[idx]), jnp.array([LEARNING_RATE] * DEVICES))
            wandb.log(
                {"loss": loss[0], "learning_rate": LEARNING_RATE, "val_loss": val_loss[0]})
        else:
            wandb.log({"loss": loss[0], "learning_rate": LEARNING_RATE})
        bar.text = "loss: " + str(loss[0])
        bar()

model = jax.device_get(jax.tree_map(lambda x: x[0], replicated_params))

# run on validation set
val_loss = sum([loss_fn(model, x_val[i], y_val[i])
               for i in range(len(x_val))]) / len(x_val)
print("Validation loss:", val_loss)
wandb.log({"val_loss": val_loss})
# create outdir if doesn't exist
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

pkl.dump(model, open(OUT_DIR + "model_coarse.pkl" if coarse else "model.pkl", "wb"))
