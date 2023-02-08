import os
import pickle

from config import *
dev = os.environ["dev"]

from utils import *

rng = np.random.RandomState(int(dev))
key = jax.random.PRNGKey(int(dev))

# load everything into memory
ctFiles, ptFiles, maskFiles = get_files(inDir)
files = list(zip(ctFiles, ptFiles, maskFiles))
rng.shuffle(files)

train_count = int((len(files) * 0.8)//3)
train_files, val_files = files[:train_count], files[train_count:]

x, y = load_data(train_files)
x_val, y_val = load_data(val_files)
#pickle x,y and x_val,y_val
if not os.path.exists(tmpfsDir):
    os.makedirs(tmpfsDir)
with open(tmpfsDir+"data.pkl", "wb") as f:
    pickle.dump((x,y,x_val,y_val), f)