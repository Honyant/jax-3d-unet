from tiernet import *
key=jax.random.PRNGKey(1)

model=TierNet(num_blocks=3,
    start_channels=2,
    image_dim=128,
    num_tiers=4,#should be at max log2(image_dim)-1 i think
    first_kernel_size=3,
    in_size=32,
    hidden_size=64,
    out_size=32,
    key=key)
#65.64526557922363
#average loop time: 0.6956079053878784
print("done compiling")


#estimate average loop time
import time
start=time.time()
print(model(jax.random.normal(key,(2,128,128,128))))
print(time.time()-start)
start=time.time()

times=[]
for i in range(100):
        model(jax.random.normal(key,(2,128,128,128)))
print("average loop time:",(time.time()-start)/100)
