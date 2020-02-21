import gym
import time
import numpy as np

env = gym.make('Breakout-v0')
env.reset()

print(env.env.ale.getScreenRGB2().mean())

# env.step(1)

# for _ in range(10):
#     env.render()
#     env.step(np.random.randint(2, 4))
#     time.sleep(0.1)

env.close()
