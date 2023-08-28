import gym
import gym_super_mario_bros

# Create the Mario environment
env = gym.make('SuperMarioBros-1-1-v0')
env.reset()
# Choose an action (e.g., 0 = no-op)
action = 0

# Perform one step in the environment and print the output
observation, reward, done, info = env.step(action)
print("Observation:", observation)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)


env.close()
