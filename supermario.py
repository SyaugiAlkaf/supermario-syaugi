import gym
import gym_super_mario_bros
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Create the Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-v3')

# Get the dimensions of the game screen images
height, width, channels = env.observation_space.shape

# Get the number of actions
num_actions = env.action_space.n

# Define CNN model for Q-value estimation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_actions)  # Output layer, one neuron for each action
])

# Hyperparameters
learning_rate = 0.00025
optimizer = tf.keras.optimizers.Adam(learning_rate)
discount_factor = 0.99
exploration_prob = 1.0
exploration_decay = 0.995
min_exploration_prob = 0.1
num_episodes = 1000
replay_buffer = []

# Q-learning training
rewards_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Process the state through the CNN to get Q-value estimates
        q_values = model.predict(np.expand_dims(state, axis=0))

        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_values)

        next_state, reward, done, info = env.step(action)

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Update the Q-values using the replay buffer and Q-learning update rule

    rewards_per_episode.append(total_reward)
    exploration_prob = max(exploration_prob * exploration_decay, min_exploration_prob)

# Plot the rewards over episodes
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()
