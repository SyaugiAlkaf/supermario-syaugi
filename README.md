# Super Mario RL Agent with Deep Q-Networks

This repository contains code for training a Reinforcement Learning (RL) agent to play the game Super Mario Bros using Deep Q-Networks (DQN) algorithm.

## Technologies Used

- [gym](https://gym.openai.com/): A toolkit for developing and comparing reinforcement learning algorithms. In this project, it provides the environment for Super Mario Bros.

- [gym_super_mario_bros](https://github.com/Kautenja/gym-super-mario-bros): An extension of OpenAI Gym for Super Mario Bros environments.

- [numpy](https://numpy.org/): A powerful library for numerical computations in Python. It is used here for array manipulation and calculations.

- [matplotlib](https://matplotlib.org/): A popular visualization library in Python. It's used to plot training progress.

- [tensorflow](https://www.tensorflow.org/): An open-source machine learning framework. It's used to define, train, and manage the deep neural network.

## How It Works

1. The Super Mario Bros environment is created using `gym_super_mario_bros`.

2. A convolutional neural network (CNN) model is defined using `tensorflow.keras`. It's used to estimate Q-values, which represent the expected future rewards of taking different actions in different states.

3. The RL agent uses the DQN algorithm to learn the optimal policy for playing Super Mario Bros. The algorithm employs an exploration-exploitation strategy to balance between exploring new actions and exploiting learned knowledge.

4. The agent interacts with the environment, collects experiences, and stores them in a replay buffer.

5. The agent periodically samples experiences from the replay buffer and updates the Q-values using the Q-learning update rule.

6. Training progress is tracked by plotting the total rewards achieved per episode using `matplotlib`.

## How to Use

1. **Clone this repository:**

    ```bash
    git clone https://github.com/SyaugiAlkaf/supermario-syaugi.git
    cd your-repo
    ```

2. **Install the required dependencies using `pip` & Run the main training script:**

    ```bash
    pip install gym gym_super_mario_bros numpy matplotlib tensorflow
    python supermario.py
    ```

3. **Observe the training progress and the rewards plot that visualizes the agent's learning.**

## Note

- This repository provides an overview of training a Super Mario Bros-playing agent using DQN. For a detailed understanding of the implementation and results, refer to the code and comments in the main script.

- Feel free to contribute, experiment, and improve upon the code!

   
