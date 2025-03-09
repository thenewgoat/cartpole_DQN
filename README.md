# Deep Q-Network (DQN) for CartPole

This repository contains a PyTorch implementation of the Deep Q-Network (DQN) algorithm applied to the CartPole-v1 environment from OpenAI Gym. The code trains an agent to balance a pole on a moving cart by approximating the Q-function with a neural network.

## Features

- **Discrete Action Space**: CartPole has two discrete actions (move left or move right).
- **Replay Buffer**: A `ReplayBuffer` class stores transitions `(state, action, reward, next_state, done)` and samples them in minibatches, breaking the correlation between sequential observations.
- **Target Network with Soft Updates**: Maintains a separate target network (`targetQNet`) that slowly tracks the main network (`QNet`) via a soft update rule. This helps stabilize training.
- **Epsilon-Greedy Exploration**: Uses an ε-greedy policy to balance exploration (random actions) and exploitation (greedy actions according to the Q-network).
- **Experience-Driven Training**: Samples minibatches from the replay buffer to perform gradient descent steps that reduce the temporal difference error.

## Main Components

1. **QNetwork**  
   A simple feedforward neural network with:
   - Two hidden layers (e.g., 24 units each) using ReLU activation.
   - An output layer that produces Q-values for each discrete action.

2. **ReplayBuffer**  
   - Stores recent transitions in a fixed-size buffer (e.g., 1,000,000 transitions).
   - Supports random sampling of minibatches for training.

3. **Training Loop**  
   - Runs for a specified number of episodes (e.g., 500–1,000).  
   - In each episode:
     1. Reset the environment.
     2. For each time step, choose an action via ε-greedy.
     3. Step the environment, store the transition in the replay buffer.
     4. If the buffer has enough samples, sample a minibatch and update the Q-network.
     5. Soft-update the target network every few steps.

4. **Soft Update**
   - Target Network partially updated every few steps with parameters from the Main Network, using the following equation:
   `θ⁻ ← τ * θ + (1 - τ) * θ⁻`
