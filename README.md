# Deep Q-Network (DQN) for CartPole

This repository contains a PyTorch implementation of the Deep Q-Network (DQN) algorithm applied to the CartPole-v1 environment from OpenAI Gym. The code trains an agent to balance a pole on a moving cart by approximating the Q-function with a neural network.

## How to Run

0. **Set up Virtual Environment** (Not necessary)
   ```bash
   conda create -n py312env python=3.12
   conda activate py312env
   conda install jupyter
   conda install pip
   python -m ipykernel install --user --name py312env --display-name "Python 3.12 (py312env)"
   ```

1. **Clone this Repository**
   ```bash
   git clone https://github.com/thenewgoat/cartpole_DQN.git
   cd cartpole_DQN
   ```
2. **Open Jupyter Notebook**
   On a separate terminal, run jupyter notebook
   ```bash
   jupyter notebook
   ```
3. **Open the Notebook**\
   Search for the repository at http://localhost:8888/tree \
   Open the notebook in this repository and select "Python 3.12 (py312env)" as the kernel.
2. **Run the Notebook**\
   Click on Run > Run All Cells

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
   - A fraction tau (e.g., 0.005) of the main network parameters are blended into the target network parameters, smoothing updates and preventing instability. \
   `θ⁻ ← τ * θ + (1 - τ) * θ⁻`

5. **Hyperparameters**
   - Learning rate (e.g., 1e-3)
   - Discount factor gamma (e.g., 0.98)
   - Epsilon decay schedule
   - Soft update rate tau
   - Batch size (e.g., 64)
