import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import mplcyberpunk
import os
import torch
import random
from torch import nn
from collections import deque
import seaborn as sns
from multiprocessing import Pool
import pickle

# Hyperparameters for the DQN training process
EPSILON_INIT = 1.0          # Initial exploration rate
LEARNING_RATE = 0.00003     # Initial learning rate
DECAY_VAL = 0.000005        # Rate at which epsilon decays
MEMORY_SIZE = 5000          # Size of the replay memory
BATCH_SIZE = 64             # Number of experiences to sample for training
GAMMA = 0.99                # Discount factor for future rewards
MAX_STEPS = 500             # Maximum steps per episode

# Minimum learning rate for dynamic adjustment
LR_MIN = 1e-6  

# Directory for saving trained models and logs
SAVE_DIR = 'non_linear_function_approximation_DQN_models'
os.makedirs(SAVE_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# Function to create the Gym environment for training
def create_environment():
    return gym.make("CartPole-v1")

# DQN Neural Network Model
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        # Define the network architecture
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)  # Forward pass through the network

# Function to compute the Temporal Difference (TD) loss
def compute_td_loss(model, replay_buffer, optimizer, batch_size):
    # Check if there's enough memory to sample from
    if len(replay_buffer) < batch_size:
        return None

    # Sample a batch of experiences from the replay buffer
    state, next_state, reward, done, action = zip(*random.sample(replay_buffer, batch_size))
    state_tensor = torch.stack(list(state)).float()             # Convert states to tensor
    next_state_tensor = torch.stack(list(next_state)).float()   # Convert next states to tensor
    reward_tensor = torch.tensor(reward).float()                # Convert rewards to tensor
    done_tensor = torch.tensor(done).float()                    # Convert done flags to tensor
    action_tensor = torch.tensor(action).long()                 # Convert actions to tensor

    # Calculate Q-values for current and next states
    q_values = model(state_tensor)
    next_q_values = model(next_state_tensor)

    # Gather the Q-values for the taken actions
    q_vals = q_values.gather(dim=-1, index=action_tensor.unsqueeze(-1)).squeeze(-1)
    max_next_q_values = torch.max(next_q_values, dim=-1)[0].detach()  # Max Q-value for next state

    # Compute the TD loss
    loss = ((reward_tensor + GAMMA * max_next_q_values * (1 - done_tensor) - q_vals) ** 2).mean()

    optimizer.zero_grad()   # Clear previous gradients
    loss.backward()         # Backpropagate to compute gradients
    optimizer.step()        # Update model parameters
    return loss.item()      # Return the computed loss

# Epsilon-greedy action selection function
def choose_action(state, model, epsilon, action_space):
    if np.random.uniform(0, 1) < epsilon:
        return action_space.sample()  # Select a random action
    with torch.no_grad():
        return np.argmax(model(torch.tensor(state, dtype=torch.float32)).numpy())  # Select the best action

# Function to adjust learning rate based on moving average reward
def adjust_learning_rate(optimizer, avg_reward):
    """As training progresses and mean reward increases, dramatically lower the learning rate."""
    if avg_reward < 200:
        lr = 0.00005
    elif avg_reward < 300:
        lr = 0.00001
    elif avg_reward < 400:
        lr = 0.000005
    else:
        lr = 0.000001

    # Update the learning rate in the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Main training loop for the DQN agent
def train_dqn(run_num, num_episodes):
    env = create_environment()  # Create the Gym environment
    model = DQNModel(env.observation_space.shape[0], env.action_space.n)        # Instantiate the model
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)   # Set up the optimizer
    replay_buffer = deque(maxlen=MEMORY_SIZE)  # Initialize replay buffer
    
    total_rewards = []      # Store rewards for each episode
    total_losses = []       # Store losses for each episode
    epsilon = EPSILON_INIT  # Set initial exploration rate

    last_50_rewards = deque(maxlen=50)  # Track recent rewards for moving average

    current_lr = LEARNING_RATE  # Initialize learning rate

    # Main training loop
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()  # Reset the environment for a new episode
        done = False
        steps = 0
        eps_reward = 0          # Initialize episode reward
        eps_loss = 0            # Initialize episode loss

        # Episode loop
        while not done and steps < MAX_STEPS:
            action = choose_action(state, model, epsilon, env.action_space)     # Choose an action
            next_state, reward, done, truncated, info = env.step(action)        # Execute action

            # Store experience in replay buffer
            replay_buffer.append((torch.tensor(state, dtype=torch.float32), 
                                  torch.tensor(next_state, dtype=torch.float32), 
                                  reward, 
                                  done, 
                                  action))

            # Sample and compute TD loss if enough experiences are available
            if len(replay_buffer) > BATCH_SIZE:
                loss = compute_td_loss(model, replay_buffer, optimizer, BATCH_SIZE)
                if loss is not None:
                    eps_loss += loss  # Accumulate episode loss

            # Decay epsilon for exploration
            epsilon = max(0, epsilon - DECAY_VAL)
            eps_reward += reward    # Accumulate reward for the episode
            state = next_state      # Transition to the next state
            steps += 1              # Increment step count

        total_rewards.append(eps_reward)    # Record total reward for the episode
        last_50_rewards.append(eps_reward)  # Update recent rewards
        total_losses.append(eps_loss)       # Record total loss for the episode

        # Calculate moving average reward for the last 50 episodes
        if len(last_50_rewards) == 50:
            avg_recent_reward = np.mean(last_50_rewards)

            # Adjust the learning rate based on the average reward
            current_lr = adjust_learning_rate(optimizer, avg_recent_reward)

        # Check for early termination based on average reward
        if len(last_50_rewards) == 50 and np.mean(last_50_rewards) >= 500:
            print(f"Run {run_num} - Early termination at episode {episode} due to average reward >= 500.")
            break

        # Print statistics every 10 episodes
        if episode % 10 == 0:
            avg_recent_reward = np.mean(last_50_rewards) if len(last_50_rewards) == 50 else 0
            stats = (f"Run {run_num} - Episode: {episode}/{num_episodes}, "
                     f"Current Epsilon: {epsilon:.4f}, "
                     f"Latest Reward: {eps_reward}, "
                     f"Average Reward (last 50): {avg_recent_reward:.2f}, "
                     f"Loss: {eps_loss:.4f}, "
                     f"Learning Rate: {current_lr:.8f}")
            print(stats)  # Output training progress

    return total_rewards, total_losses  # Return accumulated rewards and losses
    
# Function to run multiple training instances in parallel
def run_parallel_training(num_runs, episodes):
    with Pool(num_runs) as pool:
        results = pool.starmap(train_dqn, [(run_num + 1, episodes) for run_num in range(num_runs)])
    return results

# Plot function to visualize the average rewards across all runs
def plot_all_average_rewards(all_rewards, num_runs, window=100):
    # Calculate average rewards over specified window size
    all_average_rewards = [
        np.convolve(rewards, np.ones(window) / window, mode='valid')
        for rewards in all_rewards
    ]

    palette = sns.color_palette("flare", num_runs)
    plt.style.use("cyberpunk")  # Apply a custom plotting style
    plt.figure(figsize=(12, 8))

    # Plot each run's average rewards
    for i, avg_rewards in enumerate(all_average_rewards):
        plt.plot(avg_rewards, color=palette[i], label=f'Run {i + 1}')

    plt.xlabel('Episodes')                                          # X-axis label
    plt.ylabel(f'Average Reward (over {window} episodes)')          # Y-axis label
    plt.title('Average Reward Tracking in DQN Over Multiple Runs')  # Plot title
    plt.legend()                                                    # Display legend
    plt.grid(True)                                                  # Enable grid
    plt.tight_layout()                                              # Adjust layout
    plt.savefig('DQN_Average_Rewards_Multiple_Runs.png')            # Save plot as an image
    plt.show()                                                      # Display the plot

if __name__ == '__main__':
    num_runs = 7        # Define number of parallel training runs
    episodes = 15000    # Total episodes per run
    results = run_parallel_training(num_runs, episodes)  # Execute parallel training

    # Extract and plot average rewards from all runs
    all_rewards = [rewards for rewards, _ in results]   # Unpack rewards from results
    plot_all_average_rewards(all_rewards, num_runs)     # Visualize average rewards
