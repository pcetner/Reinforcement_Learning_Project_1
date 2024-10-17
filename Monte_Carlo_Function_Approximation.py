import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import mplcyberpunk
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle

# Define the neural network architecture for function approximation
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)     # First hidden layer
        self.fc2 = nn.Linear(64, 64)            # Second hidden layer
        self.fc3 = nn.Linear(64, output_dim)    # Output layer

    def forward(self, x):
        # Forward pass through the network with ReLU activation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Return output

# Epsilon-greedy action selection
def choose_action(state, model, epsilon, action_space):
    if np.random.rand() < epsilon:
        return action_space.sample()  # Select a random action with probability epsilon
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)    # Convert state to tensor
            return torch.argmax(model(state_tensor)).item()         # Select the action with highest Q-value

# Normalize state inputs to a standard range
def normalize_state(state):
    cart_pos, cart_vel, pole_angle, pole_vel = state
    normalized_state = [
        cart_pos / 2.4,           # Normalize cart position
        cart_vel / 3.0,           # Normalize cart velocity
        pole_angle / 0.20943951,  # Normalize pole angle
        pole_vel / 3.0            # Normalize pole velocity
    ]
    return np.array(normalized_state, dtype=np.float32)

# Adjust learning rate based on average reward over recent episodes
def get_learning_rate(mean_reward):
    """As training progresses and mean reward increases, dramatically lower the learning rate."""
    if mean_reward < 200:
        return 0.001
    elif 200 <= mean_reward < 300:
        return 0.0001
    elif 300 <= mean_reward < 400:
        return 0.00001
    else:
        return 0.000001  

# Monte Carlo method with function approximation
def monte_carlo(run_id, num_episodes, gamma=0.99, epsilon_start=1.0, epsilon_decay_rate=0.0001, min_epsilon=0.01, max_total_reward=500):
    env = gym.make('CartPole-v1')  # Create the CartPole environment
    action_space = env.action_space.n
    input_dim = env.observation_space.shape[0]              # Number of continuous state variables
    model = QNetwork(input_dim, action_space)               # Initialize the Q-Network
    optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam optimizer for training
    criterion = nn.MSELoss()                                # Mean Squared Error loss function

    rewards_per_episode = []    # Track total rewards for each episode
    epsilon = epsilon_start     # Initial epsilon value

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()  # Reset environment and get initial state
        episode_memory = []     # Memory to store episode transitions
        done = False
        total_reward = 0

        # Run an episode
        while not done:
            normalized_state = normalize_state(state)                                   # Normalize the state
            action = choose_action(normalized_state, model, epsilon, env.action_space)  # Select action
            next_state, reward, terminated, truncated, _ = env.step(action)             # Take action in the environment
            total_reward += reward                                                      # Accumulate total reward

            # Handle episode termination conditions
            if total_reward >= max_total_reward:
                terminated = True
                truncated = False

            done = terminated or truncated

            episode_memory.append((normalized_state, action, reward))  # Store transition in episode memory
            state = next_state  # Update to next state

        rewards_per_episode.append(total_reward)  # Store total reward for this episode

        # Calculate mean of the last 100 rewards
        if len(rewards_per_episode) >= 100:
            mean_last_100 = np.mean(rewards_per_episode[-100:])
        else:
            mean_last_100 = np.mean(rewards_per_episode)

        # Adjust the learning rate based on recent performance
        new_lr = get_learning_rate(mean_last_100)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        # Calculate returns and update the model based on episode memory
        G = 0
        visited = set()  # Track visited state-action pairs
        for state, action, reward in reversed(episode_memory):
            G = gamma * G + reward      # Compute return
            state_tuple = tuple(state)  # Convert state to tuple for hashing
            if (state_tuple, action) not in visited:  # Check if state-action pair was visited
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                target = model(state_tensor).clone().detach()
                target[0][action] = G  # Update only the action taken

                # Perform a training step
                optimizer.zero_grad()                           # Clear previous gradients
                loss = criterion(model(state_tensor), target)   # Compute loss
                loss.backward()                                 # Backpropagate the loss
                optimizer.step()                                # Update model parameters

                visited.add((state_tuple, action))  # Mark this state-action pair as visited

        # Print progress at regular intervals
        if episode % 200 == 0:
            avg_reward = np.mean(rewards_per_episode[-200:])  # Average reward over last 200 episodes
            print(f'Run {run_id} - Episode: {episode}/{num_episodes}, Epsilon: {epsilon:.4f}, '
                  f'Average Reward: {avg_reward:.2f}, Mean Last 100 Rewards: {mean_last_100:.2f}, '
                  f'Learning Rate: {new_lr:.6f}')

        # Decay epsilon to reduce exploration over time
        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)

    env.close()  # Close the environment after training

    # Save the trained model to a file
    folder_path = 'monte_carlo_function_approximation_models'   # Directory to save models
    os.makedirs(folder_path, exist_ok=True)                     # Create directory if it doesn't exist
    file_name = f'monte_carlo_function_approximation_run{run_id}.pkl'
    file_path = os.path.join(folder_path, file_name)

    # Use pickle to save the model
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

    print(f'Model for run {run_id} saved to {file_path}.')

    # Calculate moving average of rewards over the last 200 episodes
    if len(rewards_per_episode) >= 200:
        average_rewards = np.convolve(rewards_per_episode, np.ones(200)/200, mode='valid')
    else:
        average_rewards = np.array(rewards_per_episode)

    print(f'Run {run_id} completed.')
    return average_rewards  # Return the computed average rewards

# Function to initialize and run Monte Carlo for a given run
def run_monte_carlo_wrapper(run_id, num_episodes):
    return monte_carlo(run_id, num_episodes)  

# Main execution with multiprocessing to run multiple simulations in parallel
if __name__ == '__main__':
    num_episodes = 11000    # Total number of episodes for training
    num_runs = 7            # Number of independent runs

    # Create a pool of worker processes for parallel execution
    with mp.Pool(processes=num_runs) as pool:
        # Prepare arguments for each run
        args = [(run, num_episodes) for run in range(1, num_runs + 1)]
        # Execute runs in parallel and collect results
        all_average_rewards = pool.starmap(run_monte_carlo_wrapper, args)

    # Plotting all average rewards from different runs
    palette = sns.color_palette("flare", num_runs)                                  # Set color palette for plots
    plt.style.use("cyberpunk")                                                      # Set plotting style
    plt.figure(figsize=(12, 8))                                                     # Create a new figure for plotting
    for i, avg_rewards in enumerate(all_average_rewards):
        plt.plot(avg_rewards, color=palette[i % len(palette)], label=f'Run {i+1}')  # Plot each run's average rewards
    plt.xlabel('Episodes')                                                          # X-axis label
    plt.ylabel('Average Reward (over 200 episodes)')                                # Y-axis label
    plt.title('Average Reward Tracking in Monte Carlo Function Approximation')      # Plot title
    plt.legend()                                                                    # Show legend
    plt.grid(True)                                                                  # Enable grid
    plt.tight_layout()                                                              # Adjust layout
    plt.savefig('average_rewards_monte_carlo_with_function_approximation.png')      # Save plot to file
    plt.show()                                                                      # Display plot
