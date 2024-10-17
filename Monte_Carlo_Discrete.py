import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from collections import defaultdict
import seaborn as sns
import mplcyberpunk
import pickle
import argparse
import os

# Default functions for Q-table entries and returns
def default_q_value():
    """Default factory for Q-table entries."""
    return np.zeros(2)  #  CartPole has 2 actions

def default_return_sum():
    """Default factory for returns_sum."""
    return np.zeros(2)

def default_return_count():
    """Default factory for returns_count."""
    return np.zeros(2)

# Discretizes the continuous state space into bins
def discretize_state(state, bins):
    """Discretizes the continuous state variables of the environment."""
    cart_pos, cart_vel, pole_angle, pole_vel = state
    # Define the ranges for each state variable
    cart_pos_bins = np.linspace(-2.4, 2.4, bins[0] - 1)
    cart_vel_bins = np.linspace(-3.0, 3.0, bins[1] - 1)
    pole_angle_bins = np.linspace(-0.20943951, 0.20943951, bins[2] - 1)  # ~12 degrees
    pole_vel_bins = np.linspace(-3.0, 3.0, bins[3] - 1)
    return tuple([
        np.digitize(cart_pos, cart_pos_bins),
        np.digitize(cart_vel, cart_vel_bins),
        np.digitize(pole_angle, pole_angle_bins),
        np.digitize(pole_vel, pole_vel_bins)
    ])

# Implements epsilon-greedy action selection strategy
def choose_action(state, Q, epsilon, action_space):
    """Selects an action based on the epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return action_space.sample()  # Select a random action
    else:
        return np.argmax(Q[state])  # Select the best action from Q-table

# Implements the first-visit Monte Carlo method
def monte_carlo(run_id, num_episodes, gamma=1, epsilon_start=.99, epsilon_decay_rate=0.00002, min_epsilon=0, max_total_reward=500, bins=(15, 15, 15, 15)):
    """Runs the Monte Carlo algorithm for CartPole."""
    env = gym.make('CartPole-v1')
    action_space = env.action_space.n  # Number of possible actions

    # Initialize Q table and returns using default factories
    Q = defaultdict(default_q_value)
    returns_sum = defaultdict(default_return_sum)
    returns_count = defaultdict(default_return_count)
    rewards_per_episode = []    # Store total rewards for each episode
    epsilon = epsilon_start     # Initialize epsilon

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state_discrete = discretize_state(state, bins)  # Discretize the state
        episode_memory = []     # Memory for storing (state, action, reward)
        done = False
        total_reward = 0        # Initialize total reward for the episode

        # Run the episode until it's done
        while not done:
            action = choose_action(state_discrete, Q, epsilon, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)  # Step in the environment
            total_reward += reward

            # Check if the episode should terminate
            if total_reward >= max_total_reward:
                terminated = True
                truncated = False

            done = terminated or truncated
            next_state_discrete = discretize_state(next_state, bins)    # Discretize the next state

            episode_memory.append((state_discrete, action, reward))     # Store the transition
            state_discrete = next_state_discrete

        rewards_per_episode.append(total_reward)  # Record the total reward for this episode

        # First visit Monte Carlo update for Q-values
        visited = set()  # Track visited state-action pairs
        G = 0  # Initialize return
        for state, action, reward in reversed(episode_memory):
            G = gamma * G + reward  # Calculate the return
            if (state, action) not in visited:
                returns_sum[state][action] += G     # Update returns sum
                returns_count[state][action] += 1   # Increment returns count
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]  # Update Q-value
                visited.add((state, action))        # Mark state-action as visited

        # Print progress every 200 episodes
        if episode % 200 == 0:
            avg_reward = np.mean(rewards_per_episode[-200:])  # Calculate average reward for the last 200 episodes
            print(f'Run {run_id} - Episode: {episode}/{num_episodes}, Epsilon: {epsilon:.4f}, Average Reward: {avg_reward:.2f}')

        # Decay epsilon
        epsilon -= epsilon_decay_rate           # Reduce epsilon
        epsilon = max(epsilon, min_epsilon)     # Ensure epsilon does not go below min_epsilon

    env.close()  # Close the environment

    # Calculate moving average of rewards over 200 episodes
    if len(rewards_per_episode) >= 200:
        average_rewards = np.convolve(rewards_per_episode, np.ones(200)/200, mode='valid')
    else:
        average_rewards = np.array(rewards_per_episode)

    print(f'Run {run_id} completed.')   # Indicate completion of the run
    return average_rewards, Q           # Return average rewards and Q-table

# Wrapper function for running Monte Carlo in a separate process
def run_monte_carlo_wrapper(run_id, num_episodes):
    """Wrapper to call the Monte Carlo function."""
    return monte_carlo(run_id, num_episodes)

# Safely loads a pickle file and handles errors
def load_pickle_file(filename):
    """Loads data from a pickle file, handling potential loading errors."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
        print(f'Error loading pickle file {filename}: {e}')
        return None

# Main execution function to handle training and loading
def main():
    """Main function for executing the Monte Carlo simulation."""
    parser = argparse.ArgumentParser(description='Monte Carlo Control for CartPole with Pickle Saving/Loading')
    parser.add_argument('--mode', choices=['train', 'load'], default='train',
                        help='Mode to run the script: "train" to train the model, "load" to load a pretrained model')
    args = parser.parse_args()

    # Directory to save individual pickle files
    pickle_dir = 'monte_carlo_discrete_models'
    os.makedirs(pickle_dir, exist_ok=True)  # Create directory if it does not exist

    image_filename = 'average_rewards_Monte_Carlo_Discrete.png'
    
    if args.mode == 'train':
        num_episodes = 52000        # Total episodes for training
        num_runs = 7                # Number of independent runs
        bins = (15, 15, 15, 15)     # Number of bins for each state variable

        # Create a pool of worker processes to run simulations in parallel
        with mp.Pool(processes=min(num_runs, mp.cpu_count())) as pool:
            args_list = [(run, num_episodes) for run in range(1, num_runs + 1)]     # Prepare arguments for each run
            results = pool.starmap(run_monte_carlo_wrapper, args_list)              # Run simulations

        # Separate average_rewards and Q_tables from results
        all_average_rewards = [result[0] for result in results]
        all_Q_tables = [result[1] for result in results]

        # Save each run's data in a separate pickle file
        for run_id, (avg_rewards, Q_table) in enumerate(zip(all_average_rewards, all_Q_tables), start=1):
            pickle_filename = os.path.join(pickle_dir, f'Monte_Carlo_Discrete_Run{run_id}.pkl')
            with open(pickle_filename, 'wb') as f:
                pickle.dump({
                    'average_rewards': avg_rewards,
                    'Q_table': Q_table,
                    'bins': bins
                }, f, protocol=pickle.HIGHEST_PROTOCOL)  # Save data with highest protocol
            print(f'Run {run_id} data saved to {pickle_filename}.')

        print(f'All runs completed and saved in directory "{pickle_dir}".')

        # Plotting all average rewards on the same plot
        palette = sns.color_palette("flare", 8)     # Set color palette for the plot
        plt.style.use("cyberpunk")                  # Use cyberpunk style for the plot
        plt.figure(figsize=(12, 8))                 # Set figure size
        for i, avg_rewards in enumerate(all_average_rewards):
            plt.plot(avg_rewards, color=palette[i % len(palette)], label=f'Run {i+1}')  # Plot each run's average rewards
        plt.xlabel('Episodes')                              # Label x-axis
        plt.ylabel('Average Reward (over 200 episodes)')    # Label y-axis
        plt.title('Average Reward Tracking in Monte Carlo Discrete Over Multiple Runs')  # Set plot title
        plt.legend()                    # Show legend
        plt.grid(True)                  # Enable grid
        plt.tight_layout()              # Adjust layout
        plt.savefig(image_filename)     # Save the plot as an image
        plt.show()                      # Display the plot

    elif args.mode == 'load':
        # Check if the pickle directory exists
        if not os.path.exists(pickle_dir):
            print(f'Pickle directory "{pickle_dir}" not found. Please train the model first using --mode train.')
            return

        # List all pickle files in the directory
        pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pkl')]
        if not pickle_files:
            print(f'No pickle files found in directory "{pickle_dir}". Please train the model first using --mode train.')
            return

        all_average_rewards = []
        all_Q_tables = []
        bins = None

        # Load each pickle file and gather data
        for pickle_file in pickle_files:
            filepath = os.path.join(pickle_dir, pickle_file)
            data = load_pickle_file(filepath)  # Load data from pickle file
            if data is None:
                print(f'Skipping file {filepath} due to loading error.')
                continue
            all_average_rewards.append(data.get('average_rewards', []))     # Store average rewards
            all_Q_tables.append(data.get('Q_table', {}))                    # Store Q-table
            if bins is None:
                bins = data.get('bins', (15, 15, 15, 15))  # Store bin configuration

        if not all_average_rewards:
            print('No valid average rewards found in the pickle files.')
            return

        # Plotting the loaded average rewards
        palette = sns.color_palette("flare", 8)
        plt.style.use("cyberpunk")              # Use cyberpunk style for the plot
        plt.figure(figsize=(12, 8))             # Set figure size
        for i, avg_rewards in enumerate(all_average_rewards):
            plt.plot(avg_rewards, color=palette[i % len(palette)], label=f'Run {i+1}')  # Plot each run's average rewards
        plt.xlabel('Episodes')                  # Label x-axis
        plt.ylabel('Average Reward (over 200 episodes)')  # Label y-axis
        plt.title('Loaded Average Reward Tracking in Monte Carlo Control Over Multiple Runs')  # Set plot title
        plt.legend()                            # Show legend
        plt.grid(True)                          # Enable grid
        plt.tight_layout()                      # Adjust layout
        plt.savefig(image_filename)             # Save the plot as an image
        plt.show()                              # Display the plot

        print(f'All models loaded from directory "{pickle_dir}" and average rewards plotted as "{image_filename}".')

if __name__ == '__main__':
    main()  # Execute the main function
