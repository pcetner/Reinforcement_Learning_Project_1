import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
import pickle
import seaborn as sns
from multiprocessing import Pool
import argparse
import os

# Function to plot average rewards for all runs at once
def plot_all_average_rewards(all_rewards, num_runs, window=10000):
    """
    Plots the moving average rewards over multiple runs.

    Args:
        all_rewards (list of lists): Rewards from all runs.
        num_runs (int): Number of parallel runs.
        window (int, optional): Window size for moving average. Defaults to 1000.
    """
    # Calculate moving averages for each run
    all_average_rewards = []
    for rewards in all_rewards:
        # Calculate the moving average using a simple convolution
        moving_average = np.convolve(rewards, np.ones(window)/window, mode='valid')
        all_average_rewards.append(moving_average)

    # Convert to a numpy array for easier handling
    all_average_rewards = np.array(all_average_rewards)

    palette = sns.color_palette("flare", num_runs)
    plt.style.use("cyberpunk")  
    plt.figure(figsize=(12, 8))

    # Plot individual runs' average rewards
    for i in range(num_runs):
        plt.plot(all_average_rewards[i], color=palette[i], label=f'Run {i + 1}')

    plt.xlabel('Episodes')
    plt.ylabel(f'Moving Average Reward (over {window} episodes)')
    plt.title('Q Learning Discrete - Moving Average Reward Tracking Over Multiple Runs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Q_Learning_Discrete_Average_Rewards_Multiple_Runs.png')  # Save the combined plot as an image file
    plt.show()  # Display the combined plot


# Function to compute nonlinear bin edges
def compute_nonlinear_bins(bound, density_strength, num_bins):
    """
    Compute nonlinear bin edges using a custom formula.

    Args:
        bound (float): The maximum absolute value for the variable.
        density_strength (float): Controls the degree of nonlinearity. 0 is linear, 1 is very centered around the center. (0 <= density_strength <= 1).
        num_bins (int): Number of bins.

    Returns:
        np.ndarray: Array of bin edges.
    """
    # Generate raw bins from -1 to 1
    raw_bins = np.linspace(-1, 1, num_bins)

    # Apply the nonlinear transformation
    bins = bound * (density_strength * (raw_bins) ** 5 + (1 - density_strength) * raw_bins)

    return bins

def train_q_learning(run_num, episodes=40000, is_training=True, render=False, density_strength=0.2):
    """
    Runs the Q-learning algorithm on the CartPole environment with nonlinear binning.

    Args:
        run_num (int): The run number (for saving unique Q-tables and results).
        episodes (int, optional): Number of episodes to run. Defaults to 40000.
        is_training (bool, optional): Whether to train the agent or load a pre-trained Q-table. Defaults to True.
        render (bool, optional): Whether to render the environment. Defaults to False.
        density_strength (float, optional): Controls the degree of nonlinearity for binning. Defaults to 0.2.

    Returns:
        list: Rewards per episode.
    """
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Define the bounds for each state variable
    pos_min, pos_max = -2.4, 2.4
    vel_min, vel_max = -4.0, 4.0
    ang_min, ang_max = -0.2095, 0.2095  # radians
    ang_vel_min, ang_vel_max = -4.0, 4.0

    num_bins = 10  # Number of bins for each state variable

    # Compute nonlinear bins for each state variable
    xpos_bins = compute_nonlinear_bins(pos_max, density_strength, num_bins)
    xpos_bins[0] = pos_min
    xpos_bins[-1] = pos_max

    xvel_bins = compute_nonlinear_bins(vel_max, density_strength, num_bins)
    xvel_bins[0] = vel_min
    xvel_bins[-1] = vel_max

    theta_bins = compute_nonlinear_bins(ang_max, density_strength, num_bins)
    theta_bins[0] = ang_min
    theta_bins[-1] = ang_max

    ang_vel_bins = compute_nonlinear_bins(ang_vel_max, density_strength, num_bins)
    ang_vel_bins[0] = ang_vel_min
    ang_vel_bins[-1] = ang_vel_max

    # Initialize or load the Q-table
    if is_training:
        # Q-table dimensions: position x velocity x angle x angular velocity x actions
        q = np.zeros((len(xpos_bins) + 1, len(xvel_bins) + 1, len(theta_bins) + 1, len(ang_vel_bins) + 1, env.action_space.n))
    else:
        q_filename = f'cartpole_run{run_num}.pkl'
        if not os.path.exists(q_filename):
            print(f'Q-table file {q_filename} does not exist. Cannot load.')
            return []
        with open(q_filename, 'rb') as f:
            q = pickle.load(f)

    # Hyperparameters
    learning_rate_a = 0.1           # Alpha: learning rate
    discount_factor_g = 0.99        # Gamma: discount factor
    epsilon = 1.0                   # Exploration rate (1 = 100% random actions)
    epsilon_decay_rate = 0.00001    # Decay rate for epsilon
    rng = np.random.default_rng()   # Random number generator

    rewards_per_episode = []

    for i in range(episodes):
        state = env.reset()[0]  # Reset the environment and get the initial state

        # Discretize the state variables using the defined bins
        state_p = np.digitize(state[0], xpos_bins)
        state_v = np.digitize(state[1], xvel_bins)
        state_a = np.digitize(state[2], theta_bins)
        state_av = np.digitize(state[3], ang_vel_bins)

        terminated = False  # Flag to check if the episode has ended
        rewards = 0         # Initialize reward counter

        while not terminated and rewards < 10000:
            # Epsilon-greedy action selection
            if is_training and rng.random() < epsilon:
                # Choose a random action (exploration)
                action = env.action_space.sample()
            else:
                # Choose the best action based on the current Q-table (exploitation)
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            # Execute the chosen action
            new_state, reward, terminated, _, _ = env.step(action)

            # Cap the reward to ensure it never exceeds 500
            if rewards + reward > 500:
                reward = 500 - rewards

            # Discretize the new state
            new_state_p = np.digitize(new_state[0], xpos_bins)
            new_state_v = np.digitize(new_state[1], xvel_bins)
            new_state_a = np.digitize(new_state[2], theta_bins)
            new_state_av = np.digitize(new_state[3], ang_vel_bins)

            if is_training:
                # Q-learning update rule
                current_q = q[state_p, state_v, state_a, state_av, action]
                max_future_q = np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :])
                new_q = current_q + learning_rate_a * (reward + discount_factor_g * max_future_q - current_q)
                q[state_p, state_v, state_a, state_av, action] = new_q

            # Transition to the new state
            state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av
            rewards += reward

        rewards_per_episode.append(rewards)
        
        # Calculate the mean reward over the last 100 episodes
        if len(rewards_per_episode) >= 100:
            mean_rewards = np.mean(rewards_per_episode[-100:])
        else:
            mean_rewards = np.mean(rewards_per_episode)

        # Print progress every 100 episodes
        if is_training and i % 100 == 0:
            print(f'Run {run_num} | Episode: {i} | Rewards: {rewards} | Epsilon: {epsilon:.4f} | Mean Rewards (Last 100): {mean_rewards:.1f}')

        # Terminate training if mean rewards are sufficiently high
        if is_training and mean_rewards > 1000:
            print(f'Run {run_num} | Terminating early at episode {i} with mean rewards {mean_rewards:.1f}')
            break

        # Decay epsilon to reduce exploration over time
        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, 0)

    env.close()

    # Save the Q-table if training
    if is_training:
        q_filename = f'cartpole_run{run_num}.pkl'
        with open(q_filename, 'wb') as f:
            pickle.dump(q, f)

    return rewards_per_episode

# Function to run multiple training instances in parallel
def run_parallel_training(num_runs, episodes, is_training, density_strength):
    """
    Executes multiple training runs in parallel.

    Args:
        num_runs (int): Number of parallel runs.
        episodes (int): Number of episodes per run.
        is_training (bool): Whether to train or load models.
        density_strength (float): Controls the degree of nonlinearity for binning.

    Returns:
        list: Rewards per episode for each run.
    """
    with Pool(num_runs) as pool:
        rewards_per_run = pool.starmap(train_q_learning, [(run_num + 1, episodes, is_training, False, density_strength) for run_num in range(num_runs)])  # Start run_num from 1
    return rewards_per_run

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate Q-learning on CartPole environment.')
    parser.add_argument('--runs', type=int, default=1, help='Number of parallel training runs.')
    parser.add_argument('--episodes', type=int, default=40000, help='Number of episodes to train for each run.')
    parser.add_argument('--training', action='store_true', help='Set this flag to train a new model.')
    parser.add_argument('--density_strength', type=float, default=0.2, help='Density strength for binning.')
    parser.add_argument('--plot', action='store_true', help='Plot the average rewards of all runs.')

    args = parser.parse_args()

    # Run training or evaluation
    all_rewards = run_parallel_training(args.runs, args.episodes, args.training, args.density_strength)

    # If plotting is enabled, create a plot
    if args.plot:
        plot_all_average_rewards(all_rewards, args.runs)

if __name__ == '__main__':
    main()

