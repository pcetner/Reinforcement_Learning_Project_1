import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
import pickle
import multiprocessing as mp
import seaborn as sns

def compute_nonlinear_bins(bound, density_strength, num_bins):
    """
    Compute nonlinear bin edges using a custom formula.

    Args:
        bound (float): The maximum absolute value for the variable (theta, angular velocity, xpos, or xvel).
        density_strength (float): Controls the degree of nonlinearity. 0 is linear, 1 is very centered around the center. (0 <= density_strength <= 1).
        num_bins (int): Number of bins.

    Returns:
        np.ndarray: Array of bin edges.
    """
    # Generate evenly spaced raw bins from -1 to 1
    raw_bins = np.linspace(-1, 1, num_bins)

    # Apply the nonlinear transformation based on density strength
    bins = bound * (density_strength * (raw_bins) ** 5 + (1 - density_strength) * raw_bins)

    return bins

def run_sarsa(
    run_number,           # Track the current run number
    episodes=52000,       # Default number of episodes
    is_training=True,     # Flag to indicate training or evaluation mode
    render=False,         # Render the environment (visualization)
    density_strength=0.2, # Controls nonlinearity for binning
    save_filename=None,   # Optional: file to save the rewards plot
    q_table_filename='sarsa_cartpole.pkl'  # File to save/load the Q-table
):
    """
    Run the SARSA algorithm on the CartPole environment with nonlinear binning.

    Args:
        run_number (int): The run number for tracking.
        episodes (int): Number of episodes to run.
        is_training (bool): Whether to train the agent or load a pre-trained Q-table.
        render (bool): Whether to render the environment.
        density_strength (float): Controls the degree of nonlinearity for binning.
        save_filename (str): If provided, save the rewards plot to this file.
        q_table_filename (str): Filename to save/load the Q-table.
    """
    # Create the CartPole environment (render only if specified)
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Define the bounds for each state variable
    pos_min, pos_max = -2.4, 2.4          # Cart position bounds
    vel_min, vel_max = -4.0, 4.0          # Cart velocity bounds
    ang_min, ang_max = -0.2095, 0.2095    # Pole angle bounds (in radians)
    ang_vel_min, ang_vel_max = -4.0, 4.0  # Angular velocity bounds

    num_bins = 10  # Number of bins for discretizing state variables

    # Compute nonlinear bins for each state variable
    xpos_bins = compute_nonlinear_bins(pos_max, density_strength, num_bins)
    xpos_bins[0], xpos_bins[-1] = pos_min, pos_max  # Ensure bin edges include min/max

    xvel_bins = compute_nonlinear_bins(vel_max, density_strength, num_bins)
    xvel_bins[0], xvel_bins[-1] = vel_min, vel_max

    theta_bins = compute_nonlinear_bins(ang_max, density_strength, num_bins)
    theta_bins[0], theta_bins[-1] = ang_min, ang_max

    ang_vel_bins = compute_nonlinear_bins(ang_vel_max, density_strength, num_bins)
    ang_vel_bins[0], ang_vel_bins[-1] = ang_vel_min, ang_vel_max

    # Initialize Q-table or load from file if not training
    if is_training:
        q = np.zeros((
            len(xpos_bins) + 1,
            len(xvel_bins) + 1,
            len(theta_bins) + 1,
            len(ang_vel_bins) + 1,
            env.action_space.n
        ))
    else:
        try:
            with open(q_table_filename, 'rb') as f:
                q = pickle.load(f)
            print(f"Loaded Q-table from {q_table_filename}")
        except FileNotFoundError:
            print(f"No Q-table found at {q_table_filename}. Exiting.")
            return

    # SARSA algorithm hyperparameters
    learning_rate = 0.04          # Alpha: learning rate
    discount_factor = 0.99        # Gamma: discount factor
    epsilon = 1.0                 # Exploration rate
    epsilon_decay_rate = 0.00002  # Decay rate for epsilon
    min_epsilon = 0               # Minimum exploration rate
    rng = np.random.default_rng() # Random number generator

    rewards_per_episode = []  # Track rewards for each episode

    for episode in range(1, episodes + 1):
        state = env.reset()[0]  # Reset the environment to get the initial state

        # Discretize the state variables using the predefined bins
        state_p = np.digitize(state[0], xpos_bins)
        state_v = np.digitize(state[1], xvel_bins)
        state_a = np.digitize(state[2], theta_bins)
        state_av = np.digitize(state[3], ang_vel_bins)

        terminated, truncated = False, False  # Flags to handle episode end
        total_rewards = 0                     # Track cumulative rewards

        # Epsilon-greedy policy for choosing the initial action
        if rng.random() < epsilon:
            action = env.action_space.sample()  # Random action (exploration)
        else:
            action = np.argmax(q[state_p, state_v, state_a, state_av, :])  # Best action (exploitation)

        while not (terminated or truncated):
            # Perform the action and observe the next state and reward
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Discretize the new state
            new_state_p = np.digitize(new_state[0], xpos_bins)
            new_state_v = np.digitize(new_state[1], xvel_bins)
            new_state_a = np.digitize(new_state[2], theta_bins)
            new_state_av = np.digitize(new_state[3], ang_vel_bins)

            if is_training:
                # Epsilon-greedy policy for the next action
                if rng.random() < epsilon:
                    next_action = env.action_space.sample()  # Random action
                else:
                    next_action = np.argmax(q[new_state_p, new_state_v, new_state_a, new_state_av, :])  # Best action

                # Apply SARSA update rule
                current_q = q[state_p, state_v, state_a, state_av, action]
                next_q = q[new_state_p, new_state_v, new_state_a, new_state_av, next_action]
                td_target = reward + discount_factor * next_q
                td_error = td_target - current_q
                q[state_p, state_v, state_a, state_av, action] = current_q + learning_rate * td_error

                # Update the current action
                action = next_action

            # Transition to the new state and accumulate rewards
            state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av
            total_rewards += reward

        rewards_per_episode.append(total_rewards)  # Store total rewards for this episode

        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)  # Decay epsilon after each episode

        # Calculate mean reward for the last 100 episodes
        mean_rewards = np.mean(rewards_per_episode[-100:]) if episode >= 100 else np.mean(rewards_per_episode)

        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f'Run {run_number} | Episode: {episode} | Total Rewards: {total_rewards} | Epsilon: {epsilon:.4f} | Mean Rewards (Last 100): {mean_rewards:.1f}')

    env.close()

    # Save the Q-table if training is complete
    if is_training:
        with open(q_table_filename, 'wb') as f:
            pickle.dump(q, f)
        print(f"Run {run_number} | Saved Q-table to {q_table_filename}")

    return rewards_per_episode

def plot_all_average_rewards(all_rewards, num_runs, window=100, maximum_reward=495):
    """
    Plot the average rewards for all runs. If a run reaches the maximum reward,
    continue plotting the maximum reward for the remaining episodes.

    Args:
        all_rewards (list of lists): Rewards per episode for each run.
        num_runs (int): Number of runs.
        window (int): Window size for moving average.
        maximum_reward (float): The maximum reward to continue plotting after it's reached.
    """
    # Calculate moving average of rewards for each run
    all_average_rewards = [np.convolve(rewards, np.ones(window) / window, mode='valid') for rewards in all_rewards]

    # Plot the average rewards with a cyberpunk theme
    plt.figure(figsize=(12, 8), facecolor="#282c44")
    plt.style.use("cyberpunk")

    # Use a color palette for the multiple runs
    palette = sns.color_palette("flare", num_runs)

    for i, avg_rewards in enumerate(all_average_rewards):
        plt.plot(avg_rewards, color=palette[i], label=f'Run {i + 1}')

    plt.xlabel('Episodes')
    plt.ylabel(f'Average Reward (over {window} episodes)')
    plt.title('Average Reward Tracking in SARSA Over Multiple Runs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('SARSA_Discrete_Average_Rewards.png')  # Save plot as PNG file
    plt.show()

def run_sarsa_wrapper(args):
    """
    Wrapper function for running SARSA in parallel using multiprocessing.
    """
    return run_sarsa(*args)

if __name__ == '__main__':
    # Define density values for multiple runs (identical values for this case)
    density_values = [0.2] * 7              # 7 runs with density 0.2
    num_processes = len(density_values)     # Run in parallel using multiple processes

    # Prepare arguments for each run
    args = [(i + 1, 52000, True, False, density) for i, density in enumerate(density_values)]

    # Execute runs in parallel using multiprocessing
    with mp.Pool(num_processes) as pool:
        results = pool.map(run_sarsa_wrapper, args)

    # Plot the average rewards across all runs
    plot_all_average_rewards(results, num_processes)
