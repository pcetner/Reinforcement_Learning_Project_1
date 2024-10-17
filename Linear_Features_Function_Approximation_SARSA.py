import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
import mplcyberpunk
import pickle
import os

# Function to extract and normalize user-defined features from the state
def extract_features(state):
    x, x_dot, theta, theta_dot = state
    # Define typical maximum values for normalization
    xm = 2.4                # Maximum x position
    xdm = 4                 # Maximum x velocity
    tm = np.radians(12)     # Maximum theta angle (in radians)
    tdm = 5                 # Maximum theta velocity
    # Return normalized features as a numpy array
    return np.array([
        x / xm,
        x_dot / xdm,
        theta / tm,
        theta_dot / tdm,
        (x ** 2) / (xm ** 2),
        (x_dot ** 2) / (xdm ** 2),
        (theta ** 2) / (tm ** 2),
        (theta_dot ** 2) / (tdm ** 2),
        (x * theta) / (xm * tm),
        (x_dot * theta_dot) / (xdm * tdm)
    ])

# Class for a linear Q function approximator
class LinearQFunction:
    def __init__(self, num_features, num_actions):
        # Initialize weights with small random values
        self.weights = np.random.uniform(low=-0.01, high=0.01, size=(num_actions, num_features))

    # Compute the Q-value for a given state-action pair
    def q_value(self, state, action):
        features = extract_features(state)                  # Extract features from the state
        return np.dot(self.weights[action], features)       # Calculate Q-value using weights

    # Update the weights based on the TD error
    def update(self, state, action, target, alpha):
        features = extract_features(state)                  # Extract features from the state
        td_error = target - self.q_value(state, action)     # Calculate temporal-difference error
        # Update weights using TD error and learning rate
        self.weights[action] += alpha * td_error * features

# Function for epsilon-greedy action selection
def choose_action(state, q_function, epsilon, action_space):
    # Select a random action with a probability of epsilon, else exploit
    if np.random.rand() < epsilon:
        return action_space.sample()  # Explore: random action
    else:
        # Exploit: choose the action with the highest Q-value
        return np.argmax([q_function.q_value(state, a) for a in range(action_space.n)])

# Implementation of the SARSA reinforcement learning algorithm
def sarsa(run_id, num_episodes, models_dir, alpha=0.001, gamma=0.99, epsilon=1, epsilon_decay_rate=0.0001, max_total_reward=500, train=True):
    env = gym.make('CartPole-v1')       # Initialize the CartPole environment
    num_actions = env.action_space.n    # Number of possible actions
    q_function = LinearQFunction(num_features=10, num_actions=num_actions)  # Initialize Q function

    # Create a unique model filename for saving/loading
    model_file = os.path.join(models_dir, f'Linear_Features_Function_Approximation_SARSA_run{run_id}.pkl')

    # Load the model if not training and it exists
    if not train and os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            q_function = pickle.load(f)  # Load previously saved model
        print(f'Run {run_id}: Model loaded from {model_file}')

    rewards_per_episode = []  # List to store total rewards for each episode

    # Main loop over episodes
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()      # Reset the environment for a new episode
        action = choose_action(state, q_function, epsilon, env.action_space)  # Select initial action
        total_reward = 0            # Initialize the total reward for this episode
        done = False                # Flag to indicate episode termination

        # Loop until the episode is done
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)  # Take action in the environment
            total_reward += reward  # Accumulate the reward

            # Check if the cumulative reward exceeds the maximum threshold
            if total_reward >= max_total_reward:
                terminated = True  # Mark the episode as terminated
                truncated = False  # Reset truncated flag

            done = terminated or truncated  # Check if the episode is done

            # Select the next action based on the updated state
            next_action = choose_action(next_state, q_function, epsilon, env.action_space)

            # Compute target for Q-value update
            if not done:
                target = reward + gamma * q_function.q_value(next_state, next_action)  # Future reward
            else:
                target = reward  # No future rewards if the episode is done

            # Update the Q function if training is enabled
            if train:
                q_function.update(state, action, target, alpha)

            # Transition to the next state and action
            state, action = next_state, next_action

        rewards_per_episode.append(total_reward)  # Store the total reward for this episode

        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f'Run {run_id} - Episode: {episode}/{num_episodes}, Current Epsilon: {epsilon:.4f}, Latest Reward: {total_reward}')

        # Decay epsilon after each episode
        epsilon -= epsilon_decay_rate
        epsilon = max(epsilon, 0.02)  # Ensure epsilon remains above 0.02

    env.close()  # Close the environment

    # Save the model after training if applicable
    if train:
        with open(model_file, 'wb') as f:
            pickle.dump(q_function, f)  # Save the Q function to a file
        print(f'Run {run_id}: Model saved to {model_file}')

    # Calculate the average reward per episode using a moving average
    if len(rewards_per_episode) >= 100:
        average_rewards = np.convolve(rewards_per_episode, np.ones(100)/100, mode='valid')  # Moving average
    else:
        average_rewards = np.array(rewards_per_episode)  # Use raw rewards if fewer than 100 episodes

    print(f'Run {run_id} completed.')  # Indicate completion
    return average_rewards  # Return the computed average rewards

# Wrapper function for multiprocessing to handle arguments
def run_sarsa_wrapper(args):
    run_id, num_episodes, models_dir, train = args                  # Unpack arguments
    return sarsa(run_id, num_episodes, models_dir, train=train)     # Call the sarsa function

# Main execution block with multiprocessing setup
if __name__ == '__main__':
    num_episodes = 12000    # Total episodes for training
    num_runs = 7            # Number of independent runs
    train = True            # Flag to determine if we are training or loading a model

    # Create a directory for saving models if it doesn't exist
    models_dir = 'linear_features_function_approximation_SARSA_models'
    os.makedirs(models_dir, exist_ok=True)

    # Prepare arguments for each run as tuples
    args = [(run, num_episodes, models_dir, train) for run in range(1, num_runs + 1)]

    # Create a pool of worker processes for parallel execution
    with mp.Pool(processes=num_runs) as pool:
        all_average_rewards = pool.map(run_sarsa_wrapper, args)  # Execute runs in parallel

    # Plotting the average rewards from all runs
    palette = sns.color_palette("flare", num_runs + 1)  # Generate a color palette for plots
    plt.style.use("cyberpunk")      # Set plot style
    plt.figure(figsize=(12, 8))     # Initialize plot size
    for i, avg_rewards in enumerate(all_average_rewards, 1):
        plt.plot(avg_rewards, color=palette[i], label=f'Run {i}')  # Plot average rewards

    plt.xlabel('Episodes')                                              # X-axis label
    plt.ylabel('Average Reward (over 100 episodes)')                    # Y-axis label
    plt.title('Average Reward Tracking in SARSA Over Multiple Runs')    # Plot title
    plt.legend()                                                        # Display legend
    plt.grid(True)                                                      # Show grid
    plt.tight_layout()                                                  # Adjust layout
    plt.savefig('Linear_Features_Function_Approximation_SARSA_Average_Rewards.png')     # Save plot as an image
    plt.show()                                                          # Display the plot
