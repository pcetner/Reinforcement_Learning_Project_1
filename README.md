# Reinforcement Learning Project 1

This repository contains implementations of various reinforcement learning algorithms applied to the gymnasium cartpole-v1 environment. The project explores different approaches to reinforcement learning, including discrete methods, function approximation, and deep learning techniques.

## Project Structure

The project consists of the following main components:

### Discrete Methods

- `Monte_Carlo_Discrete.py`: Implementation of the Monte Carlo method for discrete state spaces
- `Q_Learning_Discrete.py`: Implementation of Q-Learning for discrete state spaces
- `SARSA_Discrete.py`: Implementation of SARSA (State-Action-Reward-State-Action) for discrete state spaces

### Function Approximation Methods

- `Linear_Features_Function_Approximation_SARSA.py`: SARSA with linear function approximation
- `Monte_Carlo_Function_Approximation.py`: Monte Carlo method with function approximation

### Deep Reinforcement Learning

- `Non_Linear_Function_Approximation_DQN.py`: Deep Q-Network (DQN) implementation

### Visualization and Analysis

- `VisualizeBinning.py`: Script for visualizing the binning process (likely for state space discretization)

## Results

The project includes several visualizations of the results. These images show the performance of the different algorithms, plotting a moving average for seven runs trained in parallel.

<div align="center">
  <strong>Monte_Carlo_Discrete_Average_Rewards.png</strong>
  <br/>
  <img src="Monte_Carlo_Discrete_Average_Rewards.png" alt="Monte Carlo Discrete Average Rewards" width="1200" />
</div>

<br/>
<br/>

<div align="center">
  <strong>Monte_Carlo_with_Function_Approximation_Average_Rewards_.png</strong>
  <br/>
  <img src="Monte_Carlo_with_Function_Approximation_Average_Rewards_.png" alt="Monte Carlo with Function Approximation Average Rewards" width="1200" />
</div>

<br/>
<br/>

<div align="center">
  <strong>Non_Linear_Function_Approximation_DQN_Average_Rewards.png</strong>
  <br/>
  <img src="Non_Linear_Function_Approximation_DQN_Average_Rewards.png" alt="Non Linear Function Approximation DQN Average Rewards" width="1200" />
</div>

<br/>
<br/>

<div align="center">
  <strong>Q_Learning_Discrete_Average_Rewards_Multiple_Runs.png</strong>
  <br/>
  <img src="Q_Learning_Discrete_Average_Rewards_Multiple_Runs.png" alt="Q-Learning Discrete Average Rewards (Multiple Runs)" width="1200" />
</div>

<br/>
<br/>

<div align="center">
  <strong>SARSA_Discrete_Average_Rewards.png</strong>
  <br/>
  <img src="SARSA_Discrete_Average_Rewards.png" alt="SARSA Discrete Average Rewards" width="1200" />
</div>

<br/>
<br/>

<div align="center">
  <strong>Linear_Features_Function_Approximation_SARSA_Average_Rewards.png</strong>
  <br/>
  <img src="Linear_Features_Function_Approximation_SARSA_Average_Rewards.png" alt="Linear Features Function Approximation SARSA Average Rewards" width="1200" />
</div>

<br/>
<br/>


In addition, the discrete methods can use nonlinear binning in the discretization processes, visualized in`Visualize_Binning.png`.

### Getting Started

To run these scripts, you'll need Python installed on your system, along with several libraries commonly used in reinforcement learning projects. Below is a list of the key libraries required:

- **NumPy**: For numerical operations and array manipulations.
- **Gymnasium**: A toolkit for developing and comparing reinforcement learning algorithms.
- **Matplotlib**: For visualizing results through plots and graphs.
- **PyTorch**: A popular deep learning framework used for building neural networks.
- **Seaborn**: An extension of Matplotlib for more aesthetically pleasing statistical visualizations.
- **MPLCyberpunk**: A style sheet for adding a cyberpunk aesthetic to Matplotlib plots.
- **Pickle**: For saving and loading serialized Python objects.

## Usage

To run any of the algorithms, use Python to execute the corresponding script. For example:

`python Monte_Carlo_Discrete.py`

Additionally, the `Q_Learning_Discrete.py`file has flags built in, and needs to be run as follows:
`python /<PATH_TO_FILE>/Q_Learning_Discrete.py --training --episodes 100000 --density_strength .4 --plot --runs 7`

For additional information, refer to the comments within each file.

Make sure you have the necessary dependencies installed and the correct environment set up before running the scripts.
