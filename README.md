# DQN (Deep Q-Network) Reinforcement Learning

This repository contains a Deep Q-Network (DQN) implementation using PyTorch. The DQN model is trained using a Gym environment, and it aims to learn how to solve reinforcement learning tasks.

## Requirements

- Python 3.10 or higher
- `gym`
- `numpy`
- `torch`
- `torchvision`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/dqn-reinforcement-learning.git
   cd dqn-reinforcement-learning


Create a virtual environment:
python3.10 -m venv new_rl_env
source new_rl_env/bin/activate


Install the required packages:
pip install -r requirements.txt

Alternatively, you can install each package manually:
pip install gym
pip install numpy==1.24.3
pip install torch torchvision

Running the Script
Once the environment is set up, you can run the DQN script with the following command:
python dqn.py
This will start the training of the DQN model on a Gym environment.

Model Training
The DQN model will be trained over multiple episodes, where it will learn to take actions based on the environment's state to maximize cumulative rewards.

Example output
The script will display progress and the current rewards achieved by the model as it trains.

Contributing
Feel free to fork this repository and make changes as needed. If you would like to contribute, please follow the steps below:

Fork the repository.
Create a new branch (git checkout -b feature-name).
Make changes and commit them (git commit -am 'Add feature').
Push the branch (git push origin feature-name).
Create a pull request.


