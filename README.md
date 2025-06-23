
markdown
Copy
Edit
# Deep Convolutional Q-Learning for Pac-Man

This project implements a Deep Q-Learning algorithm using Convolutional Neural Networks (CNNs) to train an AI agent to play Pac-Man. The environment is based on OpenAI's `gymnasium` framework, and the neural network is built using PyTorch.

## 📌 Features

- Uses Deep Q-Learning with experience replay
- Convolutional Neural Network (CNN) for processing visual input
- Custom reward shaping for better learning
- Target network updates for stabilized training
- Based on Gymnasium's `ALE/Pacman-v5` environment

## 🧠 Algorithms Used

- **Deep Q-Network (DQN)**: An extension of Q-learning using deep neural networks to approximate Q-values.
- **Experience Replay**: A replay buffer stores past experiences to break correlation and improve learning stability.
- **Target Network**: A copy of the main network that is updated at fixed intervals to stabilize training.

## 📦 Requirements

Install the required packages using:

```bash
pip install gymnasium[atari] gymnasium[accept-rom-license] ale-py torch torchvision numpy matplotlib
⚠️ Make sure you have ROM licenses accepted to use Atari environments.

🛠 Project Structure
Part 0: Install dependencies and import necessary libraries.

Part 1: Build the AI agent with CNN-based architecture and DQN logic.

Part 2: Train the agent in the Pac-Man environment.

Part 3 (Optional): Visualize performance, plot rewards, or save/load models.

🕹 Environment
This project uses:

ALE/Pacman-v5 (from Gymnasium)

Python 3.x

PyTorch

OpenAI Gymnasium for the game environment

📊 Results
The agent is expected to learn to play Pac-Man through trial and error, optimizing its behavior using rewards. Training time and performance will depend on system resources and the number of episodes.

🚀 Running the Project
Clone the repository or download the notebook.

Install dependencies.

Run the notebook in Jupyter or any Python IDE supporting notebooks.

Adjust hyperparameters (episodes, learning rate, etc.) as needed.

📁 Notes
This is a partial implementation and may require further development for:

Logging performance metrics

Saving/loading trained models

Playing a demo of the trained agent

# deep_convolutional_q-learning_for_pacman
