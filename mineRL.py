"""

!!! Modified MineRL Library do not change or update the MineRL Library  !!!


"""

# Imports
import gym
import minerl
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict 

# Silicon Chip GPU Acceleration -- Comment out if not using Silicon Chip
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Logging
# import logging
# logging.basicConfig(level=logging.DEBUG)

class CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 8, kernel_size=3, stride=1, padding=1)
        # self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.fc1_float = nn.Linear(8 * 180 * 320, 32)  # Adjusted based on new input shape
        self.relu3_float = nn.ReLU()
        self.fc2_float = nn.Linear(32, 2)

        self.fc1 = nn.Linear(8 * 180 * 320, 32)  # Adjusted based on new input shape
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 22)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.relu1(x)
        x = self.pool(x)
        # x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)

        x_float = self.fc1_float(x)
        x_float = self.relu3_float(x_float)
        logits_float = self.fc2_float(x_float)

        x = self.fc1(x)
        x = self.relu3(x)
        logits = self.fc2(x)
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs, logits_float

def normalize(x):
    mean = torch.mean(x)
    x -= mean
    std = torch.std(x)
    x /= std
    return x

env = gym.make("MineRLObtainDiamondShovel-v0")
# env = gym.make('MineRLBasaltFindCave-v0')

input_shape = env.observation_space['pov'].shape
num_actions = 22

cnn = CNN(input_shape, num_actions)
optimizer = optim.Adam(cnn.parameters(), lr=0.00005)  # Set up optimizer

# env.seed(21)
state = env.reset()

done = False
num_episodes = 5  # Number of episodes to run
gamma = 0.99  # Discount factor for cumulative rewards
max_episode_steps = 500

progress_bar = tqdm(total=num_episodes, desc = "Episodes", unit="episode", ncols=80) # Set up progress bar for episodes

cumulativeRewards = []  # Store cumulative rewards per episode
losses = [] # Store loss per episode
episode_lengths = [] # Store number of actions/steps per episode

for episode in range(num_episodes):

    # Option to render the last episode
    if episode == num_episodes - 1:
        user_input = input("\nPress return or enter to continue or 'q' to quit.")

        if user_input.lower() == 'q':
            print("Continuing")
        else:
            ui_steps = int(input("Input number of steps for the last episode: "))
            max_episode_steps = ui_steps
            print("Rendering last Episode")
            env.render()

    episode_rewards = []  # Store episode rewards
    episode_log_probs = []
    steps = 0

    state = env.reset()  # Reset the environment and get initial observation

    while True:

        state = torch.tensor(state['pov'].copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        discreteActions, camera = cnn(state)

        camera = camera.tolist()[0]
        camera_y = torch.sigmoid(torch.tensor(camera[1]))

        d_action_dist = torch.distributions.Bernoulli(discreteActions)
        camera_dist = torch.distributions.Normal(camera[0], camera_y)

        action = d_action_dist.sample()  # Sample an action from the distribution
        cameraX = camera_dist.sample()
        cameraY = camera_dist.sample()

        action_dict = OrderedDict([
            ("ESC", np.array([0])),
            ("attack", np.array([int(action[0][0].item())])),
            ("back", np.array([int(action[0][1].item())])),
            ("camera", np.array([cameraX.item(), cameraY.item()], dtype=np.float32)),
            ("drop", np.array([int(action[0][2].item())])),
            ("forward", np.array([int(action[0][3].item())])),
            ("hotbar.1", np.array([int(action[0][4].item())])),
            ("hotbar.2", np.array([int(action[0][5].item())])),
            ("hotbar.3", np.array([int(action[0][6].item())])),
            ("hotbar.4", np.array([int(action[0][7].item())])),
            ("hotbar.5", np.array([int(action[0][8].item())])),
            ("hotbar.6", np.array([int(action[0][9].item())])),
            ("hotbar.7", np.array([int(action[0][10].item())])),
            ("hotbar.8", np.array([int(action[0][11].item())])),
            ("hotbar.9", np.array([int(action[0][12].item())])),
            ("inventory", np.array([int(action[0][13].item())])),
            ("jump", np.array([int(action[0][14].item())])),
            ("left", np.array([int(action[0][15].item())])),
            ("pickItem", np.array([int(action[0][16].item())])),
            ("right", np.array([int(action[0][17].item())])),
            ("sneak", np.array([int(action[0][18].item())])),
            ("sprint", np.array([int(action[0][19].item())])),
            ("swapHands", np.array([int(action[0][20].item())])),
            ("use", np.array([int(action[0][21].item())]))
        ])

        log_prob = d_action_dist.log_prob(action[0])  # Calculate the log probability of the chosen action
        log_prob_cameraX = camera_dist.log_prob(cameraX)  # Log probability of the chosen cameraX action
        log_prob_cameraY = camera_dist.log_prob(cameraY)  # Log probability of the chosen cameraY action

        next_state, reward, done, info = env.step(action_dict) # Take action in the environment

        # print("Action(s) Taken: ")
        # for key, value in action_dict.items():
        #     if np.any(value != 0):
        #         print(f"{key}: {value}")

        episode_rewards.append(reward)  # Store the reward

        log_probs = log_prob[0] + log_prob_cameraX + log_prob_cameraY

        episode_log_probs.append(log_probs)

        if done or steps >= max_episode_steps:
            discounted_rewards = np.zeros_like(episode_rewards)  # Array to store discounted rewards
            cumulative_rewards = 0
        
            # Calculate cumulative rewards using discounting
            for t in reversed(range(len(episode_rewards))):
                # Update cumulative rewards by discounting previous cumulative rewards and adding the current reward
                cumulative_rewards = cumulative_rewards * gamma + episode_rewards[t]
                # Store the calculated cumulative rewards in the discounted_rewards array
                discounted_rewards[t] = cumulative_rewards

            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
  
            episode_log_probs = torch.stack(episode_log_probs, dim = -1)

            loss = -torch.sum(episode_log_probs * discounted_rewards)

            optimizer.zero_grad() # Clear gradients - reset the gradients of the optimizer

            loss.backward() # Backpropagate - calculate gradients of the loss with respect to model parameters

            nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1.0)

            optimizer.step() # Update parameters - adjust model parameters based on gradients using optimizer

            print(f"Cumulative Rewards: {cumulative_rewards}")
            print(f"Loss: {loss.detach().item()}")

            cumulativeRewards.append(cumulative_rewards)
            episode_lengths.append(steps)
            losses.append(loss.detach().item())
            
            break

        state = next_state

        steps+=1

        if steps % 100 == 0:
            print(f"Steps: {steps}")

        # env.render()
        

    progress_bar.update(1)


progress_bar.close()

env.close()

rollingAverageWindow = int(input("Input rolling average window size: "))
# Calculate rolling average of rewards
rewardsRollingAverage = np.convolve(cumulativeRewards, np.ones(rollingAverageWindow)/rollingAverageWindow, mode='full')

# Plot episode rewards
plt.plot(cumulativeRewards, label="Rewards")
plt.plot(range(0, len(rewardsRollingAverage)), rewardsRollingAverage, label="Rolling Average")
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Reward Value")
plt.title("Episode Rewards and Running Average")
plt.legend()
plt.savefig("episode_rewards.png")
plt.close()

# Plot losses
losses_array = np.array(losses)
plt.plot(losses_array)
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Loss Value")
plt.title("Episode Losses")
plt.savefig("episode_losses.png")
plt.close()

# Plot episode lengths
plt.plot(episode_lengths)
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Amount of Steps/Actions")
plt.title("Episode Lengths")
plt.savefig("episode_lengths.png")
plt.close()