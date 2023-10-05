# Imports
import gym
import minerl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict 
import torchvision.transforms as transforms
import time

start_time = time.time()

# Silicon Chip GPU Acceleration -- Comment out if not using a Silicon Chip
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Nvidia GPU Acceleration -- Comment out if not using a Nvidia GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print('Device: {}'.format(device))

# Logging
# import logging
# logging.basicConfig(level=logging.DEBUG)

class CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 8, kernel_size=7, stride=2, padding=3)  # Increase stride to 2
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Increase pooling size and stride
        self.conv2 = nn.Conv2d(8, 16, kernel_size=7, stride=2, padding=3)

        self.fc1x_float = nn.Linear(14720, 32)
        self.relu3x_float = nn.ReLU()
        self.fc2x_mean = nn.Linear(32, 1)
        self.fc2x_std = nn.Linear(32, 1)

        self.fc1y_float = nn.Linear(14720, 32)
        self.relu3y_float = nn.ReLU()
        self.fc2y_mean = nn.Linear(32, 1)
        self.fc2y_std = nn.Linear(32, 1)

        self.fc1 = nn.Linear(14720, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_actions)

        self.fcv = nn.Linear(14720, 64)
        self.relu4 = nn.ReLU()
        self.fc_v_a = nn.Linear(64, 1)
        self.fc_v_cx = nn.Linear(64, 1)
        self.fc_v_cy = nn.Linear(64, 1)

    def a(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        
        x = x.view(x.size(0), -1)

        x_floatx = self.fc1x_float(x)
        x_floatx = self.relu3x_float(x_floatx)
        mean_x = torch.tanh(self.fc2x_mean(x_floatx))
        std_x = torch.sigmoid(self.fc2x_std(x_floatx)) + 1e-10  # Ensure std is positive

        x_floaty = self.fc1y_float(x)
        x_floaty = self.relu3y_float(x_floaty)
        mean_y = torch.tanh(self.fc2y_mean(x_floaty))
        std_y = torch.sigmoid(self.fc2y_std(x_floaty)) + 1e-10  # Ensure std is positive

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        logits = self.fc3(x)
        action_probs = torch.softmax(logits, dim=-1)

        return action_probs, mean_x, std_x, mean_y, std_y
    
    def v(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        # x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fcv(x)
        x = self.relu4(x)
        v_cx = self.fc_v_cx(x)
        v_cy = self.fc_v_cy(x)
        v_a = self.fc_v_a(x)
        return v_cx, v_cy, v_a

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

learning_rate = 0.001

cnn = CNN(input_shape, num_actions).to(device)
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)  # Set up optimizer

# env.seed(21)
state = env.reset()

done = False
num_episodes = 25  # Number of episodes to run
gamma = 0.99  # Discount factor for cumulative rewards
max_episode_steps = 200

progress_bar = tqdm(total=num_episodes, desc = "Episodes", unit="episode", ncols=80) # Set up progress bar for episodes

cumulativeRewards = []  # Store cumulative rewards per episode
losses = [] # Store loss per episode
episode_lengths = [] # Store number of actions/steps per episode
totalGradients = []

prev_inventory = {}


for episode in range(num_episodes):

    # Option to render the last episode
    # if episode == num_episodes - 1:
    #     user_input = input("\nPress return or enter to continue or 'q' to quit.")

    #     if user_input.lower() == 'q':
    #         print("Continuing")
    #     else:
    #         ui_steps = int(input("Input number of steps for the last episode: "))
    #         max_episode_steps = ui_steps
    #         print("Rendering last Episode")
    #         env.render()

    rewards = []  # Store episode rewards
    episode_log_probs = []
    log_probs = []
    log_probs_cx = []
    log_probs_cy = []
    gradients = []
    actions = []
    states = []
    mask_lst = []
    steps = 0

    state = env.reset()  # Reset the environment and get initial observation

    while True:

        state = torch.tensor(state['pov'].copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        # Resize the image
        resize = transforms.Resize((state.shape[2] // 2, state.shape[3] // 2), antialias=False)
        state = resize(state)
        state = normalize(state)

        state = state.to(device)

        discreteActions, mean_x, std_x, mean_y, std_y = cnn.a(state)

        d_action_dist = torch.distributions.Bernoulli(probs = discreteActions)
        camerax_dist = torch.distributions.Normal(mean_x, std_x)
        cameray_dist = torch.distributions.Normal(mean_y, std_y)

        # discreteActions, camera = cnn(state)

        # camera = camera.tolist()[0]
        # camera_y = torch.sigmoid(torch.tensor(camera[1]))

        # d_action_dist = torch.distributions.Bernoulli(discreteActions)
        # camera_dist = torch.distributions.Normal(camera[0], camera_y)

        action = d_action_dist.sample()  # Sample an action from the distribution
        cameraX = camerax_dist.sample()*20
        cameraY = cameray_dist.sample()*20

        cameraX = torch.clamp(cameraX, -20, 20)
        cameraY = torch.clamp(cameraY, -20, 20)

        print(f"Camera X: {cameraX}")
        print(f"Camera Y: {cameraY}")

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
        log_prob_cameraX = camerax_dist.log_prob(cameraX)  # Log probability of the chosen cameraX action
        log_prob_cameraY = cameray_dist.log_prob(cameraY)  # Log probability of the chosen cameraY action

        next_state, reward, done, info = env.step(action_dict) # Take action in the environment

        if action_dict['sprint'] == 1:
            reward += 0.02

        if action_dict['forward'] == 1:
            reward += 0.015

        if action_dict['right'] == 1:
            reward += 0.02

        if action_dict['left'] == 1:
            reward += 0.02

        if action_dict['jump'] == 1:
            reward += 0.015

        if action_dict['attack'] == 1:
            reward += 0.035

        raw_values = []
        for value in action_dict.values():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                raw_values.append(value)
            else:
                raw_values.append(np.array([value]))

        # Convert the list to a NumPy array
        act = np.concatenate(raw_values)

        actions.append(act)

        states.append(state)

        rewards.append(reward)  # Store the reward

        mask_lst.append(1-done)
        
        log_prob = d_action_dist.log_prob(action[0].to(device))[0]  # Calculate the log probability of the chosen action
        log_prob_cameraX = camerax_dist.log_prob(cameraX.to(device))  # Log probability of the chosen cameraX action
        log_prob_cameraY = cameray_dist.log_prob(cameraY.to(device))  # Log probability of the chosen cameraY action

        log_probs_cx.append(log_prob_cameraX)
        log_probs_cy.append(log_prob_cameraY)
        log_probs.append(log_prob)
        
        if done or steps >= max_episode_steps:
            valueF = torch.tensor(next_state['pov'].copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            # print(valueF.shape)

            # Resize the image
            resize = transforms.Resize((valueF.shape[2] // 2, valueF.shape[3] // 2), antialias = False)
            valueF = resize(valueF)
            # valueF = normalize(valueF)

            valueF_a, valueF_cx, valueF_cy = cnn.v(valueF)
            valueF_a = valueF_a.to(device).detach()
            valueF_cy = valueF_cy.to(device).detach()
            valueF_cx = valueF_cx.to(device).detach()

            # Target
            G = valueF_a.reshape(-1)
            target_a = []

            # Calculate cumulative rewards using discounting
            for r, mask in zip(rewards[::-1], mask_lst[::-1]):
                G = r + gamma * G * mask
                target_a.append(G)

            target_a = torch.tensor(target_a[::-1], dtype=torch.float32)

            # Target
            G = valueF_cx.reshape(-1)
            target_cx = []

            # Calculate cumulative rewards using discounting
            for r, mask in zip(rewards[::-1], mask_lst[::-1]):
                G = r + gamma * G * mask
                target_cx.append(G)

            target_cx = torch.tensor(target_cx[::-1], dtype=torch.float32)

            # Target
            G = valueF_cy.reshape(-1)
            target_cy = []

            # Calculate cumulative rewards using discounting
            for r, mask in zip(rewards[::-1], mask_lst[::-1]):
                G = r + gamma * G * mask
                target_cy.append(G)

            target_cy = torch.tensor(target_cy[::-1], dtype=torch.float32)
            discounted_rewards = np.zeros_like(rewards)  # Array to store discounted rewards
            cumulative_rewards = 0
        
            # Calculate cumulative rewards using discounting
            for t in reversed(range(len(rewards))):
                # Update cumulative rewards by discounting previous cumulative rewards and adding the current reward
                cumulative_rewards = cumulative_rewards * gamma + rewards[t]
                # Store the calculated cumulative rewards in the discounted_rewards array
                discounted_rewards[t] = cumulative_rewards

            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device).unsqueeze(1)
            log_probs = torch.stack(log_probs).to(device)
            log_probs_cx = torch.stack(log_probs_cx).to(device)
            log_probs_cy = torch.stack(log_probs_cy).to(device)

            # print(log_probs.shape, discounted_rewards.shape)
            # print(log_probs_cx.shape)
            # print(log_probs_cy.shape)

            statesVec = torch.stack(states)
            statesVec = torch.squeeze(statesVec, dim=1).to(device)

            v_a, v_cx, v_cy = cnn.v(statesVec)

            advantage_a = target_a.unsqueeze(1).to(device) - v_a

            advantage_cx = target_cx.unsqueeze(1).to(device) - v_cx
            advantage_cy = target_cy.unsqueeze(1).to(device) - v_cy

            # actions_loss = -(torch.mean(log_probs * discounted_rewards))
            # cameraX_loss = -(torch.mean(log_probs_cx * discounted_rewards))
            # cameraY_loss = -(torch.mean(log_probs_cy * discounted_rewards))
            actions_loss = -(torch.mean(log_probs * advantage_a))
            cameraX_loss = -(torch.mean(log_probs_cx * advantage_cx))
            cameraY_loss = -(torch.mean(log_probs_cy * advantage_cy))
            value_loss = (F.smooth_l1_loss(v_cx.reshape(-1), target_cx.to(device))) + F.smooth_l1_loss(v_cy.reshape(-1), target_cy.to(device)) + F.smooth_l1_loss(v_a.reshape(-1), target_a.to(device))
            print(f"Discounted Rewards: {torch.sum(discounted_rewards)}")
            print(f"Action Loss: {actions_loss}")
            print(f"Camera X Loss: {cameraX_loss}")
            print(f"Camera Y Loss: {cameraY_loss}")

            loss = actions_loss + cameraX_loss + cameraY_loss + value_loss

            optimizer.zero_grad() # Clear gradients - reset the gradients of the optimizer

            loss.backward() # Backpropagate - calculate gradients of the loss with respect to model parameters

            total_norm = 0
            for p in cnn.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm()  # Calculate the L2 norm of gradients
                    total_norm += param_norm.item() ** 2  # Accumulate the squared norm
            total_norm = total_norm ** 0.5  # take the square root to get the total norm
            # print(f"Total gradient norm: {total_norm}\n")
            gradients.append(total_norm)

            nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=500)

            optimizer.step() # Update parameters - adjust model parameters based on gradients using optimizer

            # print(f"Cumulative Rewards: {cumulative_rewards}")
            # print(f"Loss: {loss.detach().item()}")

            cumulativeRewards.append(cumulative_rewards)
            episode_lengths.append(steps)
            losses.append(loss.detach().item())
            gradients_mean = torch.tensor(gradients)
            gradients_mean = torch.mean(gradients_mean)
            totalGradients.append(gradients_mean)
            
            break

        state = next_state

        steps+=1

        if steps % 100 == 0:
            print(f"Steps: {steps}")

        env.render()
        

    progress_bar.update(1)


progress_bar.close()

env.close()

rollingAverageWindow = 3#int(input("Input rolling average window size: "))
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

# Plot gradients
gradients_array = np.array([tensor.detach().cpu().numpy() for tensor in totalGradients])
plt.plot(gradients_array)
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Gradient Value")
plt.title("Gradients")
plt.savefig("episode_gradients.png")
plt.close()

end_time = time.time()
runtime = end_time - start_time
print("Code runtime:", runtime, "seconds")