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
import model
import math
import csv
import wandb
import PE

start_time = time.time()

# Silicon Chip GPU Acceleration -- Comment out if not using a Silicon Chip
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Nvidia GPU Acceleration -- Comment out if not using a Nvidia GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print('Device: {}'.format(device))

input_shape = (380, 640, 3)
num_actions = 22

learning_rate = 0.00001

done = False
num_episodes = 2  # Number of episodes to run
gamma = 0.98  # Discount factor for cumulative rewards
max_episode_steps = 500
update_interval = 5
n_train_processes = 2

wandb.login()

# run = wandb.init(mode="disabled")

run = wandb.init(
    # Set the project where this run will be logged
    project="MineRL_A2CF",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epochs": num_episodes
    })

def normalize(x):
    x /= 255.0
    return x

progress_bar = tqdm(total=num_episodes, desc = "Episodes", unit="episode", ncols=80) # Set up progress bar for episodes

cumulativeRewards = []  # Store cumulative rewards per episode
cumulativeAverageRewards = []
cumulativeRewards1 = []
cumulativeRewards2 = []
episode_losses = [] # Store loss per episode
episode_cameraX_losses = []
episode_cameraY_losses = []
episode_actions_losses = []
episode_critic_losses = []
critic_losses = []
episode_lengths = [] # Store number of actions/steps per episode
totalGradients = []

if __name__ == '__main__':

    envs = PE.ParallelEnv(n_train_processes)

    cnn = model.ResNet50(input_shape[2], num_actions).to(device)
    # cnn = model_resNET.ResNet50(num_actions).to(device)
    # cnn = CNN(input_shape, num_actions).to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)  # Set up optimizer
    for episode in range(num_episodes):

        # Option to render the last episode
        # if episode == num_episodes - 1:
        #     user_input = input("\nPress return or enter to continue or 'q' to quit.")

        #     if user_input.lower() == 'q':
        #         print("Continuing")
        #     else:
        #         env.render()
        #         ui_steps = int(input("Input number of steps for the last episode: "))
        #         max_episode_steps = ui_steps
        #         print("Rendering last Episode")
                
        episode_rewards = 0  # Store episode rewards
        
        steps = 0
        prev_inventory = state['inventory']
        state = env.reset()  # Reset the environment and get initial observation
        env.seed(seed=1000)
        losses = []
        cameraX_losses = []
        cameraY_losses = []
        actions_losses = []

        averageRewards = []
        rewards1 = []
        rewards2= []

        while True:
            
            rewards = []  # Store episode rewards
            actions = []
            gradients = []
            states = []
            log_probs = []
            log_probs_cx = []
            log_probs_cy = []
            mask_lst = []

            for i in range(update_interval):

                img = torch.from_numpy(state).float()
                state = img.permute(0, 3, 1, 2).to(device)

                # Resize the image
                resize = transforms.Resize((state.shape[2] // 2, state.shape[3] // 2), antialias=False)
                state = resize(state)
                state = normalize(state)

                state = state.to(device)

                discreteActions, mean_x, std_x, mean_y, std_y = cnn.a(state)
                # print(f"Discrete Action Probs: {discreteActions}")
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

                # print(f"Camera X: {cameraX}")
                # print(f"Camera Y: {cameraY}")

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
                    ("sprint", np.array([int(action[0][10].item())])),
                    ("swapHands", np.array([int(action[0][20].item())])),
                    ("use", np.array([int(action[0][21].item())]))
                ])

                log_prob = d_action_dist.log_prob(action[0])  # Calculate the log probability of the chosen action
                log_prob_cameraX = camerax_dist.log_prob(cameraX)  # Log probability of the chosen cameraX action
                log_prob_cameraY = cameray_dist.log_prob(cameraY)  # Log probability of the chosen cameraY action

                next_state, reward, done, info = envs.step(action_dict) # Take action in the environment

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
                averageRewards.append(sum(reward))
                rewards1.append(reward[0])
                rewards2.append(reward[1])

                mask_lst.append(1-done)
                
                log_prob = d_action_dist.log_prob(action[0].to(device))[0]  # Calculate the log probability of the chosen action
                log_prob_cameraX = camerax_dist.log_prob(cameraX.to(device))  # Log probability of the chosen cameraX action
                log_prob_cameraY = cameray_dist.log_prob(cameraY.to(device))  # Log probability of the chosen cameraY action

                log_probs_cx.append(log_prob_cameraX)
                log_probs_cy.append(log_prob_cameraY)
                log_probs.append(log_prob)

                state = next_state

                steps+=1
            
            valueF = torch.from_numpy(next_state).float().permute(0, 3, 1, 2).to(device)
            # print(valueF.shape)

            # Resize the image
            resize = transforms.Resize((valueF.shape[2] // 2, valueF.shape[3] // 2), antialias = False)
            valueF = resize(valueF)
            valueF = normalize(valueF)

            valueF_a, valueF_cx, valueF_cy = cnn.v(valueF)
            valueF_a = valueF_a.valueF_m.detach().cpu().clone().numpy()
            valueF_cy = valueF_cy.valueF_m.detach().cpu().clone().numpy()
            valueF_cx = valueF_cx.valueF_m.detach().cpu().clone().numpy()

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
            
            log_probs = torch.stack(log_probs, dim = -1).to(device).reshape(-1)
            log_probs_cx = torch.stack(log_probs_cx, dim = -1).to(device).reshape(-1)
            log_probs_cy = torch.stack(log_probs_cy, dim = -1).to(device).reshape(-1)

            # print(log_probs.shape, discounted_rewards.shape)
            # print(log_probs_cx.shape)
            # print(log_probs_cy.shape)
            print(states)
            statesVec = torch.stack(states).float().reshape(-1, 3, 320, 180).to(device)
            # statesVec = torch.squeeze(statesVec, dim=1).to(device)

            v_a, v_cx, v_cy = cnn.v(statesVec)

            advantage_a = target_a.unsqueeze(1).to(device) - v_a

            advantage_cx = target_cx.unsqueeze(1).to(device) - v_cx
            advantage_cy = target_cy.unsqueeze(1).to(device) - v_cy

            # actions_loss = -(torch.mean(log_probs * discounted_rewards))
            # cameraX_loss = -(torch.mean(log_probs_cx * discounted_rewards))
            # cameraY_loss = -(torch.mean(log_probs_cy * discounted_rewards))
            # print(f"Log Probs: {log_probs}")
            # print(f"Advantage Actions: {advantage_a}")
            actions_loss = -(torch.mean(log_probs * advantage_a))
            cameraX_loss = -(torch.mean(log_probs_cx * advantage_cx))
            cameraY_loss = -(torch.mean(log_probs_cy * advantage_cy))
            value_loss = (F.smooth_l1_loss(v_cx.reshape(-1), target_cx.to(device))) + F.smooth_l1_loss(v_cy.reshape(-1), target_cy.to(device)) + F.smooth_l1_loss(v_a.reshape(-1), target_a.to(device))
            # print(f"Discounted Rewards: {torch.sum(discounted_rewards)}")
            # print(f"Action Loss: {actions_loss}")
            # print(f"Camera X Loss: {cameraX_loss}")
            # print(f"Camera Y Loss: {cameraY_loss}")
            # print(f"Value Loss: {value_loss}")

            loss = actions_loss + cameraX_loss + cameraY_loss + value_loss
            print(f"Loss: {loss}")

            optimizer.zero_grad() # Clear gradients - reset the gradients of the optimizer

            loss.backward() # Backpropagate - calculate gradients of the loss with respect to model parameters

            nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=10000)

            total_norm = 0
            for p in cnn.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm()  # Calculate the L2 norm of gradients
                    total_norm += param_norm.item() ** 2  # Accumulate the squared norm
            total_norm = total_norm ** 0.5  # take the square root to get the total norm
            print(f"Total gradient norm: {total_norm}\n")
            gradients.append(total_norm)

            optimizer.step() # Update parameters - adjust model parameters based on gradients using optimizer

            actions_losses.append(actions_loss)
            cameraX_losses.append(cameraX_loss)
            cameraY_losses.append(cameraY_loss)
            critic_losses.append(value_loss)
            losses.append(loss)

            # print(NotInInventory)
            # print(f"Cumulative Rewards: {cumulative_rewards}")
            # print(f"Loss: {loss.detach().item()}")
            if done or steps >= max_episode_steps:
                
                cumulativeRewards1.append(sum(rewards1))
                cumulativeRewards2.append(sum(rewards1))
                episodeReward = sum(averageRewards)
                episodeAverageReward = episodeReward/n_train_processes
                cumulativeAverageRewards.append(episodeAverageReward)
                cumulativeRewards.append(episodeReward)

                episode_lengths.append(steps)

                losses_mean = torch.tensor(losses)
                losses_mean = torch.mean(losses_mean)
                episode_losses.append(losses_mean.item())

                losses_mean = torch.tensor(cameraX_losses)
                losses_mean = torch.mean(losses_mean)
                episode_cameraX_losses.append(losses_mean.item())

                losses_mean = torch.tensor(cameraY_losses)
                losses_mean = torch.mean(losses_mean)
                episode_cameraY_losses.append(losses_mean.item())

                losses_mean = torch.tensor(critic_losses)
                losses_mean = torch.mean(losses_mean)
                episode_critic_losses.append(losses_mean.item())

                losses_mean = torch.tensor(actions_losses)
                losses_mean = torch.mean(losses_mean)
                episode_actions_losses.append(losses_mean.item())

                plt.plot(cumulativeRewards, label="Rewards")
                plt.xlim(0, num_episodes)
                plt.xlabel("Episode Number")
                plt.ylabel("Reward Value")
                plt.title("Episode Rewards and Running Average")
                plt.legend()
                plt.savefig("episode_rewards.png")
                plt.close()

                # Plot losses
                losses_array = np.array([tensor for tensor in episode_losses])
                plt.plot(losses_array)
                plt.xlim(0, num_episodes)
                plt.xlabel("Episode Number")
                plt.ylabel("Loss Value")
                plt.title("Episode Losses")
                plt.savefig("episode_losses.png")
                plt.close()

                losses_mean = torch.tensor(cameraY_losses)
                losses_mean = torch.mean(losses_mean)

                # Plot losses
                losses_array = np.array([tensor for tensor in episode_cameraX_losses])
                plt.plot(losses_array)
                plt.xlim(0, num_episodes)
                plt.xlabel("Episode Number")
                plt.ylabel("Loss Value")
                plt.title("Episode CameraX Losses")
                plt.savefig("episode_CameraX_losses.png")
                plt.close()

                losses_mean = torch.tensor(cameraX_losses)
                losses_mean = torch.mean(losses_mean)

                # Plot losses
                losses_array = np.array([tensor for tensor in episode_cameraY_losses])
                plt.plot(losses_array)
                plt.xlim(0, num_episodes)
                plt.xlabel("Episode Number")
                plt.ylabel("Loss Value")
                plt.title("Episode CameraY Losses")
                plt.savefig("episode_CameraY_losses.png")
                plt.close()

                losses_mean = torch.tensor(actions_losses)
                losses_mean = torch.mean(losses_mean)

                # Plot losses
                losses_array = np.array([tensor for tensor in episode_critic_losses])
                plt.plot(losses_array)
                plt.xlim(0, num_episodes)
                plt.xlabel("Episode Number")
                plt.ylabel("Loss Value")
                plt.title("Episode Critic Losses")
                plt.savefig("episode_critic_losses.png")
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
                
                wandb.log({"Length": steps, "Average Rewards": episodeAverageReward, "Loss": sum(l), "Total Norm": sum(gradients)})

                break

            # if steps % 100 == 0:
            #     print(f"Steps: {steps}")

            # envs.render()
            

        progress_bar.update(1)


    progress_bar.close()

    envs.close()

# Plot losses
losses_array = np.array(episode_losses)
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

rollingAverageWindow = 2 # int(input("Input rolling average window size: "))
# Calculate rolling average of rewards
print(cumulativeAverageRewards)
rewardsRollingAverage = np.convolve(cumulativeAverageRewards, np.ones(rollingAverageWindow)/rollingAverageWindow, mode='same')

# Plot episode rewards
plt.plot(cumulativeAverageRewards, label="Rewards")
plt.plot(range(0, len(rewardsRollingAverage)), rewardsRollingAverage, label="Rolling Average")
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Reward Value")
plt.title("Mean Episode Rewards and Running Average")
plt.legend()
plt.savefig("mean_episode_rewards.png")
plt.close()

rewardsRollingAverage = np.convolve(cumulativeRewards, np.ones(rollingAverageWindow)/rollingAverageWindow, mode='same')

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

rewardsRollingAverage = np.convolve(cumulativeRewards1, np.ones(rollingAverageWindow)/rollingAverageWindow, mode='same')

# Plot episode rewards
plt.plot(cumulativeRewards1, label="Rewards")
plt.plot(range(0, len(rewardsRollingAverage)), rewardsRollingAverage, label="Rolling Average")
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Reward Value")
plt.title("Episode Rewards and Running Average")
plt.legend()
plt.savefig("episode_rewards1.png")
plt.close()

rewardsRollingAverage = np.convolve(cumulativeRewards2, np.ones(rollingAverageWindow)/rollingAverageWindow, mode='same')

# Plot episode rewards
plt.plot(cumulativeRewards2, label="Rewards")
plt.plot(range(0, len(rewardsRollingAverage)), rewardsRollingAverage, label="Rolling Average")
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Reward Value")
plt.title("Episode Rewards and Running Average")
plt.legend()
plt.savefig("episode_rewards2.png")
plt.close()

end_time = time.time()
runtime = end_time - start_time
print("Code runtime:", runtime, "seconds")

# Path to the CSV file
csv_file_path = "average_rewards.csv"

# Write the rewards to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Episode", "Reward"])  # Write header

    for episode, reward in enumerate(cumulativeAverageRewards, start=1):
        csv_writer.writerow([episode, reward])
        
print("Rewards saved to: ", csv_file_path)
# Path to the CSV file
csv_file_path = "rewards.csv"

# Write the rewards to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Episode", "Reward"])  # Write header

    for episode, reward in enumerate(cumulativeRewards, start=1):
        csv_writer.writerow([episode, reward])
        
print("Rewards saved to: ", csv_file_path)

# Path to the CSV file
csv_file_path = "rewards1.csv"

# Write the rewards to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Episode", "Reward"])  # Write header

    for episode, reward in enumerate(cumulativeRewards1, start=1):
        csv_writer.writerow([episode, reward])
        
print("Rewards saved to: ", csv_file_path)

# Path to the CSV file
csv_file_path = "rewards2.csv"

# Write the rewards to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Episode", "Reward"])  # Write header

    for episode, reward in enumerate(cumulativeRewards2, start=1):
        csv_writer.writerow([episode, reward])
        
print("Rewards saved to: ", csv_file_path)

print("Successfully Run!")

end_time = time.time()
runtime = end_time - start_time
print("Code runtime:", runtime, "seconds")