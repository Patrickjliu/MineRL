import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import racecar_gym.envs.gym_api
import numpy.core.multiarray
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import wandb
import csv
import model
from PIL import Image
import PE

start_time = time.time()

# Nvidia GPU cceleration -- Comment out if not using a Nvidia GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device: {}".format(device))

env_id = 'SingleAgentAustria-v0'
render_mode = 'rgb_array_birds_eye'
reset_options = dict(mode='grid')
scenario = "torino.yml"

input_shape = env.observation_space['pov'].shape
num_actions = 22

learning_rate = 0.00001

done = False
num_episodes = 500  # Number of episodes to run
gamma = 0.98  # Discount factor for cumulative rewards
update_interval = 5
n_train_processes = 2

wandb.login()

# run = wandb.init(mode="disabled")

run = wandb.init(
    # Set the project where this run will be logged
    project="MineRL_A2C",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epochs": num_episodes
    })

def normalize(x):
    x /= 255.0
    return x

cumulativeRewards = []  # Store cumulative rewards per episode
cumulativeRewards1 = []
cumulativeRewards2 = []
# cumulativeRewards3 = []
# cumulativeRewards4 = []
# cumulativeRewards5 = []
episode_losses = [] # Store loss per episode
episode_cameraX_losses = []
episode_cameraY_losses = []
episode_actions_losses = []
episode_critic_losses = []
critic_losses = []
episode_lengths = [] # Store number of actions/steps per episode
totalGradients = []

progress_bar = tqdm(total=num_episodes, desc = "Episodes", unit="episode", ncols=80) # Set up progress bar for episodes

if __name__ == '__main__':

    envs = PE.ParallelEnv(n_train_processes)

    cnn = model.ResNet50(input_shape[2], num_actions).to(device)
    # cnn = model_resNET.ResNet50(num_actions).to(device)
    # cnn = CNN(input_shape, num_actions).to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)  # Set up optimizer

    for episode in range(num_episodes):

        motor_action = []
        steering_action = []
        l = []
        
        # rewards3= []
        # rewards4= []
        # rewards5= []

        steps = 0
        
        done = False
        obs = envs.reset()
        state = obs
            
        while True:
        
            gradients = []
            states = []
            episode_log_probs_steering = []
            episode_log_probs_motor = []
            mask_lst = []
            
            for i in range(update_interval):
                
                pose_tensors = [torch.tensor(entry['pose'], dtype=torch.float32).to(device) for entry in state]
                velocity_tensors = [torch.tensor(entry['velocity'], dtype=torch.float32).to(device) for entry in state]
                acceleration_tensors = [torch.tensor(entry['acceleration'], dtype=torch.float32).to(device) for entry in state]
                lidar_tensors = [torch.tensor(entry['lidar'], dtype=torch.float32).to(device) for entry in state]
                img_tensors = [normalize(torch.from_numpy(entry['rgb_camera']).permute(2, 0, 1).to(device).float()) for entry in state]
                
                pose_tensors = torch.stack(pose_tensors, dim=0)
                velocity_tensors = torch.stack(velocity_tensors, dim=0)
                acceleration_tensors = torch.stack(acceleration_tensors, dim=0)
                lidar_tensors = torch.stack(lidar_tensors, dim=0)
                # img_tensors = np.array(img_tensors)
                img_tensors = torch.stack(img_tensors, dim=0)

                mean_steering, std_steering, mean_motor, std_motor = cnn.a(img_tensors, pose_tensors, velocity_tensors, acceleration_tensors, lidar_tensors)


                motor_dist = torch.distributions.Normal(mean_motor, std_motor)
                steering_dist = torch.distributions.Normal(mean_steering, std_steering)
                motor = motor_dist.sample()
                steering = steering_dist.sample()
                
                motor = torch.clamp(motor, 0, 1).cpu().numpy()
                steering = torch.clamp(steering, -1, 1).cpu().numpy()
                
                next_state, reward, dones, truncated, info = envs.step(steering, motor)
    
                reward *= 9
    
                for i in range(n_train_processes):
                    wandb.log({f"Steering Action {i}": steering[i], f"Motor Action {i}": motor[i]})

                mask_lst.append(1-dones)
                states.append(img_tensors)
                rewards.append(sum(reward)/n_train_processes)  # Store the reward
                rewards1.append(reward[0])
                rewards2.append(reward[1])
                rewards3.append(reward[2])
                rewards4.append(reward[3])
                rewards5.append(reward[4])
                episode_log_probs_motor.append(motor_dist.log_prob(torch.tensor(motor).to(device)))
                episode_log_probs_steering.append(steering_dist.log_prob(torch.tensor(steering).to(device)))
                
                steps+=1

            rgb_images = [entry['rgb_camera'] for entry in next_state]

            img_tensors_np = np.array(rgb_images)

            img = torch.from_numpy(img_tensors_np).to(device).float()
            
            img = normalize(img.permute(0, 3, 1, 2).to(device))
                
            valueF_m, valueF_a = cnn.v(img)
            valueF_m = valueF_m.detach().cpu().clone().numpy()
            valueF_a = valueF_a.detach().cpu().clone().numpy()

            # Target
            G = valueF_a.reshape(-1)
            target_a = []

            # Calculate cumulative rewards using discounting
            for r, mask in zip(rewards[::-1], mask_lst[::-1]):
                G = r + gamma * G * mask
                target_a.append(G)
            
            target_a = torch.tensor(np.array(target_a[::-1]), dtype=torch.float32).reshape(-1)
                                                                        
            G = valueF_m.reshape(-1)
            target_m = []

            # Calculate cumulative rewards using discounting
            for r, mask in zip(rewards[::-1], mask_lst[::-1]):
                G = r + gamma * G * mask
                target_m.append(G)
            
            target_m = torch.tensor(np.array(target_m[::-1]), dtype=torch.float32).reshape(-1)
            
            statesVec = torch.stack(states).float().reshape(-1, 3,128,128).to(device)
                                                                        
            mv, sv = cnn.v(statesVec)
 
            advantage_a = target_a.to(device) - sv
                                                                        
            advantage_m = target_m.to(device) - mv

            episode_log_probs_motor = torch.stack(episode_log_probs_motor, dim = -1).to(device).reshape(-1)
            episode_log_probs_steering = torch.stack(episode_log_probs_steering, dim = -1).to(device).reshape(-1)

            vm, vs = cnn.v(statesVec)
            
            steering_loss = -torch.mean(episode_log_probs_steering * advantage_a)
            motor_loss = -torch.mean(episode_log_probs_motor * advantage_m)
            value = F.smooth_l1_loss(vm.reshape(-1), target_a.to(device)) + F.smooth_l1_loss(vs.reshape(-1), target_m.to(device))
            loss = steering_loss + value + motor_loss
        
            optimizer.zero_grad() # Clear gradients - reset the gradients of the optimizer

            loss.backward() # Backpropagate - calculate gradients of the loss with respect to model parameters

            nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=10000)

            optimizer.step() # Update parameters - adjust model parameters based on gradients using optimizer

            total_norm = 0
            for p in cnn.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm()  # Calculate the L2 norm of gradients
                    total_norm += param_norm.item() ** 2  # ccumulate the squared norm
            total_norm = total_norm ** 0.5

            gradients.append(total_norm)

            l.append(loss.detach().item())

            if np.any(dones) or np.any(truncated):
                totalGradients.append(sum(gradients))
                
                episodeReward = sum(rewards).item()
                episodeReward1 = float(sum(rewards1))
                episodeReward2 = float(sum(rewards2))
                episodeReward3 = float(sum(rewards3))
                episodeReward4 = float(sum(rewards4))
                episodeReward5 = float(sum(rewards5))

                cumulativeRewards.append(episodeReward)
                cumulativeRewards1.append(episodeReward1)
                cumulativeRewards2.append(episodeReward2)
                cumulativeRewards3.append(episodeReward3)
                cumulativeRewards4.append(episodeReward4)
                cumulativeRewards5.append(episodeReward5)

                episode_lengths.append(steps)

                losses.append(sum(l))

                wandb.log({"Length": steps, "Average Rewards": episodeReward, "Loss": sum(l), "Total Norm": sum(gradients)})
                break
     
            state = next_state
            
        progress_bar.update(1)

    envs.close()
    
# Data/Plotting Code
    
rollingAverageWindow = 20 # int(input("Input rolling average window size: "))
# Calculate rolling average of rewards
rewardsRollingAverage = np.convolve(cumulativeRewards, np.ones(rollingAverageWindow)/rollingAverageWindow, mode='same')

# Plot episode rewards
plt.plot(cumulativeRewards, label="Rewards")
plt.plot(range(0, len(rewardsRollingAverage)), rewardsRollingAverage, label="Rolling Average")
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Reward Value")
plt.title("Mean Episode Rewards and Running Average")
plt.legend()
plt.savefig("mean_episode_rewards.png")
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

rewardsRollingAverage = np.convolve(cumulativeRewards3, np.ones(rollingAverageWindow)/rollingAverageWindow, mode='same')

# Plot episode rewards
plt.plot(cumulativeRewards3, label="Rewards")
plt.plot(range(0, len(rewardsRollingAverage)), rewardsRollingAverage, label="Rolling Average")
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Reward Value")
plt.title("Episode Rewards and Running Average")
plt.legend()
plt.savefig("episode_rewards3.png")
plt.close()

rewardsRollingAverage = np.convolve(cumulativeRewards4, np.ones(rollingAverageWindow)/rollingAverageWindow, mode='same')

# Plot episode rewards
plt.plot(cumulativeRewards4, label="Rewards")
plt.plot(range(0, len(rewardsRollingAverage)), rewardsRollingAverage, label="Rolling Average")
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Reward Value")
plt.title("Episode Rewards and Running Average")
plt.legend()
plt.savefig("episode_rewards4.png")
plt.close()

rewardsRollingAverage = np.convolve(cumulativeRewards5, np.ones(rollingAverageWindow)/rollingAverageWindow, mode='same')

# Plot episode rewards
plt.plot(cumulativeRewards5, label="Rewards")
plt.plot(range(0, len(rewardsRollingAverage)), rewardsRollingAverage, label="Rolling Average")
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Reward Value")
plt.title("Episode Rewards and Running Average")
plt.legend()
plt.savefig("episode_rewards5.png")
plt.close()

# print(losses)
# Plot losses
losses_array = np.array([tensor for tensor in losses])
plt.plot(losses_array)
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Loss Value")
plt.title("Episode Losses")
plt.savefig("episode_losses.png")
plt.close()

# Plot gradients
gradients_array = np.array([tensor for tensor in totalGradients])
plt.plot(gradients_array)
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("Gradient Value")
plt.title("Gradients")
plt.savefig("episode_gradients.png")
plt.close()

# Plot episode lengths
plt.plot(episode_lengths)
plt.xlim(0, num_episodes)
plt.xlabel("Episode Number")
plt.ylabel("mount of Steps/ctions")
plt.title("Episode Lengths")
plt.savefig("episode_lengths.png")
plt.close()

end_time = time.time()
runtime = end_time - start_time
print("Code runtime:", runtime, "seconds")

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

# Path to the CSV file
csv_file_path = "rewards3.csv"

# Write the rewards to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Episode", "Reward"])  # Write header

    for episode, reward in enumerate(cumulativeRewards3, start=1):
        csv_writer.writerow([episode, reward])
        
print("Rewards saved to: ", csv_file_path)

# Path to the CSV file
csv_file_path = "rewards4.csv"

# Write the rewards to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Episode", "Reward"])  # Write header

    for episode, reward in enumerate(cumulativeRewards4, start=1):
        csv_writer.writerow([episode, reward])
        
print("Rewards saved to: ", csv_file_path)

# Path to the CSV file
csv_file_path = "rewards5.csv"

# Write the rewards to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Episode", "Reward"])  # Write header

    for episode, reward in enumerate(cumulativeRewards5, start=1):
        csv_writer.writerow([episode, reward])
        
print("Rewards saved to: ", csv_file_path)

print("Successfully Run!")