import gym
from time import sleep
from gym.spaces import Discrete
from gym.wrappers import Monitor

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import minerl
import wandb

# import logging
# logging.basicConfig(level=logging.DEBUG)

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 60000,
    "env_name": "MineRLObtainDiamondShovel-v0",
    "run_name": "PPO_MineRLObtainDiamondShovel-v0"
}

run = wandb.init(
    project="MineRL PPO SB",
    name=config["run_name"],
    config=config,
    sync_tensorboard=True, 
    monitor_gym=True,
    save_code=True,  
)

def make_env():
    try:
        env = gym.make(config["env_name"])
        env = BitMaskWrapper(env)  # Apply BitMaskWrapper first
        env = Monitor(env, directory="monitor_results", force=True)  # Then apply Monitor
        print("Nuevo entorno creado!!!")

    except TimeoutError:
        print("Ha ocurrido un TimeoutError. Intentando volver a crear el entorno...\n")
        env.close()
        env = make_env()  # Recreate the environment
        sleep(30)  # Adding a delay to ensure proper cleanup
    
    return env

class BitMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BitMaskWrapper, self).__init__(env)
        self.orig_action_space = self.action_space
        self.action_space = gym.spaces.Discrete(32)  # Modify the action space to Discrete(32)
        self.observation_space = self.observation_space['pov']
        self.noop_action = self.orig_action_space.noop()  # Pre-calculate no-op action

    
    def step(self, action):
        while True:  # Keep trying to step until successful
            try:
                assert 0 <= action < 64, "Invalid action"
                masked_action = self._apply_bit_mask(action)
                obs, reward, done, info = self.env.step(masked_action)
                if info:  # Print info only if it's not empty
                    print("Info dictionary:", info)
                obs = obs["pov"]
                obs = obs / 255.0
                return obs, reward, done, info
            except TimeoutError:
                print("Ha ocurrido un TimeoutError. Intentando volver a crear el entorno...\n")
                self.env.close()
                self.env = make_env()  # Recreate the environment
                sleep(30)  # Adding a delay to ensure proper cleanup

    def reset(self, **kwargs):
        while True:  # Keep trying to reset until successful
            try:
                obs = self.env.reset(**kwargs)
                obs = obs["pov"]
                obs = obs / 255.0
                return obs
            except TimeoutError:
                print("Ha ocurrido un TimeoutError durante el reset. Intentando volver a crear el entorno...")
                self.env.close()
                self.env = make_env()  # Recreate the environment
                sleep(30)  # Adding a delay to ensure proper cleanup


    def _apply_bit_mask(self, action):
        """Applies the bit mask to the action."""

        back_m = action & 1
        forward_m = (action >> 1) & 1
        left_m = (action >> 2) & 1
        right_m = (action >> 3) & 1
        sprint_m = (action >> 4) & 1

        action = self.noop_action.copy()

        action['sprint'] = sprint_m
        action['right'] = right_m
        action['left'] = left_m
        action['forward'] = forward_m
        action['back'] = back_m

        return action

    def get_action_meanings(self):
        # Override this method to reflect the modified action space
        return [str(i) for i in range(self.action_space.n)]

    def render(self, mode='human', **kwargs):
        # Override the render method if necessary
        return self.env.render(mode, **kwargs)

    def seed(self, seed=None):
        # Forward the seed call to the wrapped environment
        return self.env.seed(seed)


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.last_logged_episode = -1

    def _on_rollout_end(self):
        env = self.training_env.envs[0].env

        rewards = env.get_episode_rewards()
        lengths = env.get_episode_lengths()

        if len(rewards) > self.last_logged_episode + 1:
            mean_reward = sum(rewards[self.last_logged_episode+1:]) / len(rewards[self.last_logged_episode+1:])
            mean_length = sum(lengths[self.last_logged_episode+1:]) / len(lengths[self.last_logged_episode+1:])

            total_timesteps = env.get_total_steps()  # Retrieve total steps from the Monitor wrapper
            
            wandb.log({
                'mean_reward': mean_reward,
                'mean_episode_length': mean_length,
                'total_timesteps': total_timesteps
            })

            self.last_logged_episode = len(rewards) - 1

    def _on_step(self):
        return True

# Create the BitMaskWrapper around the MineRL environment
env = make_env()
print("Environment created!\n")

# Create your model (e.g., PPO)
model = PPO(config["policy_type"], env, verbose=0, device="cuda")
print("PPO model created!\n")

# Train your model with the callback
model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback())