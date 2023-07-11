# Project Name


MineRL Reinforcement Learning with REINFORCE


## Description


This project uses the MineRL library for reinforcement learning in the Minecraft environment. It includes code to train an agent in the MineRLObtainDiamondShovel-v0 environment. The agent learns using the REINFORCE algorithm, a policy gradient method, to accomplish tasks and interact with the Minecraft world.


## Dependencies


- Python 3.9.16
- Gym
- Minerl
- Torch
- Matplotlib
- NumPy


## Installation (MAC)


1. Install Java Development Kit (JDK) 8:

   `brew tap AdoptOpenJDK/openjdk`
   
   `brew install --cask adoptopenjdk8`


2. Install Python dependencies:

   `pip install -r requirements.txt`


3. Install the MineRL Library:

   `pip install git+https://github.com/minerllabs/minerl`


4. Run the mineRL Python script:

   `python mineRL.py`

The script will run a specified number of episodes in the MineRLObtainDiamondShovel-v0 environment using the CNN model.

During the execution, the script will display the actions taken and render the environment.

After the episodes, it will generate plots of the cumulative rewards, losses, and episode lengths.


## Contributing

Contributions to this project are welcome. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and test them.
4. Submit a pull request.


## License

This project is licensed under the MIT License.
