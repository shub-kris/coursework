# Self Driving Cars-Projects


This repository contains three different approaches for building an end to end autonomous self-driving agent in OpenAI Gym Environment. This was a part of the course **Self Driving Cars (Winter 2019)**.

Solutions have been implemented in PyTorch.


### Imitation Learning: 
In this project, the simplest form of imitation learning : behaviour cloning (BC), which focuses on learning the expert’s
policy using supervised learning has been implemented. First, we record the imitations(snapshots) of driving a car in the OpenAI simulator. These snapshots(images) along with the action taken (accelerate, turn left, turn right, brakes) serves as the training data.
Using this supervised data, a Deep Neural Network is trained and is an autonomous self driving agent.


### Reinforcement Learning:

In this project, the autonomous self driving car agent is trained using Deep Q Leanring and Double Deep Q Learning to learn the optimal driving policy for maximizing the reward.  


### Modular Pipeline:

In this project, I got an opportunity to implement a modular approach for building a self driving car agent. The task comprised of implementing modules for Lane Detection, Path Planning, Lateral and Longitudinal Control.






