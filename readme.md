# Introduction
This repo is a ROS wrapper for GelSight Mini tactile sensor. It is modified from [gsrobotics](https://github.com/gelsightinc/gsrobotics).

Usage:
First put this repo in the src folder of your catkin workspace, and run catkin_make to build the essential messages and services.
Then run the following commands to start the node:
```
cd scripts
python3 gs_node.py
```

Note that the serial numbers of the two GelSight Mini sensors are hard-coded in the script. These two sensors are used for the ManiSkill-ViTac Challenge.