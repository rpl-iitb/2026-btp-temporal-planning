#!/bin/bash
# Keyboard teleop for a running FactorySim Gazebo run (started with run_gazebo.sh): select robots, drive them, give them tasks.
docker exec -it ${FACTORYSIM_NAME:-factorysim_gazebo} bash -lc "source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=${FACTORYSIM_DOMAIN:-42}; python3 -m factory_sim.ros.teleop"
