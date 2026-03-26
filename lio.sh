#!/bin/bash


source /opt/ros/humble/setup.bash
source install/local_setup.sh


# 实机 / 全用系统时间：默认 use_sim_time:=false
ros2 launch fast_lio multi_mid360.launch.py "$@"

# Gazebo 仿真：必须让 fast_lio 与 rviz 同源时钟（与 /clock 一致）
# ros2 launch fast_lio multi_sim_mid360.launch.py use_sim_time:=true "$@"