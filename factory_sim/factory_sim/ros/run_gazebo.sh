#!/bin/bash
# Run FactorySim in Gazebo with multi-robot Nav2 inside the factorysim-ros:1.0 Docker image (docker build -t factorysim-ros:1.0 docker/).
#   factory_sim/ros/run_gazebo.sh [launch args...]      e.g.  robots:=2 orders:=2 proc_scale:=0.1
# Results: ros_runs/<timestamp>/result.json (metrics, RFID log, measured travel times); world + map in the same folder.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="ros_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/$RUN"
IMAGE="${FACTORYSIM_ROS_IMAGE:-factorysim-ros:1.0}"   # docker build -t factorysim-ros:1.0 docker/
PORTS=""
case " $* " in *" gui:=true "*) PORTS="-p 127.0.0.1:${FACTORYSIM_VNC_PORT:-6081}:6080"; echo "Gazebo window: http://localhost:${FACTORYSIM_VNC_PORT:-6081}/vnc.html";; esac
docker run --rm --name ${FACTORYSIM_NAME:-factorysim_gazebo} $PORTS -v "$REPO":/ws/factory_sim -e PYTHONPATH=/ws/factory_sim "$IMAGE" bash -lc "source /opt/ros/humble/setup.bash && source /opt/turtlebot3_ws/install/setup.bash 2>/dev/null; export DISPLAY=:1 TURTLEBOT3_MODEL=waffle ROS_DOMAIN_ID=${FACTORYSIM_DOMAIN:-42}; timeout ${FACTORYSIM_TIMEOUT:-3600} ros2 launch /ws/factory_sim/factory_sim/ros/launch/factory_gazebo.launch.py out:=/ws/factory_sim/$RUN $* 2>&1 | grep -E --line-buffered 'factory_sim|factory done|Nav2 ready|navigation failed|failed at|teleop|ERROR|Error|Traceback|process has died'"
echo "result: $REPO/$RUN/result.json"
