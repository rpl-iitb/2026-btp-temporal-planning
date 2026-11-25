# FactorySim Gazebo twin: ROS 2 Humble, Gazebo 11, Nav2, TurtleBot3 (+ OpenMANIPULATOR-X meshes), a VNC desktop for the
# Gazebo window (noVNC on port 6080) and Xvfb-style rendering for camera recording.
#   docker build -t factorysim-ros:1.0 docker/
FROM osrf/ros:humble-desktop
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-gazebo-ros-pkgs ros-humble-navigation2 ros-humble-nav2-bringup \
      ros-humble-turtlebot3 ros-humble-turtlebot3-msgs ros-humble-turtlebot3-simulations \
      ros-humble-turtlebot3-manipulation-description \
      tigervnc-standalone-server novnc websockify xfce4-terminal fonts-dejavu-core \
      python3-numpy python3-yaml python3-pil \
    && rm -rf /var/lib/apt/lists/*
ENV TURTLEBOT3_MODEL=waffle ROS_DOMAIN_ID=42 GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && useradd -m ros
USER ros
WORKDIR /home/ros
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
