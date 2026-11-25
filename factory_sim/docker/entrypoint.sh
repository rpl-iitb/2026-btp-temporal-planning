#!/bin/bash
# Start a virtual display :1 (VNC on 5901, noVNC on 6080, no password: publish the port only to localhost), then run the command.
Xvnc :1 -geometry 1600x900 -depth 24 -SecurityTypes None -localhost no >/tmp/xvnc.log 2>&1 &
websockify --web /usr/share/novnc 6080 localhost:5901 >/tmp/novnc.log 2>&1 &
export DISPLAY=:1
source /opt/ros/humble/setup.bash
exec "$@"
