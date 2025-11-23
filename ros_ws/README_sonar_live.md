# SOLAQUA ROS1 — Live Sonar Net Detection

This folder contains the ROS1 setup for streaming sonar frames from `rosbag` into the SOLAQUA net tracker and publishing per-frame detections (no NPZ needed).

## Prereqs (inside the container)
```bash
cd /home/shared_folder/ros_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Run (multi-shell)
- Shell A (master):
  ```bash
  roscore
  ```
- Shell B (play sonar bag; use the video bag that carries SonoptixECHO):
  ```bash
  rosbag play --clock --loop /home/shared_folder/ros_ws/bags/2024-08-20_13-39-34_video.bag
  # add -r 100 for faster playback
  ```
- Shell C (live detector):
  ```bash
  rosrun solaqua_tools sonar_live_net_node.py \
    _topic:=/sensor/sonoptix_echo/image \
    _range_max_m:=20.0 \
    _log_every:=5 \
    _log_csv_path:=/home/shared_folder/exports/net_detection_live.csv
  ```
- Shell D (inspect; CSV auto-saves to `_log_csv_path`):
  ```bash
  rostopic hz /solaqua/net_detection
  rostopic echo -n5 /solaqua/net_detection
  # optional: also record to bag:
  rosbag record -O /home/shared_folder/exports/net_detection_live.bag /solaqua/net_detection
  ```
  - `rostopic hz` confirms messages are flowing (frequency should match frame rate).
  - `rostopic echo -n5` prints a few messages so you can sanity-check values ([distance_px, angle_deg, distance_m]).
  - `_log_csv_path` auto-saves detections (frame index, timestamp, px/m distances, angle, success flag) to CSV. Set empty to disable.
  - `rosbag record ...` stores detections as a ROS bag for later replay or merging with other topics.

## Topics
- Input: `/sensor/sonoptix_echo/image` (sensors/SonoptixECHO with Float32MultiArray payload)
- Output: `/solaqua/net_detection` (std_msgs/Float32MultiArray: [distance_px, angle_deg, distance_m], -1 for missing)

## Notes
- The `sensors` package defines `SonoptixECHO.msg` so the bag can be decoded.
- The detector uses the existing SOLAQUA NetTracker pipeline per frame (binary + enhancement + tracking).
