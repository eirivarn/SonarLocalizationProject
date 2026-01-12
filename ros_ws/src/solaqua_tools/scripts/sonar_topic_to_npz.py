#!/usr/bin/env python3
"""Collect sonar frames from a ROS topic and write a cones NPZ file.

Usage (in two shells):
  # shell 1: play bag natively
  rosbag play /home/shared_folder/raw_data/2024-08-22_14-06-43_data.bag --clock 100 --loop

  # shell 2: collect frames to NPZ
  rosrun solaqua_tools sonar_topic_to_npz.py _topic:=/sensor/sonoptix_echo/image \
    _max_frames:=1500 _run_id:=2024-08-22_14-06-43 _exports_dir:=/home/shared_folder/exports
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROS_WS_ROOT = Path(__file__).resolve().parents[4]


def _resolve_output_path(run_id: str, exports_dir: str) -> Path:
    exports = Path(exports_dir or ROS_WS_ROOT / "exports")
    exports.mkdir(parents=True, exist_ok=True)
    outputs_dir = exports / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir / f"{run_id}_cones.npz"


class SonarTopicCollector:
    def __init__(self, topic: str, max_frames: int, run_id: str, exports_dir: str):
        self.topic = topic
        self.max_frames = max_frames
        self.run_id = run_id
        self.exports_dir = exports_dir
        self.frames: List[np.ndarray] = []
        self.shape: Tuple[int, int] | None = None
        self.output_path = _resolve_output_path(run_id, exports_dir)
        self.sub = rospy.Subscriber(topic, Float32MultiArray, self._cb, queue_size=10)
        rospy.loginfo(f"[sonar_topic_to_npz] Collecting from {topic}, max_frames={max_frames}, run_id={run_id}")

    def _cb(self, msg: Float32MultiArray):
        dims = msg.layout.dim
        if len(dims) >= 2:
            h = dims[0].size
            w = dims[1].size
            self.shape = (h, w)
        arr = np.array(msg.data, dtype=np.float32)
        if self.shape and arr.size >= self.shape[0] * self.shape[1]:
            arr = arr[: self.shape[0] * self.shape[1]].reshape(self.shape)
        self.frames.append(arr)
        if len(self.frames) >= self.max_frames:
            rospy.loginfo("Reached max_frames; shutting down.")
            rospy.signal_shutdown("max_frames reached")

    def write_npz(self):
        if not self.frames:
            rospy.logwarn("No frames collected; skipping write.")
            return
        np.savez_compressed(self.output_path, cones=np.stack(self.frames, axis=0))
        rospy.loginfo(f"Wrote NPZ: {self.output_path} (frames={len(self.frames)})")


def main():
    rospy.init_node("sonar_topic_to_npz")
    topic = rospy.get_param("~topic", "/sensor/sonoptix_echo/image")
    max_frames = int(rospy.get_param("~max_frames", 1500))
    run_id = rospy.get_param("~run_id", f"run_{int(time.time())}")
    exports_dir = rospy.get_param("~exports_dir", str(ROS_WS_ROOT / "exports"))

    collector = SonarTopicCollector(topic, max_frames, run_id, exports_dir)
    try:
        rospy.spin()
    finally:
        collector.write_npz()


if __name__ == "__main__":
    main()
