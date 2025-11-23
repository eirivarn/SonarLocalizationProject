#!/usr/bin/env python3
"""Online sonar net detection node.

Subscribes to a Float32MultiArray sonar topic, runs the existing NetTracker
pipeline per frame, and publishes per-frame detections to /solaqua/net_detection.

Published message: std_msgs/Float32MultiArray with layout:
  dim[0].label = "distance_px"
  dim[1].label = "angle_deg"
  dim[2].label = "distance_m"
Data = [distance_px or -1, angle_deg or -1, distance_m or -1]
"""
from __future__ import annotations

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

from pathlib import Path
import sys

try:
    from sensors.msg import SonoptixECHO  # type: ignore
except Exception:  # pragma: no cover
    SonoptixECHO = None

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG, RANGE_MAX_M_DEFAULT
from utils.image_enhancement import preprocess_edges
from utils.sonar_tracking import NetTracker
from utils.sonar_utils import to_uint8_gray


class SonarLiveNetNode:
    def __init__(self):
        topic = rospy.get_param("~topic", "/sensor/sonoptix_echo/image")
        self.range_max_m = float(rospy.get_param("~range_max_m", RANGE_MAX_M_DEFAULT))
        self.binary_threshold = int(rospy.get_param("~binary_threshold", IMAGE_PROCESSING_CONFIG.get("binary_threshold", 128)))
        log_every = int(rospy.get_param("~log_every", 30))
        self.log_every = max(1, log_every)

        config = {**IMAGE_PROCESSING_CONFIG, **TRACKING_CONFIG}
        config["binary_threshold"] = self.binary_threshold
        self.tracker = NetTracker(config)

        self.pub = rospy.Publisher("solaqua/net_detection", Float32MultiArray, queue_size=10)
        msg_type = SonoptixECHO if SonoptixECHO is not None else Float32MultiArray
        self.sub = rospy.Subscriber(topic, msg_type, self._cb, queue_size=10)

        self.frame_count = 0
        rospy.loginfo(
            f"[sonar_live_net_node] Listening on {topic}, publishing to /solaqua/net_detection "
            f"(range_max_m={self.range_max_m}, binary_threshold={self.binary_threshold})"
        )

    def _publish_result(self, distance_px, angle_deg, distance_m):
        msg = Float32MultiArray()
        msg.layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label="distance_px", size=1, stride=3),
                MultiArrayDimension(label="angle_deg", size=1, stride=3),
                MultiArrayDimension(label="distance_m", size=1, stride=3),
            ],
            data_offset=0,
        )
        msg.data = [
            float(distance_px) if distance_px is not None else -1.0,
            float(angle_deg) if angle_deg is not None else -1.0,
            float(distance_m) if distance_m is not None else -1.0,
        ]
        self.pub.publish(msg)

    def _extract_frame(self, msg):
        """Handle Float32MultiArray or sensors/SonoptixECHO."""
        # SonoptixECHO wraps a Float32MultiArray in array_data
        if SonoptixECHO is not None and isinstance(msg, SonoptixECHO):
            if hasattr(msg, "array_data"):
                ma = msg.array_data
            elif hasattr(msg, "data"):
                ma = msg.data  # fallback
            else:
                return None, None
        else:
            ma = msg

        dims = getattr(ma, "layout", None).dim if hasattr(ma, "layout") else []
        data = getattr(ma, "data", None)

        if len(dims) < 2:
            return None, None
        h, w = dims[0].size, dims[1].size
        arr = np.array(data, dtype=np.float32)
        if arr.size < h * w:
            return None, None
        return arr[: h * w].reshape((h, w)), (h, w)

    def _cb(self, msg):
        self.frame_count += 1
        frame, shape = self._extract_frame(msg)
        if frame is None or shape is None:
            if self.frame_count % self.log_every == 0:
                try:
                    debug_dims = getattr(msg, "array_data", msg)
                    dims = getattr(getattr(debug_dims, "layout", None), "dim", [])
                    size_debug = [d.size for d in dims] if dims else []
                    rospy.logwarn(f"[sonar_live_net_node] Unable to parse frame; dims={size_debug}")
                except Exception:
                    rospy.logwarn("[sonar_live_net_node] Unable to parse frame; skipping.")
            return
        h, w = shape

        frame_u8 = to_uint8_gray(frame)
        binary = (frame_u8 > self.binary_threshold).astype(np.uint8) * 255
        try:
            _, edges = preprocess_edges(binary, self.tracker.config)
        except Exception:
            edges = binary

        best_contour = self.tracker.find_and_update(edges, frame_u8.shape)
        distance_px, angle_deg = self.tracker.calculate_distance(w, h)
        distance_m = None
        if distance_px is not None and self.range_max_m:
            distance_m = (distance_px / float(h)) * self.range_max_m

        if self.frame_count % self.log_every == 0:
            status = self.tracker.get_status()
            rospy.loginfo(
                f"[sonar_live_net_node] frame={self.frame_count} status={status} "
                f"dist_px={distance_px} angle_deg={angle_deg} dist_m={distance_m}"
            )

        self._publish_result(distance_px, angle_deg, distance_m)


def main():
    rospy.init_node("sonar_live_net_node")
    SonarLiveNetNode()
    rospy.spin()


if __name__ == "__main__":
    main()
