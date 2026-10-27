# VISION_STREAM_V3
"""Visual target estimation and supervisory planner. Default: alignment only."""
import json
import math
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray, String
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from .core import Mission

PIXELS = np.array([[636.6779150500306, 533.9600775985297], [641.0408472012103, 489.15582450832073], [469.59956159579133, 396.0886777513856], [790.8719723183391, 394.18116417480724], [543.462915601023, 217.66751918158567], [722.3328846254378, 218.1965703397904]], dtype=float)
FLOOR = np.array([[0.2, 0.0], [0.25, 0.0], [0.4, 0.15], [0.4, -0.15], [1.0, 0.15], [1.0, -0.15]], dtype=float)


class Planner(Node):
    def __init__(self):
        super().__init__('rm_planner')
        cv2.setNumThreads(1)
        self.vision_diagnostic = 'WAITING_FOR_IMAGE'
        self.declare_parameter('mode', 'align')
        self.declare_parameter('target_id', 3)
        self.declare_parameter('front_to_rotation_center_m', 0.0)
        self.target_id = int(self.get_parameter('target_id').value)
        self.mission = Mission(str(self.get_parameter('mode').value),
            float(self.get_parameter('front_to_rotation_center_m').value))
        self.H, _ = cv2.findHomography(PIXELS, FLOOR, method=0)
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11))
        self.vision = None
        self.observation_id = 0
        self.range = (None, math.inf, -math.inf)
        self.esc = (math.inf, -math.inf)
        self.start = self.last_print = time.monotonic()
        self.old_state = None
        self.image_group = MutuallyExclusiveCallbackGroup()
        self.cmd_pub = self.create_publisher(Twist, '/rm/cmd_vel', 1)
        self.state_pub = self.create_publisher(String, '/rm/state', 10)
        self.detail_pub = self.create_publisher(String, '/rm/details', 10)
        self.create_subscription(CompressedImage, '/rm/image/compressed', self.image_cb,
                                 qos_profile_sensor_data.__class__(depth=1, reliability=qos_profile_sensor_data.reliability), callback_group=self.image_group)
        self.create_subscription(Float64MultiArray, '/rm/range', self.range_cb, 10)
        self.create_subscription(Float64MultiArray, '/rm/wheels', self.wheel_cb, 10)
        self.create_timer(.05, self.tick)
        self.get_logger().info('Mode: '+self.mission.mode+'; Ctrl+C stops planner commands.')

    def range_cb(self, msg):
        if len(msg.data) == 2:
            self.range = (msg.data[0], msg.data[1], time.monotonic())

    def wheel_cb(self, msg):
        if len(msg.data) == 5:
            self.esc = (msg.data[4], time.monotonic())

    def image_cb(self, msg):
        source_stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        latency = self.get_clock().now().nanoseconds*1e-9-source_stamp
        if latency < -.05 or latency > .30:
            self.vision_diagnostic = f'FRAME_AGE_REJECTED_{latency:.3f}s'
            self.vision = None
            return
        try:
            gray = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        except cv2.error:
            gray = None
        if gray is None or gray.shape != (720, 1280):
            self.vision_diagnostic = 'IMAGE_DECODE_OR_SIZE_ERROR'
            self.vision = None
            return
        corners, ids, _ = self.detector.detectMarkers(gray)
        # FAR_TAG_FALLBACK_V2: stretch only the distant floor region for detection.
        # All corners are mapped back to original pixels before applying calibration.
        if ids is None or not np.any(ids.flatten() == self.target_id):
            far = cv2.resize(gray[40:360], None, fx=1, fy=3,
                             interpolation=cv2.INTER_CUBIC)
            far_corners, far_ids, _ = self.detector.detectMarkers(far)
            if far_ids is not None and np.any(far_ids.flatten() == self.target_id):
                corners = [c.astype(np.float32) / np.array([1., 3.], dtype=np.float32)
                           + np.array([0., 40.], dtype=np.float32) for c in far_corners]
                ids = far_ids
        if ids is None:
            # Retain the original observation and timestamp; Mission expires it at 0.30 s.
            self.vision_diagnostic = 'NO_MARKERS_DETECTED'
            return
        matches = np.flatnonzero(ids.flatten() == self.target_id)
        if len(matches) == 0:
            self.vision_diagnostic = 'TARGET_ID_NOT_FOUND'
            return
        if len(matches) > 1:
            self.vision_diagnostic = 'AMBIGUOUS_TARGET_ID'
            self.vision = None
            return
        c = corners[int(matches[0])].reshape(4, 2)
        p = np.column_stack((c, np.ones(4)))
        center = np.cross(np.cross(p[0],p[2]), np.cross(p[1],p[3]))
        mapped = self.H @ center
        if abs(mapped[2]) < 1e-9:
            self.vision = None
            return
        x, y = mapped[:2]/mapped[2]
        if not np.all(np.isfinite([x,y])) or not .08 < x < 2.5 or abs(y) > 1.5:
            self.vision = None
            return
        # Check age after detection too; timestamp must reflect frame receipt.
        age = self.get_clock().now().nanoseconds*1e-9-source_stamp
        if age > .30:
            self.vision_diagnostic = f'PROCESSING_AGE_REJECTED_{age:.3f}s'
            self.vision = None
            return
        self.vision_diagnostic = f'TARGET_OK_frame_age_{age:.3f}s'
        self.observation_id += 1
        self.vision = (time.monotonic()-max(age,0), float(x), float(y))

    def tick(self):
        now = time.monotonic()
        vision = self.vision
        d, da, dr = self.range
        ea, er = self.esc
        v, w = self.mission.update(now, vision, d, da+now-dr, ea+now-er,
                                   self.observation_id)
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.cmd_pub.publish(msg)
        state = self.mission.state
        self.state_pub.publish(String(data=state))
        detail = {
            'state': state, 'reason': self.mission.reason,
            'x_m': None if vision is None else vision[1],
            'y_m': None if vision is None else vision[2],
            'bearing_rad': self.mission.last_bearing,
            'tof_m': d if d is not None and math.isfinite(d) else None,
            'T_s': self.mission.T, 'mode': self.mission.mode,
            'vision_diagnostic': self.vision_diagnostic,
            'vision_age_s': None if vision is None else round(now-vision[0], 3),
        }
        self.detail_pub.publish(String(data=json.dumps(detail)))
        if state != self.old_state or now-self.last_print >= 1:
            self.get_logger().info(f'{state}: v={v:.3f} m/s, w={w:.3f} rad/s; '+json.dumps(detail))
            self.old_state, self.last_print = state, now


def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
