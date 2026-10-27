# VISION_STREAM_V3
"""SDK owner. Independent command timeout and ToF stop guard."""
import math
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray, String
from robomaster import robot


class Driver(Node):
    def __init__(self):
        super().__init__('rm_driver')
        cv2.setNumThreads(1)
        self.declare_parameter('sdk_yaw_sign', -1.0)
        self.declare_parameter('initialize_arm', True)
        self.declare_parameter('tof_index', 0)
        self.yaw_sign = float(self.get_parameter('sdk_yaw_sign').value)
        if self.yaw_sign not in (-1, 1):
            raise ValueError('sdk_yaw_sign must be -1 or +1')
        self.tof_index = int(self.get_parameter('tof_index').value)
        if not 0 <= self.tof_index <= 3:
            raise ValueError('tof_index must be 0..3')
        self.cmd = (0., 0., -math.inf)
        self.tof = (math.nan, -math.inf)
        self.esc = (None, -math.inf)
        self.stop_latched = False
        self.running = True
        self.frame = None
        self.frame_id = 0
        self.sent_frame = -1
        self.image_group = MutuallyExclusiveCallbackGroup()
        self.thread = None
        self.ep = robot.Robot()
        self.started = self.dist_sub = self.esc_sub = False
        image_qos = qos_profile_sensor_data.__class__(depth=1, reliability=qos_profile_sensor_data.reliability)
        self.image_pub = self.create_publisher(CompressedImage, '/rm/image/compressed', image_qos)
        self.range_pub = self.create_publisher(Float64MultiArray, '/rm/range', 10)
        self.wheel_pub = self.create_publisher(Float64MultiArray, '/rm/wheels', 10)
        self.applied_pub = self.create_publisher(Twist, '/rm/applied_cmd', 10)
        self.health_pub = self.create_publisher(String, '/rm/driver_status', 10)
        self.create_subscription(Twist, '/rm/cmd_vel', self.on_cmd, 1)
        try:
            self.ep.initialize(conn_type='ap')
            self.ep.chassis.drive_speed(x=0, y=0, z=0, timeout=0.3)
            if self.get_parameter('initialize_arm').value:
                self.get_logger().info('Moving arm to (105,30) mm; mounting MUST be clear.')
                if not self.ep.robotic_arm.moveto(x=105, y=30).wait_for_completed(timeout=15):
                    raise RuntimeError('Arm startup action failed')
                time.sleep(1)
            self.dist_sub = self.ep.sensor.sub_distance(freq=20, callback=self.on_tof)
            self.esc_sub = self.ep.chassis.sub_esc(freq=20, callback=self.on_esc)
            if not self.dist_sub or not self.esc_sub:
                raise RuntimeError('ToF or ESC subscription rejected')
            self.started = self.ep.camera.start_video_stream(display=False, resolution='720p')
            if not self.started:
                raise RuntimeError('Camera startup failed')
            self.thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.thread.start()
            self.create_timer(0.05, self.tick)
            self.create_timer(0.10, self.publish_image, callback_group=self.image_group)
            self.get_logger().info('Driver ready; no motion until planner commands arrive.')
        except BaseException:
            self.close()
            raise

    def on_cmd(self, msg):
        self.cmd = (float(msg.linear.x), float(msg.angular.z), time.monotonic())

    def on_tof(self, data):
        d = float(data[self.tof_index])/1000.
        self.tof = (d, time.monotonic())
        if math.isfinite(d) and 0.10 <= d <= 0.20:
            self.stop_latched = True

    def on_esc(self, data):
        self.esc = (list(data[0]), time.monotonic())

    def camera_loop(self):
        while self.running:
            try:
                frame = self.ep.camera.read_cv2_image(strategy='newest', timeout=1)
                if frame is not None:
                    self.frame_id += 1
                    self.frame = (frame, self.get_clock().now().to_msg(), self.frame_id)
            except queue.Empty:
                pass
            except Exception as exc:
                self.get_logger().error('Camera read failed: '+str(exc))
                break
            time.sleep(0.010)

    def tick(self):
        now = time.monotonic()
        d, td = self.tof
        rpm, te = self.esc
        v, w, tc = self.cmd
        status = 'READY'
        if self.stop_latched:
            v = w = 0.
            status = 'TOF_STOP_LATCHED'
        elif not math.isfinite(d) or not 0.10 <= d <= 10 or now-td > .40:
            v = w = 0.
            status = 'TOF_INVALID_OR_STALE'
        elif now-te > .50:
            v = w = 0.
            status = 'ESC_STALE'
        elif now-tc > .25 or not math.isfinite(v+w):
            v = w = 0.
            status = 'COMMAND_TIMEOUT'
        v, w = float(np.clip(v, 0, .16)), float(np.clip(w, -.30, .30))
        self.ep.chassis.drive_speed(x=v, y=0, z=self.yaw_sign*math.degrees(w), timeout=.30)
        applied = Twist()
        applied.linear.x, applied.angular.z = v, w
        self.applied_pub.publish(applied)
        self.health_pub.publish(String(data=status))
        self.range_pub.publish(Float64MultiArray(data=[d, now-td]))
        if rpm is not None:
            self.wheel_pub.publish(Float64MultiArray(data=[*map(float, rpm), now-te]))

    def publish_image(self):
        item = self.frame
        if item is None or item[2] == self.sent_frame:
            return
        frame, stamp, seq = item
        # Compress full-resolution grayscale without changing any detection pixels.
        age = self.get_clock().now().nanoseconds*1e-9 - (stamp.sec+stamp.nanosec*1e-9)
        if age > .30:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, encoded = cv2.imencode('.png', gray, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp, msg.header.frame_id = stamp, 'camera'
        msg.format = 'mono8; png compressed'
        msg.data = encoded.tobytes()
        self.image_pub.publish(msg)
        self.sent_frame = seq

    def close(self):
        self.running = False
        try:
            self.ep.chassis.drive_speed(x=0, y=0, z=0, timeout=.3)
        except Exception:
            pass
        for enabled, fn in [(self.dist_sub, self.ep.sensor.unsub_distance),
                            (self.esc_sub, self.ep.chassis.unsub_esc),
                            (self.started, self.ep.camera.stop_video_stream)]:
            if enabled:
                try:
                    fn()
                except Exception:
                    pass
        if self.thread:
            self.thread.join(timeout=2)
        try:
            self.ep.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = Driver()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor:
            executor.shutdown()
        if node:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
