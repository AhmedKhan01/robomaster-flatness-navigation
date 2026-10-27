"""Log real execution data and explicitly labeled model-estimated voltages."""
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64MultiArray
from .core import wheel_rates


class Logger(Node):
    def __init__(self):
        super().__init__('rm_voltage_logger')
        self.declare_parameter('output_dir', str(Path.home()/'rm_flatness_logs'))
        root = Path(str(self.get_parameter('output_dir').value)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.directory = root/time.strftime('%Y%m%d_%H%M%S')
        self.directory.mkdir(exist_ok=False)
        self.fields = ['timestamp_s','V1_V','V2_V','V3_V','V4_V','state',
                       'v_m_s','omega_rad_s','camera_forward_m','camera_left_m',
                       'tof_m','T_s','voltage_source','driver_status',
                       'rpm1','rpm2','rpm3','rpm4','esc_age_s']
        self.files = [open(self.directory/name, 'w', newline='') for name in
                      ['motor_voltage_log.csv','execution_log.csv']]
        self.writers = [csv.DictWriter(f, fieldnames=self.fields) for f in self.files]
        for writer in self.writers:
            writer.writeheader()
        self.t0 = None
        self.previous = None
        self.next_second = 0
        self.state = 'WAITING'
        self.health = ''
        self.details = {}
        self.rpm, self.esc_stamp, self.esc_age = [math.nan]*4, -math.inf, math.inf
        self.create_subscription(String, '/rm/state', self.state_cb, 10)
        self.create_subscription(String, '/rm/driver_status', self.health_cb, 10)
        self.create_subscription(String, '/rm/details', self.detail_cb, 10)
        self.create_subscription(Float64MultiArray, '/rm/wheels', self.wheel_cb, 10)
        self.create_subscription(Twist, '/rm/applied_cmd', self.command_cb, 10)
        self.get_logger().info('Logs: '+str(self.directory))

    def state_cb(self, msg):
        self.state = msg.data

    def health_cb(self, msg):
        self.health = msg.data

    def detail_cb(self, msg):
        self.details = json.loads(msg.data)

    def wheel_cb(self, msg):
        if len(msg.data) == 5:
            self.rpm = list(msg.data[:4])
            self.esc_age = msg.data[4]
            self.esc_stamp = time.monotonic()

    def command_cb(self, msg):
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        elapsed = now-self.t0
        rates = wheel_rates(msg.linear.x, msg.angular.z)
        if self.previous is None:
            voltage = .38*rates  # no pre-start acceleration history is invented
        else:
            previous_time, previous_rates = self.previous
            dt = now-previous_time
            voltage = .38*rates+.06*(rates-previous_rates)/dt if dt > 0 else np.full(4, math.nan)
        self.previous = (now, rates)
        row = dict(zip(self.fields[:5], [elapsed, *map(float, voltage)]))
        row.update(state=self.state, v_m_s=msg.linear.x, omega_rad_s=msg.angular.z,
                   camera_forward_m=self.details.get('x_m'), camera_left_m=self.details.get('y_m'),
                   tof_m=self.details.get('tof_m'), T_s=self.details.get('T_s'),
                   voltage_source='Eq5_from_applied_command_wheel_rates', driver_status=self.health,
                   rpm1=self.rpm[0], rpm2=self.rpm[1], rpm3=self.rpm[2], rpm4=self.rpm[3],
                   esc_age_s=self.esc_age+now-self.esc_stamp)
        self.writers[1].writerow(row)
        if elapsed >= self.next_second:
            # Record the ACTUAL timestamp, rather than relabeling it as exact integer time.
            self.writers[0].writerow(row)
            self.next_second = math.floor(elapsed)+1
            self.get_logger().info(f'{elapsed:7.3f}s  '+
                ' '.join(f'{value:+.3f}V' for value in voltage)+f'  {self.state}')
            for f in self.files:
                f.flush()

    def close(self):
        for f in self.files:
            f.close()


def main(args=None):
    rclpy.init(args=args)
    node = Logger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
