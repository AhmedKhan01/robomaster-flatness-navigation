# CAMERA_RANGE_V4
# VISION_TIMING_V2
"""ROS-independent mathematics and supervisory controller. SI units throughout."""
import math
import numpy as np


def solve_flat_polynomial(z0, zdot0, zf, zdotf, T):
    if not math.isfinite(T) or T <= 0:
        raise ValueError('Horizon must be finite and positive')
    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                  [1, T, T*T, T**3], [0, 1, 2*T, 3*T*T]], dtype=float)
    return np.linalg.solve(A, [z0, zdot0, zf, zdotf])


def evaluate(a, t):
    a0, a1, a2, a3 = a
    return (a0+a1*t+a2*t*t+a3*t**3,
            a1+2*a2*t+3*a3*t*t, 2*a2+6*a3*t)


def flat_inverse(dx, dy, ddx, ddy):
    speed2 = dx*dx + dy*dy
    if speed2 < 1e-10:
        return 0.0, 0.0
    return math.sqrt(speed2), (dx*ddy-dy*ddx)/speed2


def wheel_rates(v, w):
    right, left = (v+0.20*w)/0.05, (v-0.20*w)/0.05
    return np.array([right, left, left, right])  # FR, FL, RL, RR


class Mission:
    def __init__(self, mode='align', center_offset=0.0):
        if mode not in ('observe', 'align', 'full'):
            raise ValueError('mode must be observe, align, or full')
        self.mode, self.center_offset = mode, center_offset
        self.state, self.reason = 'WAITING', ''
        self.last_bearing, self.last_seen = 0.30, None
        self.arrived, self.plan, self.aligned = False, None, 0
        self.fault_latched = False
        self.plan_start = self.T = self.d0 = None
        self.last_v = self.last_w = 0.0

    def output(self, state, v=0.0, w=0.0, reason=''):
        self.state, self.reason = state, reason
        self.last_v, self.last_w = v, w
        return v, w

    def update(self, now, vision, tof, tof_age, esc_age, observation_id=None):
        # vision: (arrival_time, forward_from_front, left_from_centerline).
        # Invalid/stale sensor values never count as zero distance.
        if self.fault_latched:
            return self.output('APPROACH_TIMEOUT', reason='Restart planner to begin a new attempt')
        valid_tof = tof is not None and 0.10 <= tof <= 10 and tof_age < 0.40
        age = math.inf if vision is None else now-vision[0]
        fresh = vision is not None and 0 <= age <= 0.30
        if fresh:
            self.last_bearing = math.atan2(vision[2], vision[1]+self.center_offset)
        if self.arrived:
            return self.output(self.state, reason=self.reason)
        if valid_tof and tof <= 0.20:
            self.arrived = True
            return self.output('ARRIVED_TOF', reason='ToF <= 0.20 m; verify target standoff physically')
        if fresh and 0 < math.hypot(vision[1], vision[2]) <= 0.20:
            self.arrived = True
            return self.output('ARRIVED_CAMERA', reason='Camera ground distance from front midpoint <= 0.20 m')
        if not valid_tof or esc_age > 0.50:
            self.plan, self.aligned = None, 0
            return self.output('SENSOR_HOLD', reason='Need fresh valid ToF and ESC telemetry')
        if self.mode == 'observe':
            return self.output('OBSERVING')
        if not fresh:
            self.plan, self.aligned = None, 0
            # Save last seen time even if a missed-frame marker follows it.
            lost_age = math.inf if self.last_seen is None else now-self.last_seen
            if lost_age > 0.45:
                sign = 1 if self.last_bearing >= 0 else -1
                return self.output('SEARCHING', w=sign*0.30)
            return self.output('VISION_HOLD', reason='No fresh target; forward command zero')

        stamp, x, y = vision
        self.last_seen = stamp
        bearing = math.atan2(y, x+self.center_offset)
        self.last_bearing = bearing
        if abs(bearing) > 0.18:
            self.plan, self.aligned = None, 0
            return self.output('ALIGNING', w=float(np.clip(1.3*bearing, -0.30, 0.30)))
        # Count distinct observations, not repeated control ticks.
        if observation_id != getattr(self, '_last_id', None):
            self.aligned += 1
            self._last_id = observation_id
        if self.aligned < 3:
            return self.output('ALIGNING', reason='Confirming centered target')
        if self.mode == 'align':
            return self.output('CENTERED', reason='Alignment-only mode: forward motion disabled')
        if self.plan is None:
            self.d0, self.plan_start = x, now
            self.T = max(4.0, 1.6*(x-0.20)/0.16)
            self.plan = (
                solve_flat_polynomial(0, 0.06, x-0.20, 0, self.T),
                solve_flat_polynomial(0, 0, 0, 0, self.T),
            )
        t = now-self.plan_start
        if t > self.T+10:
            self.fault_latched = True
            return self.output('APPROACH_TIMEOUT', reason='Horizon plus correction time exceeded; restart required')
        xr, dx, ddx = evaluate(self.plan[0], min(t, self.T))
        _, dy, ddy = evaluate(self.plan[1], min(t, self.T))
        vref, wref = flat_inverse(dx, dy, ddx, ddy)
        traveled = self.d0-x
        v = float(np.clip(vref+0.5*(xr-traveled), 0, 0.16))
        # Terminal correction above the observed low-speed stall region.
        if t >= self.T:
            v = 0.04
        # Slow approach limits override nominal cruise speed near the tag.
        v = min(v, max(0.04, 0.8*(x-0.20)))
        w = float(np.clip(wref+0.8*bearing, -0.30, 0.30))
        return self.output('CHASING', v, w)
