"""Collision checks run inside the evaluators: the Create 3 bumper and every native LiDAR scan.

Whichever fires first ends the episode as a collision; the detection records its "source".

bump:  any BUMP in <ns>/hazard_detection (irobot_create_msgs/HazardDetectionVector) while an episode runs. This is
       physical contact. The Create 3 bumper only covers the front half of the robot.

lidar: same rule as DRLEnvironment.get_state(): the closest of the 40 navigation beams (inf/nan ->
       REAL_LIDAR_DISTANCE_CAP, minus REAL_LIDAR_CORRECTION) at or below REAL_THRESHOLD_COLLISION is a collision,
       nothing is checked during the first GRACE_STEPS control steps, and a goal within REAL_THRESHOLD_GOAL wins over
       a collision. It runs on every raw scan, before the gate decides whether to forward it, so the result does not
       depend on the sensing rate under test. real_environment still checks the gated scans, but any forwarded scan
       at or below the threshold has already been caught here.
"""
import math

from rclpy.qos import qos_profile_sensor_data

from turtlebot3_drl.common.settings import (
    REAL_LIDAR_CORRECTION,
    REAL_LIDAR_DISTANCE_CAP,
    REAL_THRESHOLD_COLLISION,
    REAL_THRESHOLD_GOAL,
)
from turtlebot3_drl.drl_environment.drl_environment_real import GAZEBO_BEAM_ANGLES

GRACE_STEPS = 3          # DRLEnvironment.get_state(): no outcome while local_step <= 3
COLLISION_CHECK = "native_scan_gate"
DEFAULT_HAZARD_TOPIC = "/robot1/hazard_detection"


def min_beam_distance(msg, indices):
    """Closest of the navigation beams in metres, as DRLEnvironment.scan_callback(), and its beam number."""
    best, best_beam = math.inf, None
    for beam, idx in enumerate(indices):
        raw = float(msg.ranges[idx])
        if not math.isfinite(raw):
            raw = REAL_LIDAR_DISTANCE_CAP
        distance = min(max(raw - REAL_LIDAR_CORRECTION, 0.0), REAL_LIDAR_DISTANCE_CAP)
        if distance < best:
            best, best_beam = distance, beam
    return best, best_beam


def create_bump_subscription(node, topic, on_bump):
    """Subscribe to the Create 3 hazards and call on_bump(zones) for every message that holds a BUMP.

    zones are the frame_ids of the bump detections (bump_left, bump_front_center, ...). Returns None, with a
    warning, if irobot_create_msgs is not installed: the evaluator then runs with the LiDAR check only.
    """
    try:
        from irobot_create_msgs.msg import HazardDetection, HazardDetectionVector
    except ImportError as exc:
        node.get_logger().warning(
            f"Bump collision check disabled, irobot_create_msgs is not installed ({exc}). "
            "Install ros-humble-irobot-create-msgs to enable it."
        )
        return None

    def on_hazards(msg):
        zones = [d.header.frame_id for d in msg.detections if d.type == HazardDetection.BUMP]
        if zones:
            on_bump(zones)

    # The Create 3 publishes hazards best effort.
    subscription = node.create_subscription(HazardDetectionVector, topic, on_hazards, qos_profile_sensor_data)
    node.get_logger().info(f"Bump collision check on {topic}.")
    return subscription


class NativeCollisionMonitor:
    """Latches the first collision of an episode, from the bumper or a native scan."""

    def __init__(self, threshold=REAL_THRESHOLD_COLLISION, goal_threshold=REAL_THRESHOLD_GOAL):
        self.threshold = float(threshold)
        self.goal_threshold = float(goal_threshold)
        self.reset()

    def reset(self):
        """Disarm both checks and clear; call at the start and end of every episode."""
        self.armed = False        # LiDAR check, after the grace period
        self.bump_armed = False   # bump check, for the whole episode
        self.detection = None

    @property
    def detected(self):
        return self.detection is not None

    @staticmethod
    def _elapsed(now, episode_start):
        return float(now - episode_start) if episode_start is not None else None

    def check(self, msg, indices, now, pose, goal, episode_start, due_for_forwarding):
        """True the first time an armed episode sees a scan at or below the threshold (and the goal is not reached)."""
        if not self.armed or self.detection is not None or indices is None:
            return False
        distance, beam = min_beam_distance(msg, indices)
        if distance > self.threshold:
            return False
        if pose is not None and goal is not None:
            if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < self.goal_threshold:
                return False   # real_environment reports SUCCESS at the next step
        self.detection = {
            "source": "lidar",
            "t_s": self._elapsed(now, episode_start),
            "min_distance_m": float(distance),
            "beam": int(beam),
            "beam_angle_rad": float(GAZEBO_BEAM_ANGLES[beam]),
            "scan_due_for_forwarding": bool(due_for_forwarding),
            "stamp_ros_s": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        }
        return True

    def check_bump(self, zones, now, episode_start):
        """True the first time a running episode gets a bump."""
        if not self.bump_armed or self.detection is not None:
            return False
        self.detection = {
            "source": "bump",
            "t_s": self._elapsed(now, episode_start),
            "bump_zones": list(zones),
        }
        return True

    def describe(self):
        d = self.detection
        if d is None:
            return "no collision"
        t = f"t={d['t_s']:.2f} s" if d["t_s"] is not None else "t=?"
        if d["source"] == "bump":
            return f"Collision (bump): {', '.join(d['bump_zones'])}, {t}"
        return (
            f"Collision (native scan): min distance={d['min_distance_m']:.3f} m "
            f"<= {self.threshold:.3f} m, beam {d['beam']} "
            f"({math.degrees(d['beam_angle_rad']):.0f} deg), {t}, "
            f"scan due for forwarding={d['scan_due_for_forwarding']}"
        )

    @staticmethod
    def configuration(hazard_topic=None):
        """Recorded in every episode's metadata. hazard_topic is None when the bump check is off."""
        return {
            "collision_check": COLLISION_CHECK,
            "collision_grace_steps": GRACE_STEPS,
            "bump_collision_check": hazard_topic is not None,
            "hazard_topic": hazard_topic,
        }
