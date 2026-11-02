"""TD3 navigation state built inside the evaluator at the control tick, as the simplified env does.

The 44 values use drl_environment_real.get_state()'s formulas: 40 gated LiDAR rays from scan_callback(), goal
distance and angle from odom_callback(), then the previous action. real_environment still publishes the command and
runs the goal and collision checks.
"""
import math

import numpy

from turtlebot3_drl.common.settings import REAL_LIDAR_CORRECTION, REAL_LIDAR_DISTANCE_CAP
from turtlebot3_drl.drl_environment.drl_environment_real import MAX_GOAL_DISTANCE, real_scan_beam_indices


def scan_rays(msg):
    """Normalized 40-ray scan, as DRLEnvironment.scan_callback(); None if the scan cannot be subsampled."""
    indices = real_scan_beam_indices(msg)
    if indices is None:
        return None
    rays = []
    for idx in indices:
        raw = float(msg.ranges[idx])
        if numpy.isinf(raw) or numpy.isnan(raw):
            raw = REAL_LIDAR_DISTANCE_CAP
        rays.append(float(numpy.clip((raw - REAL_LIDAR_CORRECTION) / REAL_LIDAR_DISTANCE_CAP, 0, 1)))
    return rays


def nav_state(rays, pose, goal, previous_action):
    """44-D TD3 state: rays, goal distance / MAX_GOAL_DISTANCE, goal angle / pi, previous action."""
    if rays is None or pose is None or goal is None:
        raise RuntimeError("TD3 state needs a gated scan, odometry and a goal")
    x, y, heading = pose
    diff_x, diff_y = goal[0] - x, goal[1] - y
    goal_angle = math.atan2(diff_y, diff_x) - heading
    while goal_angle > math.pi:
        goal_angle -= 2 * math.pi
    while goal_angle < -math.pi:
        goal_angle += 2 * math.pi
    return list(rays) + [
        float(numpy.clip(math.sqrt(diff_x**2 + diff_y**2) / MAX_GOAL_DISTANCE, 0, 1)),
        float(goal_angle) / math.pi,
        float(previous_action[0]),
        float(previous_action[1]),
    ]
