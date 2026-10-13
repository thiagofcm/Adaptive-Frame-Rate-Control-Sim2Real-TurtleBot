#!/usr/bin/env python3

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class RealScanGate(Node):
    """
    Time-based LiDAR gate for the real TurtleBot.

    Input:
        /robot1/scan

    Output:
        /robot1/scan_gated

    The gate does NOT republish stale scans.
    It only forwards a newly-arrived physical LiDAR scan when the requested
    sensing time has been reached.

    Between forwarded scans, the downstream navigation environment should
    continue using its previously cached LiDAR observation (sample-and-hold).
    """

    def __init__(self, target_fps: float):
        super().__init__("real_scan_gate")

        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")

        self.target_fps = float(target_fps)
        self.period = 1.0 / self.target_fps

        # Gate state
        self.next_sample_time = None
        self.force_next_fresh = True

        # Diagnostics
        self.total_scans_received = 0
        self.total_scans_forwarded = 0
        self.last_forward_wall_time = None

        # ROS interfaces
        self.gated_pub = self.create_publisher(
            LaserScan,
            "/robot1/scan_gated",
            QoSProfile(depth=10),
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/robot1/scan",
            self._on_scan,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"RealScanGate started: "
            f"target_fps={self.target_fps:.3f} Hz, "
            f"target_period={self.period:.3f} s"
        )

    def _forward_scan(self, msg, now):
        """Forward one genuinely fresh physical LiDAR scan."""
        self.gated_pub.publish(msg)

        self.total_scans_forwarded += 1

        if self.last_forward_wall_time is not None:
            dt = now - self.last_forward_wall_time
            effective_fps = 1.0 / dt if dt > 0 else float("inf")
        else:
            dt = None
            effective_fps = None

        self.last_forward_wall_time = now

        if effective_fps is None:
            self.get_logger().info(
                f"Forwarded scan #{self.total_scans_forwarded}"
            )
        else:
            self.get_logger().info(
                f"Forwarded scan #{self.total_scans_forwarded} "
                f"(dt={dt:.3f}s, instantaneous={effective_fps:.2f} Hz)"
            )

    def _on_scan(self, msg):
        """
        Called whenever a NEW physical LiDAR scan arrives.

        We compare the current wall-clock time against the ideal target
        sensing schedule.

        Important:
        next_sample_time follows the ideal sampling phase. It is NOT reset to
        `now + period` after every forward, because doing that would accumulate
        timing delay.
        """
        now = time.monotonic()
        self.total_scans_received += 1

        # Episode/startup case:
        # always let the first available physical scan through.
        if self.force_next_fresh:
            self._forward_scan(msg, now)

            self.force_next_fresh = False
            self.next_sample_time = now + self.period
            return

        # Not yet time for another sensing update.
        if now < self.next_sample_time:
            return

        # Requested sampling instant has been reached.
        # Forward THIS newly-arrived physical scan.
        self._forward_scan(msg, now)

        # Advance the IDEAL sampling schedule.
        #
        # Example for 5 Hz:
        # target times are:
        #   t0
        #   t0 + 0.2
        #   t0 + 0.4
        #   t0 + 0.6
        #   ...
        #
        # If the physical sensor gives us a frame slightly late, we do not
        # shift the entire future schedule later.
        while self.next_sample_time <= now:
            self.next_sample_time += self.period

    def set_target_fps(self, target_fps: float):
        """
        Change the requested sensing rate dynamically.

        Useful later for the adaptive PPO policy.
        """
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")

        now = time.monotonic()

        self.target_fps = float(target_fps)
        self.period = 1.0 / self.target_fps

        # Start the NEW requested interval from the current instant.
        #
        # This means changing from, e.g., 1 Hz -> 5 Hz causes the next
        # requested observation to be 0.2 s from now.
        self.next_sample_time = now + self.period

        self.get_logger().info(
            f"Target sensing rate changed to "
            f"{self.target_fps:.3f} Hz "
            f"(period={self.period:.3f} s)"
        )

    def force_fresh_scan(self):
        """
        Force the next newly-arrived physical LiDAR scan to be forwarded.

        Useful at the start of each real-world episode.
        """
        self.force_next_fresh = True

    def print_statistics(self):
        skipped = self.total_scans_received - self.total_scans_forwarded

        if self.total_scans_received > 0:
            skipped_percent = (
                100.0 * skipped / self.total_scans_received
            )
        else:
            skipped_percent = 0.0

        self.get_logger().info(
            "Gate statistics: "
            f"received={self.total_scans_received}, "
            f"forwarded={self.total_scans_forwarded}, "
            f"skipped={skipped}, "
            f"skipped_percent={skipped_percent:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Time-based LiDAR sensing gate for the real TurtleBot."
    )
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Requested sensing rate in Hz, e.g. 5, 1, 0.5, 0.2",
    )

    # ROS2 adds its own CLI arguments, so only parse our known arguments.
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = RealScanGate(args.fps)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.print_statistics()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()