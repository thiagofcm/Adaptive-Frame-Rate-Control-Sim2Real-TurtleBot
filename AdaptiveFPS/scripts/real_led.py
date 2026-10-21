#!/usr/bin/env python3
"""Show the active sensing rate on the TurtleBot4's USER_LED_2.

Pure visualization, shared by the fixed and adaptive real-robot evaluators.
It publishes on the evaluator's existing node (no extra node, thread or
timer) and never affects gating, sensing decisions, navigation or recording.
The LED shows the logical selected sensing rate (the evaluator's
current_fps), not the physically delivered LiDAR rate. The TurtleBot4 does
the blinking itself from blink_period.

Any LED problem (message package missing, publish failure) is logged and
never stops an evaluation.
"""

from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy


# Compressed, monotonic mapping: higher sensing FPS -> shorter blink period.
SENSING_FPS_TO_LED_PERIOD_MS = {
    0.2: 2000,
    0.5: 1333,
    1.0: 1000,
    5.0: 667,
    10.0: 400,
}
LED_DUTY_CYCLE = 0.5
LED_OFF_BLINK_PERIOD_MS = 1000
DEFAULT_LED_TOPIC = "/robot1/hmi/led"

# Matches the QoS of the `ros2 topic pub` command verified on the robot.
LED_QOS = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

_OFF = "off"


class SensingLedController:
    """Blink USER_LED_2 red at a period that encodes the sensing FPS."""

    def __init__(self, node, topic=DEFAULT_LED_TOPIC, enabled=True):
        self._node = node
        self._publisher = None
        self._msg_type = None
        # Last state actually published: None (unknown, e.g. at startup),
        # _OFF, or the sensing FPS being shown.
        self._state = None

        if not enabled:
            node.get_logger().info("Sensing-rate LED disabled (--no-led).")
            return
        try:
            from turtlebot4_msgs.msg import UserLed
            self._publisher = node.create_publisher(UserLed, topic, LED_QOS)
            self._msg_type = UserLed
            node.get_logger().info(f"Sensing-rate LED on {topic} (USER_LED_2, red).")
        except Exception as exc:
            self._publisher = None
            self._warn(f"Sensing-rate LED disabled, could not set up {topic}: {exc}")

    @property
    def enabled(self):
        return self._publisher is not None

    def set_blink_rate(self, sensing_fps):
        """Blink at the period mapped to sensing_fps. Publishes only on a change."""
        if self._publisher is None:
            return False
        fps = float(sensing_fps)
        if fps == self._state:
            return False
        period = SENSING_FPS_TO_LED_PERIOD_MS.get(fps)
        if period is None:
            self._warn(
                f"No LED blink period for sensing FPS {fps:g} "
                f"(supported: {sorted(SENSING_FPS_TO_LED_PERIOD_MS)}); skipping this LED update."
            )
            return False
        msg = self._msg_type()
        msg.led = self._msg_type.USER_LED_2
        msg.color = self._msg_type.COLOR_RED
        msg.blink_period = period
        msg.duty_cycle = LED_DUTY_CYCLE
        if self._publish(msg):
            self._state = fps
            return True
        return False

    def turn_off(self):
        """Turn USER_LED_2 off. Idempotent: publishes only if it may be on.

        The first call (state unknown, e.g. at evaluator startup) always
        publishes, clearing any LED state left by an earlier run.
        """
        if self._publisher is None or self._state == _OFF:
            return False
        msg = self._msg_type()
        msg.led = self._msg_type.USER_LED_2
        msg.color = self._msg_type.COLOR_OFF
        msg.blink_period = LED_OFF_BLINK_PERIOD_MS
        msg.duty_cycle = 0.0
        if self._publish(msg):
            self._state = _OFF
            return True
        return False

    def _publish(self, msg):
        try:
            self._publisher.publish(msg)
            return True
        except Exception as exc:
            self._publisher = None
            self._warn(f"LED publish failed; sensing-rate LED disabled for this run: {exc}")
            return False

    def _warn(self, text):
        try:
            self._node.get_logger().warning(text)
        except Exception:
            print(f"WARNING: {text}")
