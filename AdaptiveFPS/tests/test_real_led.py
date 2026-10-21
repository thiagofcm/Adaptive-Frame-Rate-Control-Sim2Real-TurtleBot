import os
import sys
import time
import unittest
from unittest import mock

from rclpy.qos import QoSDurabilityPolicy, QoSReliabilityPolicy
from turtlebot4_msgs.msg import UserLed

from real_led import (
    LED_QOS,
    SENSING_FPS_TO_LED_PERIOD_MS,
    SensingLedController,
)

FPS_CHOICES = [0.2, 0.5, 1.0, 5.0, 10.0]


class FakeLogger:
    def __init__(self):
        self.warnings, self.infos = [], []

    def warning(self, text):
        self.warnings.append(text)

    def info(self, text):
        self.infos.append(text)


class FakePublisher:
    def __init__(self, fail=False):
        self.messages, self.fail = [], fail

    def publish(self, msg):
        if self.fail:
            raise RuntimeError("context is invalid")
        self.messages.append(msg)


class FakeNode:
    def __init__(self, fail=False):
        self.logger = FakeLogger()
        self.publisher = FakePublisher(fail)
        self.created = []

    def create_publisher(self, msg_type, topic, qos):
        self.created.append((msg_type, topic, qos))
        return self.publisher

    def get_logger(self):
        return self.logger


class MappingTest(unittest.TestCase):
    def test_covers_the_five_sensing_rates(self):
        self.assertEqual(sorted(SENSING_FPS_TO_LED_PERIOD_MS), FPS_CHOICES)

    def test_higher_fps_means_shorter_period(self):
        periods = [SENSING_FPS_TO_LED_PERIOD_MS[f] for f in FPS_CHOICES]
        self.assertEqual(periods, [2000, 1333, 1000, 667, 400])
        self.assertTrue(all(a > b for a, b in zip(periods, periods[1:])))


class ControllerTest(unittest.TestCase):
    def test_publisher_qos_and_topic(self):
        node = FakeNode()
        SensingLedController(node, "/robot1/hmi/led")
        msg_type, topic, qos = node.created[0]
        self.assertIs(msg_type, UserLed)
        self.assertEqual(topic, "/robot1/hmi/led")
        self.assertEqual(qos.reliability, QoSReliabilityPolicy.RELIABLE)
        self.assertEqual(qos.durability, QoSDurabilityPolicy.VOLATILE)

    def test_each_fixed_rate_publishes_once(self):
        for fps in FPS_CHOICES:
            node = FakeNode()
            led = SensingLedController(node)
            self.assertTrue(led.set_blink_rate(fps))
            self.assertFalse(led.set_blink_rate(fps))   # same rate: no republish
            (msg,) = node.publisher.messages
            self.assertEqual(msg.led, UserLed.USER_LED_2)
            self.assertEqual(msg.color, UserLed.COLOR_RED)
            self.assertEqual(msg.blink_period, SENSING_FPS_TO_LED_PERIOD_MS[fps])
            self.assertEqual(msg.duty_cycle, 0.5)

    def test_adaptive_sequence_publishes_only_changes(self):
        node = FakeNode()
        led = SensingLedController(node)
        for fps in [10.0, 10.0, 1.0, 1.0, 0.2, 10.0]:
            led.set_blink_rate(fps)
        self.assertEqual([m.blink_period for m in node.publisher.messages], [400, 1000, 2000, 400])

    def test_turn_off_is_idempotent_but_startup_call_publishes(self):
        node = FakeNode()
        led = SensingLedController(node)
        self.assertTrue(led.turn_off())    # startup: state unknown -> publishes
        self.assertFalse(led.turn_off())   # already off
        off = node.publisher.messages[0]
        self.assertEqual((off.led, off.color, off.blink_period, off.duty_cycle),
                         (UserLed.USER_LED_2, UserLed.COLOR_OFF, 1000, 0.0))
        # Episode 1 at 5 Hz, then off; episode 2 at 5 Hz again must republish.
        self.assertTrue(led.set_blink_rate(5.0))
        self.assertTrue(led.turn_off())
        self.assertFalse(led.turn_off())
        self.assertTrue(led.set_blink_rate(5.0))
        colors = [m.color for m in node.publisher.messages]
        self.assertEqual(colors, [UserLed.COLOR_OFF, UserLed.COLOR_RED, UserLed.COLOR_OFF, UserLed.COLOR_RED])

    def test_unsupported_fps_is_skipped_without_disabling(self):
        node = FakeNode()
        led = SensingLedController(node)
        self.assertFalse(led.set_blink_rate(3.0))
        self.assertEqual(node.publisher.messages, [])
        self.assertTrue(led.enabled)
        self.assertEqual(len(node.logger.warnings), 1)
        self.assertTrue(led.set_blink_rate(1.0))     # later updates still work

    def test_publish_failure_disables_without_raising(self):
        node = FakeNode(fail=True)
        led = SensingLedController(node)
        self.assertFalse(led.set_blink_rate(10.0))
        self.assertFalse(led.enabled)
        self.assertFalse(led.turn_off())
        self.assertFalse(led.set_blink_rate(1.0))
        self.assertEqual(len(node.logger.warnings), 1)

    def test_disabled_by_flag(self):
        node = FakeNode()
        led = SensingLedController(node, enabled=False)
        self.assertEqual(node.created, [])
        self.assertFalse(led.set_blink_rate(10.0))
        self.assertFalse(led.turn_off())

    def test_missing_message_package_disables_without_raising(self):
        node = FakeNode()
        with mock.patch.dict(sys.modules, {"turtlebot4_msgs.msg": None}):
            led = SensingLedController(node)
        self.assertFalse(led.enabled)
        self.assertFalse(led.set_blink_rate(10.0))
        self.assertEqual(len(node.logger.warnings), 1)


@unittest.skipUnless(os.environ.get("ROS_LOCALHOST_ONLY") == "1" and os.environ.get("ROS_DOMAIN_ID") not in (None, "", "5"),
                     "needs an isolated ROS domain (ROS_LOCALHOST_ONLY=1, ROS_DOMAIN_ID != 5)")
class RosDeliveryTest(unittest.TestCase):
    """Real rclpy publisher/subscriber in an isolated domain: QoS compatibility,
    delivery to a default-QoS subscriber, and publishing with no subscriber."""

    def test_delivery_and_no_subscriber(self):
        import rclpy
        rclpy.init()
        try:
            node = rclpy.create_node("led_test_node")
            led = SensingLedController(node, "/led_test/hmi/led")
            self.assertTrue(led.turn_off())          # no subscriber yet: no error
            received = []
            node.create_subscription(UserLed, "/led_test/hmi/led", received.append, 10)
            deadline = time.monotonic() + 5.0
            while led._publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            self.assertTrue(led.set_blink_rate(5.0))
            deadline = time.monotonic() + 5.0
            while not received and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].blink_period, 667)
            node.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
