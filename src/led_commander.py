#!/usr/bin/env python
import rospy
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
import math
import time


class LEDCommander:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.led_pub = rospy.Publisher(f"/{robot_name}/led", ColorRGBA, queue_size=10)
        self.last_position = None
        self.last_move_time = time.time()
        self.state = "exploring"

        rospy.Subscriber(f"/{robot_name}/odom", Odometry, self.odom_cb)
        rospy.Subscriber(f"/{robot_name}/laser", Range, self.laser_cb)

        self.target_position = (10.0, 10.0)  # example target
        self.rate = rospy.Rate(5)
        rospy.loginfo(f"[{self.robot_name}] LED Commander started.")

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.last_position is not None:
            dist = math.hypot(x - self.last_position[0], y - self.last_position[1])
            if dist > 0.01:
                self.last_move_time = time.time()

        self.last_position = (x, y)

        # Check proximity to target
        tx, ty = self.target_position
        if math.hypot(x - tx, y - ty) < 0.5:
            self.state = "target_found"
        elif time.time() - self.last_move_time > 3.0:
            self.state = "idle"
        else:
            self.state = "exploring"

    def laser_cb(self, msg):
        if msg.range < 0.2:  # too close to obstacle
            self.state = "blocked"

    def publish_led(self):
        color = ColorRGBA()
        if self.state == "exploring":
            color.r, color.g, color.b = 0.0, 1.0, 0.0  # green
        elif self.state == "target_found":
            color.r, color.g, color.b = 0.5, 0.0, 1.0  # purple
        elif self.state == "moving":
            color.r, color.g, color.b = 0.0, 0.0, 1.0  # blue
        elif self.state == "blocked":
            color.r, color.g, color.b = 1.0, 0.0, 0.0  # red
        elif self.state == "idle":
            color.r, color.g, color.b = 1.0, 1.0, 1.0  # white

        color.a = 1.0
        self.led_pub.publish(color)

    def run(self):
        while not rospy.is_shutdown():
            self.publish_led()
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("led_commander")

    # can be launched with different robot name
    robot_name = rospy.get_param("~robot_name", "hero_0")
    node = LEDCommander(robot_name)
    node.run()
