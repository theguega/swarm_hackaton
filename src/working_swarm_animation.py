#!/usr/bin/env python3
import rospy
import re
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import tf
from threading import Lock


class MultiRobotController:
    def __init__(self):
        rospy.init_node("multi_robot_controller", anonymous=True)

        self.robot_data = {}  # robot_name: {odom: msg, pub: Publisher}
        self.lock = Lock()

        self.rate = rospy.Rate(10)  # Hz

        rospy.loginfo("MultiRobotController started.")

        self.discover_robots()
        rospy.Timer(
            rospy.Duration(5.0), self.refresh_robot_list
        )  # refresh list every 5s

        self.run()

    def discover_robots(self):
        topics = rospy.get_published_topics()
        robot_names = set()

        for topic, msg_type in topics:
            match = re.match(r"^/hero_(\d+)/odom$", topic)
            if match:
                robot_id = match.group(1)
                robot_name = f"hero_{robot_id}"
                robot_names.add(robot_name)

        for name in robot_names:
            if name not in self.robot_data:
                self.setup_robot(name)

    def setup_robot(self, robot_name):
        odom_topic = f"/{robot_name}/odom"
        cmd_vel_topic = f"/{robot_name}/cmd_vel"

        rospy.loginfo(f"Setting up {robot_name}")

        rospy.Subscriber(
            odom_topic, Odometry, self.odom_callback, callback_args=robot_name
        )
        pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        self.robot_data[robot_name] = {
            "odom": None,
            "pub": pub,
            "angle_offset": int(robot_name.split("_")[1]) * (math.pi / 15),
        }

    def refresh_robot_list(self, event):
        with self.lock:
            self.discover_robots()

    def odom_callback(self, msg, robot_name):
        with self.lock:
            if robot_name in self.robot_data:
                self.robot_data[robot_name]["odom"] = msg

    def compute_command(self, robot_name, time_now):
        """
        Generate a circular motion for each robot, offset by index.
        """
        # Parameters for circular motion
        r = 0.5 + 0.05 * int(robot_name.split("_")[1])  # radius based on index
        w = 0.2  # angular velocity

        vel_msg = Twist()
        vel_msg.linear.x = r * w
        vel_msg.angular.z = w
        return vel_msg

    def run(self):
        while not rospy.is_shutdown():
            time_now = rospy.Time.now().to_sec()

            with self.lock:
                for robot_name, data in self.robot_data.items():
                    if data["odom"] is None:
                        continue  # wait for odometry

                    cmd = self.compute_command(robot_name, time_now)
                    data["pub"].publish(cmd)

            self.rate.sleep()


if __name__ == "__main__":
    try:
        MultiRobotController()
    except rospy.ROSInterruptException:
        pass
