#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import time


class SwarmAPI:
    def __init__(
        self, swarm_size=4, prefix="/hero_", topic="/position_controller/cmd_vel"
    ):
        rospy.init_node("swarm_api", anonymous=True)
        self.swarm_size = swarm_size
        self.prefix = prefix
        self.topic = topic
        self.pubs = {}
        for i in range(swarm_size):
            ns = f"{prefix}{i}"
            topic_full = f"{ns}{topic}"
            self.pubs[ns] = rospy.Publisher(topic_full, Twist, queue_size=1)
        rospy.sleep(1)  # wait a moment to ensure publishers connect

    def move_robot(self, idx, vx, vy=0.0, omega=0.0, duration=1.0):
        """Send a velocity command for 'duration' seconds to robot idx."""
        ns = f"{self.prefix}{idx}"
        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.angular.z = omega
        end_time = time.time() + duration
        rate = rospy.Rate(10)
        while time.time() < end_time and not rospy.is_shutdown():
            self.pubs[ns].publish(cmd)
            rate.sleep()
        # stop after move
        self.stop_robot(idx)

    def stop_robot(self, idx):
        ns = f"{self.prefix}{idx}"
        cmd = Twist()
        self.pubs[ns].publish(cmd)

    def stop_all(self):
        for i in range(self.swarm_size):
            self.stop_robot(i)

    def move_all(self, vx, vy=0.0, omega=0.0, duration=1.0):
        for i in range(self.swarm_size):
            self.move_robot(i, vx, vy, omega, duration)


# example CLI test
if __name__ == "__main__":
    api = SwarmAPI(swarm_size=3)
    api.move_all(0.2, 0.0, 0.0, 2.0)
    api.stop_all()
