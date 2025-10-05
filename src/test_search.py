#!/usr/bin/env python3
# swarm_search_surround.py
import rospy
import numpy as np
import math
import random
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


class SwarmSearchSurround:
    def __init__(self, n_robots=4, Lx=5.0, Ly=5.0):
        rospy.init_node("swarm_search_surround", anonymous=True)
        self.n_robots = n_robots
        self.prefix = "/hero_"
        self.cmd_topic = "/velocity_controller/cmd_vel"
        self.odom_topic = "/odom"

        # Params
        self.rate = rospy.Rate(10)
        self.speed = 0.2
        self.angular_speed = 0.4
        self.reach_tolerance = 0.1

        # Internal states
        self.positions = {i: np.zeros(2) for i in range(n_robots)}
        self.yaws = {i: 0.0 for i in range(n_robots)}
        self.target_found = False
        self.chosen_target = None
        self.surround_positions = None
        self.all_stopped = False

        # Publishers & subscribers
        self.cmd_pubs = {}
        for i in range(n_robots):
            ns = f"{self.prefix}{i}"
            self.cmd_pubs[i] = rospy.Publisher(
                f"{ns}{self.cmd_topic}", Twist, queue_size=1
            )
            rospy.Subscriber(
                f"{ns}{self.odom_topic}", Odometry, self.odom_callback, callback_args=i
            )

        rospy.sleep(1.0)
        rospy.loginfo(f"[Swarm] Started with {n_robots} robots.")

        # Precompute sweeping paths
        self.paths = [self.wall_then_spiral(i, Lx, Ly) for i in range(n_robots)]
        self.indices = np.zeros(n_robots, dtype=int)

    def odom_callback(self, msg, idx):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([o.x, o.y, o.z, o.w])
        self.positions[idx] = np.array([p.x, p.y])
        self.yaws[idx] = yaw

    def wall_then_spiral(self, i, Lx, Ly):
        """Just generate sweeping goals"""
        x0, y0 = random.uniform(0, Lx), random.uniform(0, Ly)
        x_range = np.linspace(0.5, Lx - 0.5, 10)
        y_range = np.linspace(0.5, Ly - 0.5, 10)
        path = []
        for j, y in enumerate(y_range):
            xs = x_range if j % 2 == 0 else x_range[::-1]
            for x in xs:
                path.append(np.array([x, y]))
        random.shuffle(path)
        return path

    def move_toward(self, idx, goal):
        """Simple proportional velocity toward goal"""
        pos = self.positions[idx]
        yaw = self.yaws[idx]
        dx, dy = goal - pos
        dist = np.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        heading_error = self.angle_wrap(angle_to_goal - yaw)
        cmd = Twist()
        if abs(heading_error) > 0.2:
            cmd.angular.z = self.angular_speed * np.sign(heading_error)
        elif dist > self.reach_tolerance:
            cmd.linear.x = self.speed
        self.cmd_pubs[idx].publish(cmd)
        return dist < self.reach_tolerance

    def stop_robot(self, idx):
        self.cmd_pubs[idx].publish(Twist())

    def surround_target(self, target):
        radius = 0.6
        surrounds = {}
        for j in range(self.n_robots):
            angle = 2 * np.pi * j / self.n_robots
            sx = target[0] + radius * math.cos(angle)
            sy = target[1] + radius * math.sin(angle)
            surrounds[j] = np.array([sx, sy])
        return surrounds

    def angle_wrap(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def step(self):
        if self.all_stopped:
            for i in range(self.n_robots):
                self.stop_robot(i)
            return

        if self.target_found:
            done = True
            for i in range(self.n_robots):
                reached = self.move_toward(i, self.surround_positions[i])
                done &= reached
            if done:
                rospy.loginfo(
                    f"✅ All robots surrounding target at {self.chosen_target}"
                )
                self.all_stopped = True
            return

        # Exploration phase
        for i in range(self.n_robots):
            if self.indices[i] < len(self.paths[i]):
                goal = self.paths[i][self.indices[i]]
                reached = self.move_toward(i, goal)
                if reached:
                    self.indices[i] += 1

                # Fake detection event (for demonstration)
                if not self.target_found and random.random() < 0.001:
                    tx, ty = goal
                    if 0.5 < tx < 4.5 and 0.5 < ty < 4.5:
                        rospy.loginfo(
                            f"🎯 Robot {i} found target at ({tx:.2f}, {ty:.2f})"
                        )
                        self.target_found = True
                        self.chosen_target = np.array([tx, ty])
                        self.surround_positions = self.surround_target(
                            self.chosen_target
                        )
                        break
            else:
                self.stop_robot(i)

    def run(self):
        rospy.loginfo("[Swarm] Running search-surround controller...")
        while not rospy.is_shutdown():
            self.step()
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = SwarmSearchSurround(n_robots=4)
        node.run()
    except rospy.ROSInterruptException:
        pass
