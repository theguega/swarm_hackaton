#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import rospy
import math
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# Parameters (configurable)
NUM_ROBOTS = 7
LEADER_ID = 0
RADIUS = 2.0  # Circle radius around leader
UPDATE_RATE = 10  # Hz
KP_LINEAR = 1.0  # Proportional gain for linear velocity
KP_ANGULAR = 2.0  # Proportional gain for angular velocity
SEPARATION_DISTANCE = 0.5  # Minimum distance for collision avoidance
OBSTACLE_THRESHOLD = 0.5  # Minimum distance to obstacles for avoidance


class SwarmController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("swarm_controller", anonymous=True)

        # Subscribers and publishers
        self.model_states_sub = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.model_states_callback
        )
        self.laser_subs = [
            rospy.Subscriber(
                "/hero_{}/laser".format(i), LaserScan, self.laser_callback, i
            )
            for i in range(NUM_ROBOTS)
        ]
        self.cmd_vel_pubs = [
            rospy.Publisher("/hero_{}/cmd_vel".format(i), Twist, queue_size=10)
            for i in range(NUM_ROBOTS)
        ]

        # Robot positions (x, y, theta)
        self.positions = [(0.0, 0.0, 0.0)] * NUM_ROBOTS

        # Laser data for obstacle avoidance
        self.laser_data = [None] * NUM_ROBOTS

        # Rate for update loop
        self.rate = rospy.Rate(UPDATE_RATE)

        # Leader's position (updated from model_states)
        self.leader_pos = (0.0, 0.0, 0.0)

        # Target positions for followers
        self.targets = [(0.0, 0.0)] * (NUM_ROBOTS - 1)

        # Separation forces
        self.separation_forces = [(0.0, 0.0)] * (NUM_ROBOTS - 1)

        # Main loop
        self.run()

    def model_states_callback(self, msg):
        # Extract positions from model_states
        for i in range(NUM_ROBOTS):
            model_name = "hero_{}".format(i)
            if model_name in msg.name:
                idx = msg.name.index(model_name)
                pose = msg.pose[idx]
                self.positions[i] = (
                    pose.position.x,
                    pose.position.y,
                    self.get_yaw_from_quaternion(pose.orientation),
                )

        # Update leader position
        self.leader_pos = self.positions[LEADER_ID]

        # Compute targets for followers
        self.compute_targets()

        # Compute separation forces
        self.compute_separation()

    def laser_callback(self, msg, robot_id):
        # Store laser scan data for obstacle avoidance
        self.laser_data[robot_id] = msg

    def get_yaw_from_quaternion(self, orientation):
        # Convert quaternion to yaw
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (
            orientation.y * orientation.y + orientation.z * orientation.z
        )
        return math.atan2(siny_cosp, cosy_cosp)

    def compute_targets(self):
        # Compute target positions in a circle around leader
        leader_x, leader_y, _ = self.leader_pos
        for i in range(1, NUM_ROBOTS):
            angle = 2 * math.pi * (i - 1) / (NUM_ROBOTS - 1)
            target_x = leader_x + RADIUS * math.cos(angle)
            target_y = leader_y + RADIUS * math.sin(angle)
            self.targets[i - 1] = (target_x, target_y)

        # Log targets for debugging
        for i, target in enumerate(self.targets):
            print(
                "Follower {} target: ({:.2f}, {:.2f})".format(
                    i + 1, target[0], target[1]
                )
            )

    def compute_separation(self):
        # Compute separation forces to avoid collisions
        for i in range(1, NUM_ROBOTS):
            force_x, force_y = 0.0, 0.0
            robot_x, robot_y, _ = self.positions[i]
            for j in range(1, NUM_ROBOTS):
                if i != j:
                    other_x, other_y, _ = self.positions[j]
                    dist = math.sqrt(
                        (robot_x - other_x) ** 2 + (robot_y - other_y) ** 2
                    )
                    if dist < SEPARATION_DISTANCE and dist > 0:
                        force_x += (robot_x - other_x) / dist
                        force_y += (robot_y - other_y) / dist
            self.separation_forces[i - 1] = (force_x, force_y)

    def compute_obstacle_force(self, robot_id):
        # Compute repulsive force from obstacles using laser scan
        if self.laser_data[robot_id] is None:
            return 0.0, 0.0
        msg = self.laser_data[robot_id]
        force_x, force_y = 0.0, 0.0
        robot_theta = self.positions[robot_id][2]
        for i, r in enumerate(msg.ranges):
            if r < OBSTACLE_THRESHOLD and r > 0.1:  # Ignore very close readings
                angle = msg.angle_min + i * msg.angle_increment + robot_theta
                # Repulsive force inversely proportional to distance squared
                force_x += math.cos(angle) / (r**2)
                force_y += math.sin(angle) / (r**2)
        return force_x, force_y

    def compute_velocity(self, robot_id, target_x, target_y):
        # Compute linear and angular velocity using proportional control with obstacle avoidance
        robot_x, robot_y, robot_theta = self.positions[robot_id]

        # Add obstacle repulsive force to target
        obs_fx, obs_fy = self.compute_obstacle_force(robot_id)
        target_x += obs_fx * 2.0  # Increased scale for stronger repulsion
        target_y += obs_fy * 2.0

        # Distance to adjusted target
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx**2 + dy**2)

        # Desired angle to adjusted target
        desired_theta = math.atan2(dy, dx)

        # Angular error
        angular_error = desired_theta - robot_theta
        angular_error = math.atan2(
            math.sin(angular_error), math.cos(angular_error)
        )  # Normalize to [-pi, pi]

        # Linear velocity (proportional to distance)
        linear_vel = KP_LINEAR * distance

        # Angular velocity (proportional to angular error)
        angular_vel = KP_ANGULAR * angular_error

        # Cap velocities for safety
        linear_vel = min(linear_vel, 0.5)
        angular_vel = max(min(angular_vel, 1.0), -1.0)

        return linear_vel, angular_vel

    def run(self):
        while not rospy.is_shutdown():
            for i in range(1, NUM_ROBOTS):
                # Get target position
                target_x, target_y = self.targets[i - 1]

                # Add separation force
                sep_x, sep_y = self.separation_forces[i - 1]
                target_x += sep_x * 0.1  # Scale separation
                target_y += sep_y * 0.1

                # Compute velocity
                linear_vel, angular_vel = self.compute_velocity(i, target_x, target_y)

                # Create Twist message
                twist = Twist()
                twist.linear.x = linear_vel
                twist.angular.z = angular_vel

                # Publish command
                self.cmd_vel_pubs[i].publish(twist)

                # Log for debugging
                print(
                    "Robot {}: pos ({:.2f}, {:.2f}), vel {:.2f}, ang {:.2f}".format(
                        i,
                        self.positions[i][0],
                        self.positions[i][1],
                        linear_vel,
                        angular_vel,
                    )
                )

            # Sleep to maintain rate
            self.rate.sleep()


if __name__ == "__main__":
    try:
        SwarmController()
    except rospy.ROSInterruptException:
        pass
