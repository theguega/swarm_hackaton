#!/usr/bin/env python3
"""
Swarm Exploration + Collective Retrieval Simulation
---------------------------------------------------
- Each robot starts in EXPLORATION mode, wandering and avoiding collisions.
- When one detects a close obstacle, it enters SURROUND mode to investigate.
- If it identifies the obstacle as a movable block, it broadcasts its position.
- Other robots switch to COLLABORATE mode, moving toward that location.
- Once enough robots gather, they collectively return the block to the BEACON (home base).
- After successful delivery, all robots resume exploration.
"""

import rospy
import math
import random
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# ===========================
# Parameters
# ===========================
NUM_ROBOTS = 12
NEIGHBOR_RADIUS = 2.5
ROBOT_REPULSION_DIST = 1.0
LASER_DANGER_DIST = 0.8

# Weight constants
W_SEPARATION = 8.0
W_COHESION = 3.0
W_ALIGNMENT = 2
W_LASER = 8.0
W_VORTEX = 0.6
W_NOISE = 0.8
W_EXPLORE = 1

# Speed limits
MAX_LINEAR_SPEED = 0.3
MAX_ANGULAR_SPEED = 1.0
SMOOTHING_FACTOR = 0.5

STATE_EXPLORE = 0
STATE_SURROUND_OBSTACLE = 1
STATE_BLOCK_FOUND = 2
STATE_RETURN_HOME = 3


class SwarmController:
    """Main controller class managing all robot behaviors and ROS communication."""

    def __init__(self):
        rospy.init_node("swarm_controller")
        rospy.loginfo("Swarm Controller initialized with %d robots.", NUM_ROBOTS)

        self.positions = {i: (0.0, 0.0) for i in range(NUM_ROBOTS)}
        self.velocities = {i: (0.0, 0.0) for i in range(NUM_ROBOTS)}
        self.quaternions = {i: None for i in range(NUM_ROBOTS)}
        self.laser_data = {i: None for i in range(NUM_ROBOTS)}

        self.last_cmd_linear = {i: 0.0 for i in range(NUM_ROBOTS)}
        self.last_cmd_angular = {i: 0.0 for i in range(NUM_ROBOTS)}
        self.noise_offsets = {
            i: random.uniform(0, 2 * math.pi) for i in range(NUM_ROBOTS)
        }

        # Global centroid for exploration force
        self.global_centroid = (0.0, 0.0)

        # ROS interfaces
        self.cmd_pubs = []
        for i in range(NUM_ROBOTS):
            robot_id = f"hero_{i}"
            rospy.Subscriber(f"/{robot_id}/odom", Odometry, self.odom_callback, i)
            rospy.Subscriber(f"/{robot_id}/laser", LaserScan, self.laser_callback, i)
            self.cmd_pubs.append(
                rospy.Publisher(f"/{robot_id}/cmd_vel", Twist, queue_size=1)
            )

        self.rate = rospy.Rate(20)

    # --- Callbacks ---
    def odom_callback(self, msg, robot_id):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y

        self.positions[robot_id] = (x, y)
        self.quaternions[robot_id] = q

        yaw = self.get_yaw(robot_id)
        g_vx = vx * math.cos(yaw) - vy * math.sin(yaw)
        g_vy = vx * math.sin(yaw) + vy * math.cos(yaw)
        self.velocities[robot_id] = (g_vx, g_vy)

    def laser_callback(self, msg, robot_id):
        self.laser_data[robot_id] = msg

    # --- Helpers ---
    def get_yaw(self, robot_id):
        q = self.quaternions[robot_id]
        if q is None:
            return 0.0
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize(self, v):
        mag = math.sqrt(v[0] ** 2 + v[1] ** 2)
        return (v[0] / mag, v[1] / mag) if mag > 1e-5 else (0.0, 0.0)

    # --- Force Computations ---
    def compute_boids_forces(self, r_id):
        """Computes separation, cohesion, and alignment forces."""
        sep, coh, ali = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
        my_x, my_y = self.positions[r_id]
        my_vx, my_vy = self.velocities[r_id]

        center_x = center_y = avg_vx = avg_vy = 0.0
        neighbors = 0

        for j in range(NUM_ROBOTS):
            if j == r_id:
                continue
            ox, oy = self.positions[j]
            dist_sq = (my_x - ox) ** 2 + (my_y - oy) ** 2
            if dist_sq == 0:
                continue

            if dist_sq < ROBOT_REPULSION_DIST**2:
                inv_d2 = 1.0 / dist_sq
                sep = (sep[0] + (my_x - ox) * inv_d2, sep[1] + (my_y - oy) * inv_d2)

            if dist_sq < NEIGHBOR_RADIUS**2:
                neighbors += 1
                center_x += ox
                center_y += oy
                ovx, ovy = self.velocities[j]
                avg_vx += ovx
                avg_vy += ovy

        if neighbors > 0:
            center_x /= neighbors
            center_y /= neighbors
            avg_vx /= neighbors
            avg_vy /= neighbors
            coh = (center_x - my_x, center_y - my_y)
            ali = (avg_vx - my_vx, avg_vy - my_vy)

        return sep, self.normalize(coh), self.normalize(ali)

    def compute_env_potential(self, r_id):
        """No hard-coded arena boundaries; rely on laser sensors for wall avoidance."""
        return (0.0, 0.0)

    def compute_laser_repulsion(self, r_id):
        """Repulsive vector from nearby obstacles using laser data."""
        msg = self.laser_data[r_id]
        if msg is None:
            return (0.0, 0.0)

        yaw = self.get_yaw(r_id)
        net_x = net_y = 0.0
        for i in range(0, len(msg.ranges), 2):
            r = msg.ranges[i]
            if msg.range_min < r < LASER_DANGER_DIST:
                angle_local = msg.angle_min + i * msg.angle_increment
                angle_global = yaw + angle_local
                force = (LASER_DANGER_DIST - r) ** 2
                net_x -= math.cos(angle_global) * force
                net_y -= math.sin(angle_global) * force

        return self.normalize((net_x, net_y))

    def compute_vortex(self, r_id):
        """Swirling force for organic, natural motion (fixed scale, no arena dependency)."""
        x, y = self.positions[r_id]
        vx, vy = y, -x  # clockwise swirl
        scale = 0.5  # Fixed scale for consistent motion without hard-coded arena size
        vn = self.normalize((vx, vy))
        return (vn[0] * scale, vn[1] * scale)

    # --- Main Loop ---
    def run(self):
        """Main control loop: combines forces and publishes velocity commands."""
        while not rospy.is_shutdown():
            # Compute global centroid for exploration force
            sum_x = sum_y = 0.0
            for pos in self.positions.values():
                sum_x += pos[0]
                sum_y += pos[1]
            self.global_centroid = (sum_x / NUM_ROBOTS, sum_y / NUM_ROBOTS)

            for i in range(NUM_ROBOTS):
                if self.quaternions[i] is None:
                    continue

                f_sep, f_coh, f_ali = self.compute_boids_forces(i)
                f_env = self.compute_env_potential(i)
                f_laser = self.compute_laser_repulsion(i)
                f_vortex = self.compute_vortex(i)
                self.noise_offsets[i] += random.uniform(-0.1, 0.1)
                f_noise = (
                    math.cos(self.noise_offsets[i]),
                    math.sin(self.noise_offsets[i]),
                )

                # New: Exploration force - repel from global centroid
                cent_x, cent_y = self.global_centroid
                my_x, my_y = self.positions[i]
                dx = my_x - cent_x
                dy = my_y - cent_y
                f_explore = self.normalize((dx, dy))

                f_sep_n = self.normalize(f_sep)
                des_x = (
                    f_sep_n[0] * W_SEPARATION
                    + f_coh[0] * W_COHESION
                    + f_ali[0] * W_ALIGNMENT
                    + f_laser[0] * W_LASER
                    + f_vortex[0] * W_VORTEX
                    + f_explore[0] * W_EXPLORE
                    + f_noise[0] * W_NOISE
                )
                des_y = (
                    f_sep_n[1] * W_SEPARATION
                    + f_coh[1] * W_COHESION
                    + f_ali[1] * W_ALIGNMENT
                    + f_laser[1] * W_LASER
                    + f_vortex[1] * W_VORTEX
                    + f_explore[1] * W_EXPLORE
                    + f_noise[1] * W_NOISE
                )

                current_yaw = self.get_yaw(i)
                desired_yaw = math.atan2(des_y, des_x)
                yaw_error = math.atan2(
                    math.sin(desired_yaw - current_yaw),
                    math.cos(desired_yaw - current_yaw),
                )

                sep_mag = math.sqrt(f_sep[0] ** 2 + f_sep[1] ** 2)
                brake = 1.0 / (1.0 + 2.0 * sep_mag)
                target_lin = MAX_LINEAR_SPEED * brake
                target_ang = MAX_ANGULAR_SPEED * yaw_error

                cmd_lin = (
                    SMOOTHING_FACTOR * target_lin
                    + (1 - SMOOTHING_FACTOR) * self.last_cmd_linear[i]
                )
                cmd_ang = (
                    SMOOTHING_FACTOR * target_ang
                    + (1 - SMOOTHING_FACTOR) * self.last_cmd_angular[i]
                )
                self.last_cmd_linear[i] = cmd_lin
                self.last_cmd_angular[i] = cmd_ang

                twist = Twist()
                twist.linear.x = cmd_lin
                twist.angular.z = cmd_ang
                self.cmd_pubs[i].publish(twist)

            self.rate.sleep()


if __name__ == "__main__":
    try:
        SwarmController().run()
    except rospy.ROSInterruptException:
        pass
