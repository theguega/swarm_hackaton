#!/usr/bin/env python3

import rospy
import math
import random
import numpy as np
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


# ==========================================
# UPGRADED TUNING CONSTANTS (V2 - Anti-Collision)
# ==========================================
NUM_ROBOTS = 7

# Environment (Square defined from -L to +L)
ARENA_HALF_LENGTH = 5.0

# Ranges
NEIGHBOR_RADIUS = 3.0  # Increased cohesion/alignment range slightly
## MODIFIED ##
# This is the "personal space" bubble. It MUST be larger than the physical robot.
# Make this value larger if collisions persist.
ROBOT_REPULSION_DIST = 1.2
LASER_DANGER_DIST = 1.0

# Weights (The "Personality" of the swarm)
## MODIFIED ##
# Separation is now the most important force by a large margin.
W_SEPARATION = 10.0
W_COHESION = 1.0  # Reduced to prevent overpowering separation
W_ALIGNMENT = 1.5
W_WALL_ENV = 4.0
W_LASER = 5.0
W_VORTEX = 0.6
W_NOISE = 0.3

# Movement Limits & Smoothing
MAX_LINEAR_SPEED = 0.4  # Slightly reduced max speed to improve reaction time
MAX_ANGULAR_SPEED = 1
## MODIFIED ##
# Increased responsiveness (less smooth) to help with fast avoidance.
# A lower value (e.g. 0.1) is more "organic" but worse for collision avoidance.
SMOOTHING_FACTOR = 0.3


class SwarmController:
    def __init__(self):
        rospy.init_node("spectacular_swarm_controller")

        self.positions = {}
        self.velocities = {}  # xy velocities from odom
        self.quaternions = {}
        self.laser_data = {}

        # For smoothing output commands
        self.last_cmd_linear = {i: 0.0 for i in range(NUM_ROBOTS)}
        self.last_cmd_angular = {i: 0.0 for i in range(NUM_ROBOTS)}

        # Noise evolution
        self.noise_offsets = {
            i: random.uniform(0, 2 * math.pi) for i in range(NUM_ROBOTS)
        }

        # ROS Setup (Keep your existing setup, ensure keys are integers 0 to N)
        self.odom_subs = []
        self.laser_subs = []
        self.cmd_pubs = []
        for i in range(NUM_ROBOTS):
            robot_id = f"hero_{i}"  # Ensure this matches your names
            # Initialize data to prevent startup errors
            self.positions[i] = (0.0, 0.0)
            self.velocities[i] = (0.0, 0.0)
            self.quaternions[i] = None
            self.laser_data[i] = None

            self.odom_subs.append(
                rospy.Subscriber(f"/{robot_id}/odom", Odometry, self.odom_callback, i)
            )
            self.laser_subs.append(
                rospy.Subscriber(
                    f"/{robot_id}/laser", LaserScan, self.laser_callback, i
                )
            )
            self.cmd_pubs.append(
                rospy.Publisher(
                    f"/{robot_id}/cmd_vel", Twist, queue_size=1
                )  # Lower queue size for real-time
            )

        self.rate = rospy.Rate(20)  # 20Hz is good for control
        rospy.loginfo("Spectacular Swarm Controller Started.")

    # ===========================
    # Callbacks (Simplified)
    # ===========================
    def odom_callback(self, msg, robot_id):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.positions[robot_id] = (x, y)
        self.quaternions[robot_id] = msg.pose.pose.orientation

        # Get current actual velocity (useful for alignment and damping)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        # Need to rotate local vx,vy into global frame if odom gives local twist
        current_yaw = self.get_yaw(robot_id)
        g_vx = vx * math.cos(current_yaw) - vy * math.sin(current_yaw)
        g_vy = vx * math.sin(current_yaw) + vy * math.cos(current_yaw)
        self.velocities[robot_id] = (g_vx, g_vy)

    def laser_callback(self, msg, robot_id):
        # Just store data, processing happens in main loop
        self.laser_data[robot_id] = msg

    # Helpers
    def get_yaw(self, robot_id):
        q = self.quaternions[robot_id]
        if q is None:
            return 0.0
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize(self, v):
        m = math.sqrt(v[0] ** 2 + v[1] ** 2)
        if m > 1e-5:
            return (v[0] / m, v[1] / m)
        return (0.0, 0.0)

    # ===========================
    # Force Computations
    # ===========================

    def compute_boids_forces(self, r_id):
        """Computes standard Separation, Cohesion, Alignment."""
        sep_x, sep_y = 0.0, 0.0
        coh_x, coh_y = 0.0, 0.0
        ali_x, ali_y = 0.0, 0.0

        my_x, my_y = self.positions[r_id]
        my_vx, my_vy = self.velocities[r_id]

        neighbors = 0
        center_x, center_y = 0.0, 0.0
        avg_vx, avg_vy = 0.0, 0.0

        for j in range(NUM_ROBOTS):
            if j == r_id:
                continue
            ox, oy = self.positions[j]
            dist_sq = (my_x - ox) ** 2 + (my_y - oy) ** 2

            # Separation (Robot-Robot) - Short range, very strong
            if dist_sq < ROBOT_REPULSION_DIST**2 and dist_sq > 0:
                dist = math.sqrt(dist_sq)
                ## MODIFIED ##
                # The force is now stronger and simpler. It's inversely
                # proportional to the squared distance. This makes it
                # extremely powerful at very close ranges.
                inv_dist_sq = 1.0 / dist_sq
                sep_x += (my_x - ox) * inv_dist_sq
                sep_y += (my_y - oy) * inv_dist_sq

            # Cohesion & Alignment - Medium range
            if dist_sq < NEIGHBOR_RADIUS**2:
                neighbors += 1
                center_x += ox
                center_y += oy
                ovx, ovy = self.velocities[j]
                avg_vx += ovx
                avg_vy += ovy

        if neighbors > 0:
            # Cohesion: Vector towards average position
            center_x /= neighbors
            center_y /= neighbors
            coh_x = center_x - my_x
            coh_y = center_y - my_y

            # Alignment: Vector matching average velocity
            avg_vx /= neighbors
            avg_vy /= neighbors
            ali_x = avg_vx - my_vx
            ali_y = avg_vy - my_vy

        # Return raw (unnormalized) separation vector and normalized others
        # This preserves the critical magnitude of the separation force.
        return (
            (sep_x, sep_y),
            self.normalize((coh_x, coh_y)),
            self.normalize((ali_x, ali_y)),
        )

    def compute_env_potential(self, r_id):
        """Global soft forces from the known square boundaries."""
        # Prevents getting stuck in corners by pushing immediately
        # when not near the center.
        x, y = self.positions[r_id]
        fx, fy = 0.0, 0.0

        # Force increases exponentially as we approach the boundary defined by ARENA_HALF_LENGTH
        # Soft buffer area
        buffer = 1.0

        # Push left from right wall
        if x > (ARENA_HALF_LENGTH - buffer):
            dist = ARENA_HALF_LENGTH - x
            fx -= 1.0 / max(0.1, dist**2)  # Push left (-)

        # Push right from left wall
        if x < (-ARENA_HALF_LENGTH + buffer):
            dist = x - (-ARENA_HALF_LENGTH)
            fx += 1.0 / max(0.1, dist**2)  # Push right (+)

        # Push down from top wall
        if y > (ARENA_HALF_LENGTH - buffer):
            dist = ARENA_HALF_LENGTH - y
            fy -= 1.0 / max(0.1, dist**2)

        # Push up from bottom wall
        if y < (-ARENA_HALF_LENGTH + buffer):
            dist = y - (-ARENA_HALF_LENGTH)
            fy += 1.0 / max(0.1, dist**2)

        return self.normalize((fx, fy))

    def compute_laser_repulsion(self, r_id):
        """Accurate local obstacle avoidance from laser."""
        msg = self.laser_data[r_id]
        if msg is None:
            return (0.0, 0.0)

        current_yaw = self.get_yaw(r_id)
        net_x, net_y = 0.0, 0.0
        count = 0

        # Process a subset of rays to save CPU
        step = 2
        for i in range(0, len(msg.ranges), step):
            r = msg.ranges[i]
            # Check valid range and danger zone
            if msg.range_min < r < LASER_DANGER_DIST:
                # Angle of obstacle relative to robot
                angle_local = msg.angle_min + i * msg.angle_increment
                angle_global = current_yaw + angle_local

                # IMPORTANT: Vector pointing FROM obstacle TO robot is negative
                # Force strength (stronger when closer)
                force = (LASER_DANGER_DIST - r) ** 2

                # Add vector pushing away from this point
                net_x -= math.cos(angle_global) * force
                net_y -= math.sin(angle_global) * force
                count += 1

        # If in a corner, this creates a net vector bisecting the corner
        return self.normalize((net_x, net_y))

    def compute_vortex(self, r_id):
        """Organic swirling force around the arena center (0,0)."""
        x, y = self.positions[r_id]
        # Vector towards center is (-x, -y)
        # Tangent (rotation counter-clockwise) is (-y, x)
        # Tangent (rotation clockwise) is (y, -x)

        # Use clockwise swirl
        vx = y
        vy = -x

        # Stronger further out, weaker in center
        dist_from_center = math.sqrt(x**2 + y**2)
        scale = min(1.0, dist_from_center / ARENA_HALF_LENGTH)

        vn = self.normalize((vx, vy))
        return (vn[0] * scale, vn[1] * scale)

    # ===========================
    # Main Loop
    # ===========================
    def run(self):
        step_count = 0
        while not rospy.is_shutdown():
            step_count += 1
            now_sec = rospy.get_time()

            for i in range(NUM_ROBOTS):
                if self.quaternions[i] is None:
                    continue

                # 1. Compute Component Forces
                # ## MODIFIED ## Separation force is now kept separate and NOT normalized
                f_sep_raw, f_coh, f_ali = self.compute_boids_forces(i)
                f_env = self.compute_env_potential(i)
                f_laser = self.compute_laser_repulsion(i)
                f_vortex = self.compute_vortex(i)
                self.noise_offsets[i] += random.uniform(-0.1, 0.1)
                f_noise = (
                    math.cos(self.noise_offsets[i]),
                    math.sin(self.noise_offsets[i]),
                )

                # 2. Weighted Sum
                # We normalize the raw separation force before weighting it
                f_sep_norm = self.normalize(f_sep_raw)
                des_x = (
                    f_sep_norm[0] * W_SEPARATION
                    + f_coh[0] * W_COHESION
                    + f_ali[0] * W_ALIGNMENT
                    + f_env[0] * W_WALL_ENV
                    + f_laser[0] * W_LASER
                    + f_vortex[0] * W_VORTEX
                    + f_noise[0] * W_NOISE
                )

                des_y = (
                    f_sep_norm[1] * W_SEPARATION
                    + f_coh[1] * W_COHESION
                    + f_ali[1] * W_ALIGNMENT
                    + f_env[1] * W_WALL_ENV
                    + f_laser[1] * W_LASER
                    + f_vortex[1] * W_VORTEX
                    + f_noise[1] * W_NOISE
                )

                # 3. Convert to Local Twist Command with Dynamic Braking
                current_yaw = self.get_yaw(i)
                desired_yaw = math.atan2(des_y, des_x)
                yaw_error = math.atan2(
                    math.sin(desired_yaw - current_yaw),
                    math.cos(desired_yaw - current_yaw),
                )

                ## NEW: DYNAMIC BRAKING LOGIC ##
                # Calculate the magnitude of the raw separation force. This tells us
                # how urgently we need to avoid a neighbor.
                separation_magnitude = math.sqrt(f_sep_raw[0] ** 2 + f_sep_raw[1] ** 2)
                # Create a speed suppression factor. If magnitude is 0, factor is 1 (full speed).
                # As magnitude increases, factor drops towards 0 (brake).
                # The '2.0' is a tuning parameter; a smaller number makes braking more aggressive.
                speed_suppression_factor = 1.0 / (1.0 + 2.0 * separation_magnitude)

                # The final speed is the max speed, scaled by how much we need to brake.
                target_lin = MAX_LINEAR_SPEED * speed_suppression_factor
                target_ang = MAX_ANGULAR_SPEED * yaw_error

                # Apply smoothing
                cmd_lin = (SMOOTHING_FACTOR * target_lin) + (
                    (1.0 - SMOOTHING_FACTOR) * self.last_cmd_linear[i]
                )
                cmd_ang = (SMOOTHING_FACTOR * target_ang) + (
                    (1.0 - SMOOTHING_FACTOR) * self.last_cmd_angular[i]
                )

                self.last_cmd_linear[i] = cmd_lin
                self.last_cmd_angular[i] = cmd_ang

                # Publish
                twist = Twist()
                twist.linear.x = cmd_lin
                twist.angular.z = cmd_ang
                self.cmd_pubs[i].publish(twist)

            self.rate.sleep()


if __name__ == "__main__":
    try:
        controller = SwarmController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
