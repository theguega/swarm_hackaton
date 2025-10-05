#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import tf

# --- PARAMETERS ---
Lx, Ly = 20.0, 20.0  # meters
cell_size = 1.0
nx, ny = int(Lx / cell_size), int(Ly / cell_size)
N_ROBOTS = 10
ROBOT_IDS = [f"robot_{i}" for i in range(N_ROBOTS)]
TARGETS = [(15, 15), (5, 5)]  # Example target positions in grid coordinates
GOAL_TOLERANCE = 0.5  # meters
LINEAR_SPEED = 0.3  # m/s
ANGULAR_SPEED = 0.5  # rad/s
SURROUND_RADIUS = 3.0 # meters
MARGIN = 1.0 # minimum distance from walls

class SwarmController:
    def __init__(self):
        rospy.init_node("swarm_controller")

        self.robot_poses = [None] * N_ROBOTS
        self.robot_orientations = [None] * N_ROBOTS

        self.odom_subs = []
        self.laser_subs = []
        self.cmd_pubs = []

        for i, robot_id in enumerate(ROBOT_IDS):
            self.odom_subs.append(
                rospy.Subscriber(f"/{robot_id}/odom", Odometry, self.odom_callback, i)
            )
            self.laser_subs.append(
                rospy.Subscriber(f"/{robot_id}/laser", LaserScan, self.laser_callback, i)
            )
            self.cmd_pubs.append(
                rospy.Publisher(f"/{robot_id}/cmd_vel", Twist, queue_size=1)
            )

        # --- STATE ---
        self.state = "SEARCHING"
        self.target_found_by = -1
        self.chosen_target = None
        self.surround_positions = [None] * N_ROBOTS
        self.robots_at_surround_pos = [False] * N_ROBOTS

        # --- PATHS ---
        self.paths = self.precompute_paths()
        self.path_indices = np.zeros(N_ROBOTS, dtype=int)

        rospy.loginfo("Swarm controller initialized.")

    def odom_callback(self, msg, robot_index):
        """Updates robot position and orientation."""
        pos = msg.pose.pose.position
        self.robot_poses[robot_index] = (pos.x, pos.y)

        orientation_q = msg.pose.pose.orientation
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        self.robot_orientations[robot_index] = yaw

    def laser_callback(self, msg, robot_index):
        """Placeholder for obstacle avoidance."""
        # TODO: Implement obstacle avoidance logic
        pass

    def world_to_grid(self, world_pos):
        """Converts world coordinates to grid coordinates."""
        if world_pos is None:
            return None
        grid_x = int(world_pos[0] / cell_size)
        grid_y = int(world_pos[1] / cell_size)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_pos):
        """Converts grid coordinates to world coordinates."""
        world_x = (grid_pos[0] + 0.5) * cell_size
        world_y = (grid_pos[1] + 0.5) * cell_size
        return (world_x, world_y)

    def precompute_paths(self):
        """Precomputes Boustrophedon paths for each robot."""
        paths = []
        initial_grid_positions = [self.world_to_grid(self.robot_poses[i]) for i in range(N_ROBOTS)]

        # Wait for initial poses
        while any(pos is None for pos in initial_grid_positions):
             rospy.sleep(0.1)
             initial_grid_positions = [self.world_to_grid(self.robot_poses[i]) for i in range(N_ROBOTS)]


        for i in range(N_ROBOTS):
             x0, y0 = initial_grid_positions[i]
             paths.append(self.wall_then_spiral(x0, y0, nx, ny))
        return paths

    def boustrophedon_path(self, x_range, y_range):
        path = []
        for j, y in enumerate(y_range):
            if j % 2 == 0:
                for x in x_range:
                    path.append((x, y))
            else:
                for x in reversed(x_range):
                    path.append((x, y))
        return path

    def wall_then_spiral(self, x0, y0, nx, ny):
        # ... (Same logic as in the simulation script)
        dists = [x0, nx - 1 - x0, y0, ny - 1 - y0]
        wall = np.argmin(dists)
        if wall == 0:
            start_x, start_y = 0, y0
            x_range = np.arange(0, nx)
            y_range = np.arange(start_y, ny)
        elif wall == 1:
            start_x, start_y = nx - 1, y0
            x_range = np.arange(nx - 1, -1, -1)
            y_range = np.arange(start_y, ny)
        elif wall == 2:
            start_x, start_y = x0, 0
            x_range = np.arange(0, nx)
            y_range = np.arange(0, ny)
        else:
            start_x, start_y = x0, ny - 1
            x_range = np.arange(0, nx)
            y_range = np.arange(ny - 1, -1, -1)

        path = []
        if wall in [0, 1]:
            step = -1 if x0 > start_x else 1
            for xx in range(x0, start_x + step, step):
                path.append((xx, y0))
        else:
            step = -1 if y0 > start_y else 1
            for yy in range(y0, start_y + step, step):
                path.append((x0, yy))
        path.extend(self.boustrophedon_path(x_range, y_range))
        return path


    def move_to_goal(self, robot_index, goal_pos):
        """Moves a robot towards a goal position."""
        if self.robot_poses[robot_index] is None:
            return

        current_pos = self.robot_poses[robot_index]
        current_angle = self.robot_orientations[robot_index]
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < GOAL_TOLERANCE:
            return True  # Goal reached

        angle_to_goal = np.arctan2(dy, dx)
        angle_diff = angle_to_goal - current_angle

        # Normalize angle
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        if angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        twist = Twist()
        if abs(angle_diff) > 0.1:
            twist.angular.z = ANGULAR_SPEED if angle_diff > 0 else -ANGULAR_SPEED
        else:
            twist.linear.x = LINEAR_SPEED

        self.cmd_pubs[robot_index].publish(twist)
        return False # Goal not reached

    def stop_robot(self, robot_index):
        """Stops a robot."""
        self.cmd_pubs[robot_index].publish(Twist())

    def run(self):
        """Main control loop."""
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            if self.state == "SEARCHING":
                self.run_searching_state()
            elif self.state == "TARGET_FOUND":
                self.run_target_found_state()
            elif self.state == "SURROUNDING":
                self.run_surrounding_state()

            rate.sleep()

    def run_searching_state(self):
        """Logic for the SEARCHING state."""
        for i in range(N_ROBOTS):
            if self.robot_poses[i] is None:
                continue

            # Check for target
            current_grid_pos = self.world_to_grid(self.robot_poses[i])
            if current_grid_pos in TARGETS:
                tx, ty = self.grid_to_world(current_grid_pos)

                if tx < MARGIN or tx > Lx - MARGIN or ty < MARGIN or ty > Ly - MARGIN:
                    rospy.logwarn(f"Robot {i} found target at {current_grid_pos}, but too close to wall. Continuing search.")
                else:
                    self.target_found_by = i
                    self.chosen_target = self.grid_to_world(current_grid_pos)
                    self.state = "TARGET_FOUND"
                    rospy.loginfo(f"Robot {i} found target at {current_grid_pos}. Transitioning to TARGET_FOUND.")
                    return # Exit to immediately start the next state


            # Move along Boustrophedon path
            if self.path_indices[i] < len(self.paths[i]):
                goal_grid_pos = self.paths[i][self.path_indices[i]]
                goal_world_pos = self.grid_to_world(goal_grid_pos)

                if self.move_to_goal(i, goal_world_pos):
                    self.path_indices[i] += 1
            else:
                self.stop_robot(i)

    def run_target_found_state(self):
        """Calculates surround positions and transitions to SURROUNDING."""
        rospy.loginfo("Calculating surround positions.")
        tx, ty = self.chosen_target
        for i in range(N_ROBOTS):
            angle = 2 * np.pi * i / N_ROBOTS
            sx = tx + SURROUND_RADIUS * np.cos(angle)
            sy = ty + SURROUND_RADIUS * np.sin(angle)
            self.surround_positions[i] = (sx, sy)

        self.state = "SURROUNDING"
        rospy.loginfo("Transitioning to SURROUNDING state.")

    def run_surrounding_state(self):
        """Moves robots to their surround positions."""
        all_robots_at_pos = True
        for i in range(N_ROBOTS):
            if not self.robots_at_surround_pos[i]:
                goal_pos = self.surround_positions[i]
                if self.move_to_goal(i, goal_pos):
                    self.robots_at_surround_pos[i] = True
                    self.stop_robot(i)
                    rospy.loginfo(f"Robot {i} has reached its surround position.")
                all_robots_at_pos = False

        if all_robots_at_pos:
            rospy.loginfo("All robots are surrounding the target. Mission complete.")
            # Stop all robots for safety
            for i in range(N_ROBOTS):
                self.stop_robot(i)
            rospy.signal_shutdown("Mission Complete")


if __name__ == "__main__":
    try:
        controller = SwarmController()
        # A short delay to ensure all subscribers are connected and poses are received
        rospy.sleep(2)
        controller.run()
    except rospy.ROSInterruptException:
        pass
