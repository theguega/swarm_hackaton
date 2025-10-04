#!/usr/bin/env python3
"""
swarm_controller.py

Spawns N robots into Gazebo (using spawn_model service) and runs the
grid-search -> surround -> stop routine you provided. It publishes
geometry_msgs/Twist to each robot namespace's /cmd_vel topic.

IMPORTANT:
 - Replace robot_model_path param to point to your Herobot URDF or SDF.
 - Adjust per-robot cmd_vel topic if your Herobot uses a different topic name.
"""

import rospy
import numpy as np
import os
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from std_srvs.srv import Empty

def world_pose_from_cell(ix, iy, cell_size, z=0.0, origin=(0.0,0.0)):
    # Convert integer grid cell → world coordinates (center of cell)
    ox, oy = origin
    x = ox + (ix + 0.5) * cell_size
    y = oy + (iy + 0.5) * cell_size
    return x, y, z

class SwarmController(object):
    def __init__(self):
        rospy.init_node('swarm_controller', anonymous=False)

        # Params
        self.Lx = rospy.get_param('~Lx', 20.0)
        self.Ly = rospy.get_param('~Ly', 20.0)
        self.cell_size = rospy.get_param('~cell_size', 1.0)
        self.nx = int(self.Lx / self.cell_size)
        self.ny = int(self.Ly / self.cell_size)
        self.N_robots = int(rospy.get_param('~n_robots', 10))
        self.margin = int(rospy.get_param('~margin', 1))
        self.robot_model_path = rospy.get_param('~robot_model_path', '')

        # Load model XML (URDF or SDF)
        if not self.robot_model_path or not os.path.isfile(self.robot_model_path):
            rospy.logwarn("robot_model_path NOT found or empty. Put your URDF/SDF path in param robot_model_path")
            self.model_xml = ""
        else:
            with open(self.robot_model_path, 'r') as f:
                self.model_xml = f.read()

        # Gazebo spawn client
        rospy.loginfo("Waiting for /gazebo/spawn_urdf_model or /gazebo/spawn_sdf_model service...")
        rospy.wait_for_service('/gazebo/spawn_urdf_model', timeout=15.0)
        self.spawn_srv = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)

        # Prepare publishers for each robot
        self.cmd_pubs = []
        for i in range(self.N_robots):
            ns = f"/robot_{i}"
            topic = ns + "/cmd_vel"
            pub = rospy.Publisher(topic, Twist, queue_size=1)
            self.cmd_pubs.append(pub)

        # Build occupancy grid and targets like your script
        self.grid = -np.ones((self.nx, self.ny), dtype=int)  # -1 unknown, 0 visited, 2 target
        self.targets = []
        num_targets = 2
        while len(self.targets) < num_targets:
            tx, ty = np.random.randint(0, self.nx), np.random.randint(0, self.ny)
            if (tx, ty) not in self.targets:
                self.targets.append((tx, ty))
                self.grid[tx, ty] = 2

        # Place robots initially at random cells
        self.robots = np.column_stack((np.random.randint(0, self.nx, self.N_robots),
                                       np.random.randint(0, self.ny, self.N_robots)))
        # Create paths (same helper functions as original)
        self.paths = [self.wall_then_spiral(int(self.robots[i,0]), int(self.robots[i,1]), self.nx, self.ny) for i in range(self.N_robots)]
        self.indices = np.zeros(self.N_robots, dtype=int)

        # State flags
        self.target_found = False
        self.chosen_target = None
        self.surround_positions = None
        self.all_stopped = False

        # Spawn models at initial world poses
        self.spawn_robots()

        # Main timer to step simulation logic and command robots (10 Hz)
        self.rate = rospy.Rate(10)
        self.run_loop()

    # --- Helper path functions (copied) ---
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
        dists = [x0, nx-1-x0, y0, ny-1-y0]
        wall = int(np.argmin(dists))
        if wall == 0:
            start_x, start_y = 0, y0; x_range = np.arange(0, nx); y_range = np.arange(start_y, ny)
        elif wall == 1:
            start_x, start_y = nx-1, y0; x_range = np.arange(nx-1, -1, -1); y_range = np.arange(start_y, ny)
        elif wall == 2:
            start_x, start_y = x0, 0; x_range = np.arange(0, nx); y_range = np.arange(0, ny)
        else:
            start_x, start_y = x0, ny-1; x_range = np.arange(0, nx); y_range = np.arange(ny-1, -1, -1)

        path = []
        if wall in [0,1]:
            step = -1 if x0 > start_x else 1
            for xx in range(x0, start_x+step, step):
                path.append((xx, y0))
        else:
            step = -1 if y0 > start_y else 1
            for yy in range(y0, start_y+step, step):
                path.append((x0, yy))
        path.extend(self.boustrophedon_path(x_range, y_range))
        return path

    def spawn_robots(self):
        # We convert grid cells to world XY (z=0.0)
        rospy.loginfo("Spawning robots into Gazebo...")
        for i in range(self.N_robots):
            name = f"robot_{i}"
            ix, iy = int(self.robots[i,0]), int(self.robots[i,1])
            x, y, z = world_pose_from_cell(ix, iy, self.cell_size)
            initial_pose = Pose()
            initial_pose.position = Point(x=x, y=y, z=0.0)
            initial_pose.orientation = Quaternion(0,0,0,1)

            if not self.model_xml:
                rospy.logwarn("No model XML loaded; skipping spawn (use param robot_model_path).")
                continue

            try:
                req = SpawnModelRequest()
                req.model_name = name
                req.model_xml = self.model_xml
                req.robot_namespace = f"/{name}"
                req.initial_pose = initial_pose
                req.reference_frame = "world"
                resp = self.spawn_srv(req)
                if resp.success:
                    rospy.loginfo(f"Spawned {name} at cell ({ix},{iy}) -> world ({x:.2f},{y:.2f})")
                else:
                    rospy.logwarn(f"Spawn FAILED for {name}: {resp.status_message}")
            except rospy.ServiceException as e:
                rospy.logerr("Spawn service failed: " + str(e))

    def send_stop(self, i):
        t = Twist()
        self.cmd_pubs[i].publish(t)

    def move_towards_cell(self, i, target_cell):
        """
        Simple discrete step controller: compute velocity needed from robot world pos to target cell world pos.
        NOTE: This is a toy velocity controller for demonstration. Replace with your robot's controller if needed.
        """
        # Compute robot world position using its grid cell
        cx_cell, cy_cell = int(self.robots[i,0]), int(self.robots[i,1])
        cx, cy, _ = world_pose_from_cell(cx_cell, cy_cell, self.cell_size)
        tx_cell, ty_cell = target_cell
        tx, ty, _ = world_pose_from_cell(tx_cell, ty_cell, self.cell_size)

        dx = tx - cx
        dy = ty - cy
        # discrete step: decide a small velocity to move one cell per second-ish
        vx = np.sign(dx) * 0.5  # tune as needed
        vy = np.sign(dy) * 0.5
        twist = Twist()
        # Many differential-drive robots take a linear x and angular z. Here we publish x/y which works for holonomic Herobot-like robots.
        # If your robot accepts only Twist.linear.x and angular.z, convert accordingly.
        twist.linear.x = vx
        twist.linear.y = vy
        self.cmd_pubs[i].publish(twist)

    def run_loop(self):
        steps = 800
        t = 0
        while not rospy.is_shutdown() and t < steps:
            if self.all_stopped:
                # send zero velocities
                for i in range(self.N_robots):
                    self.send_stop(i)
                self.rate.sleep()
                t += 1
                continue

            if self.target_found:
                # Move robots toward surround positions (cells)
                done = True
                for i in range(self.N_robots):
                    tx, ty = self.surround_positions[i]
                    cx, cy = int(self.robots[i,0]), int(self.robots[i,1])
                    if (cx, cy) != (tx, ty):
                        done = False
                        # step robots by one cell toward target cell
                        if cx < tx: cx += 1
                        elif cx > tx: cx -= 1
                        if cy < ty: cy += 1
                        elif cy > ty: cy -= 1
                        self.robots[i] = [cx, cy]
                        # command robot velocity toward that cell
                        self.move_towards_cell(i, (tx, ty))

                if done:
                    self.all_stopped = True
                    rospy.loginfo(f"All robots reached surround positions at target {self.targets[self.chosen_target]}")
            else:
                # keep searching
                for i in range(self.N_robots):
                    if self.indices[i] < len(self.paths[i]):
                        self.robots[i] = self.paths[i][self.indices[i]]
                        self.indices[i] += 1

                    x, y = int(self.robots[i,0]), int(self.robots[i,1])
                    if (x, y) in self.targets and not self.target_found:
                        tx, ty = x, y
                        if tx < self.margin or tx > self.nx-1-self.margin or ty < self.margin or ty > self.ny-1-self.margin:
                            rospy.loginfo(f"Robot {i} found target at {tx,ty} but too close to wall.")
                        else:
                            self.chosen_target = self.targets.index((tx,ty))
                            self.target_found = True
                            rospy.loginfo(f"Robot {i} found safe target at {tx,ty}. Computing surround positions.")
                            # compute surround positions
                            radius = 3
                            self.surround_positions = {}
                            for j in range(self.N_robots):
                                angle = 2*np.pi * j / self.N_robots
                                sx = tx + int(radius*np.cos(angle))
                                sy = ty + int(radius*np.sin(angle))
                                sx = min(max(0, sx), self.nx-1)
                                sy = min(max(0, sy), self.ny-1)
                                self.surround_positions[j] = (sx, sy)
                            break
                    elif self.grid[x, y] != 2:
                        self.grid[x, y] = 0

                    # publish a small move command to follow the discrete path
                    # publish toward next path cell if exists
                    if self.indices[i] < len(self.paths[i]):
                        next_cell = self.paths[i][self.indices[i]]
                        self.move_towards_cell(i, next_cell)

            self.rate.sleep()
            t += 1

if __name__ == "__main__":
    try:
        SwarmController()
    except rospy.ROSInterruptException:
        pass
