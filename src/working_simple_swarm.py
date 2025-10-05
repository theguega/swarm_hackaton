#!/usr/bin/env python3
import rospy
import math
import random
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

NUM_ROBOTS = 10  # Scalable
UPDATE_RATE = 30
REPULSION_DISTANCE = 0.8  # Distance to start repulsing from other robots
OBSTACLE_DISTANCE = 0.5  # Distance to walls
WALL_COLLISION_DISTANCE = 0.4  # Distance to consider wall collision
RANDOM_CHANGE_TIME = 7.0  # Seconds to change random direction
LINEAR_SPEED = 0.2
ANGULAR_SPEED = 0.2


class SimpleSwarm:
    def __init__(self):
        rospy.init_node("simple_swarm")
        self.rate = rospy.Rate(UPDATE_RATE)
        self.positions = [(0.0, 0.0, 0.0)] * NUM_ROBOTS
        self.laser_data = [None] * NUM_ROBOTS
        self.random_angles = [random.uniform(0, 2 * math.pi) for _ in range(NUM_ROBOTS)]
        self.last_change = [rospy.Time.now()] * NUM_ROBOTS
        self.wall_collision = [False] * NUM_ROBOTS
        self.collision_start_time = [rospy.Time.now()] * NUM_ROBOTS
        self.escape_direction = [0.0] * NUM_ROBOTS
        self.debug_counter = 0

        self.odom_subs = []
        self.laser_subs = []
        self.cmd_pubs = []

        for i in range(NUM_ROBOTS):
            self.odom_subs.append(
                rospy.Subscriber(f"/hero_{i}/odom", Odometry, self.odom_callback, i)
            )
            self.laser_subs.append(
                rospy.Subscriber(f"/hero_{i}/laser", LaserScan, self.laser_callback, i)
            )
            self.cmd_pubs.append(
                rospy.Publisher(f"/hero_{i}/cmd_vel", Twist, queue_size=10)
            )

    def odom_callback(self, msg, robot_id):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.positions[robot_id] = (x, y, theta)
        if robot_id == 0 and self.debug_counter % 50 == 0:  # Print every 50 loops
            print(f"Robot {robot_id} odom: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")

    def laser_callback(self, msg, robot_id):
        self.laser_data[robot_id] = msg
        min_range = min(msg.ranges) if msg.ranges else float("inf")

        # Check for wall collision
        was_colliding = self.wall_collision[robot_id]
        self.wall_collision[robot_id] = min_range < WALL_COLLISION_DISTANCE

        # Set escape direction when collision starts
        if self.wall_collision[robot_id] and not was_colliding:
            self.collision_start_time[robot_id] = rospy.Time.now()
            # Find direction to turn (pick left or right based on laser data)
            ranges = msg.ranges
            mid = len(ranges) // 2
            left_min = min(ranges[:mid]) if ranges[:mid] else float("inf")
            right_min = min(ranges[mid:]) if ranges[mid:] else float("inf")
            # Turn away from the side with closer obstacles
            self.escape_direction[robot_id] = 1.0 if left_min < right_min else -1.0

        if min_range < OBSTACLE_DISTANCE:
            print(
                f"Robot {robot_id} laser: min_range={min_range:.2f}, wall_collision={self.wall_collision[robot_id]}"
            )

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def compute_repulsion(self, robot_id):
        x, y, theta = self.positions[robot_id]
        rep_x, rep_y = 0, 0

        # Repulsion from other robots
        for j in range(NUM_ROBOTS):
            if j == robot_id:
                continue
            ox, oy, _ = self.positions[j]
            dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if dist < REPULSION_DISTANCE and dist > 0:
                rep_x += (x - ox) / dist**2
                rep_y += (y - oy) / dist**2

        # Strong repulsion from walls when collision detected
        if self.laser_data[robot_id] and self.wall_collision[robot_id]:
            msg = self.laser_data[robot_id]
            # Find the closest obstacle direction
            min_range = float("inf")
            closest_angle = 0

            for i, r in enumerate(msg.ranges):
                if r < min_range and r > 0.1:
                    min_range = r
                    closest_angle = msg.angle_min + i * msg.angle_increment + theta

            if min_range < WALL_COLLISION_DISTANCE:
                # Go in opposite direction with strong force
                opposite_x = -math.cos(closest_angle) * 5.0  # Strong opposite force
                opposite_y = -math.sin(closest_angle) * 5.0
                rep_x += opposite_x
                rep_y += opposite_y
                print(f"Robot {robot_id} wall collision! Going opposite direction")

        # Regular repulsion from obstacles (walls) when not in collision
        elif self.laser_data[robot_id]:
            msg = self.laser_data[robot_id]
            for i, r in enumerate(msg.ranges):
                if r < OBSTACLE_DISTANCE and r > 0.1:
                    angle = msg.angle_min + i * msg.angle_increment + theta
                    rep_x += math.cos(angle) / r**2
                    rep_y += math.sin(angle) / r**2

        return rep_x, rep_y

    def run(self):
        while not rospy.is_shutdown():
            self.debug_counter += 1
            for i in range(NUM_ROBOTS):
                now = rospy.Time.now()
                if (now - self.last_change[i]).to_sec() > RANDOM_CHANGE_TIME:
                    self.random_angles[i] = random.uniform(0, 2 * math.pi)
                    self.last_change[i] = now

                # Commands
                twist = Twist()

                # If wall collision, override normal behavior
                if self.wall_collision[i]:
                    collision_duration = (now - self.collision_start_time[i]).to_sec()

                    if collision_duration < 1.0:  # First second: aggressive turn
                        twist.linear.x = 0.0  # Stop moving forward
                        twist.angular.z = (
                            self.escape_direction[i] * ANGULAR_SPEED * 3.0
                        )  # Aggressive turn
                    elif collision_duration < 2.0:  # Next second: reverse while turning
                        twist.linear.x = -LINEAR_SPEED * 0.8  # Reverse faster
                        twist.angular.z = (
                            self.escape_direction[i] * ANGULAR_SPEED * 2.0
                        )  # Continue turning
                    else:  # After 2 seconds: forward in escape direction
                        twist.linear.x = LINEAR_SPEED * 0.8  # Move forward
                        twist.angular.z = (
                            self.escape_direction[i] * ANGULAR_SPEED * 1.5
                        )  # Keep turning

                    # Force new random direction periodically
                    if collision_duration > 0.5:
                        self.random_angles[i] = random.uniform(0, 2 * math.pi)
                        self.last_change[i] = now

                    print(
                        f"Robot {i} collision escape (t={collision_duration:.1f}s): linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}, dir={self.escape_direction[i]}"
                    )
                else:
                    # Normal behavior: desired direction with random
                    des_x = math.cos(self.random_angles[i])
                    des_y = math.sin(self.random_angles[i])

                    # Add repulsion
                    rep_x, rep_y = self.compute_repulsion(i)
                    des_x += rep_x
                    des_y += rep_y

                    # Normalize
                    mag = math.sqrt(des_x**2 + des_y**2)
                    if mag > 0:
                        des_x /= mag
                        des_y /= mag

                    # Current theta
                    _, _, theta = self.positions[i]

                    # Desired theta
                    des_theta = math.atan2(des_y, des_x)

                    # Angular error
                    error = des_theta - theta
                    error = math.atan2(math.sin(error), math.cos(error))

                    # Normal movement commands
                    twist.linear.x = LINEAR_SPEED
                    twist.angular.z = ANGULAR_SPEED * error

                self.cmd_pubs[i].publish(twist)
                if self.debug_counter % 10 == 0:  # Print every 10 loops
                    print(
                        f"Robot {i} cmd: linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}"
                    )

            self.rate.sleep()


if __name__ == "__main__":
    try:
        swarm = SimpleSwarm()
        swarm.run()
    except rospy.ROSInterruptException:
        pass
