#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import tf
import math

N_ROBOTS = 3  # adjust depending on swarm size
leader_id = 0
odom_data = {}


def odom_callback(msg, robot_id):
    odom_data[robot_id] = msg


def swarm_controller():
    rospy.init_node("swarm_controller")

    # Publishers for each robot
    pubs = {
        i: rospy.Publisher(
            f"/hero_{i}/velocity_controller/cmd_vel", Twist, queue_size=1
        )
        for i in range(N_ROBOTS)
    }

    # Subscribers
    for i in range(N_ROBOTS):
        rospy.Subscriber(f"/hero_{i}/odom", Odometry, odom_callback, i)

    rate = rospy.Rate(10)  # 10 Hz control loop

    while not rospy.is_shutdown():
        if len(odom_data) < N_ROBOTS:
            rate.sleep()
            continue

        # Leader motion (simple forward motion)
        leader_twist = Twist()
        leader_twist.linear.x = 0.2  # constant forward speed
        pubs[leader_id].publish(leader_twist)

        # Get leader pose
        leader_pose = odom_data[leader_id].pose.pose.position

        # Followers: try to follow leader
        for i in range(N_ROBOTS):
            if i == leader_id:
                continue

            pose = odom_data[i].pose.pose.position
            dx = leader_pose.x - pose.x
            dy = leader_pose.y - pose.y
            dist = math.sqrt(dx * dx + dy * dy)

            # Simple proportional controller
            cmd = Twist()
            cmd.linear.x = 0.2 * dist
            cmd.angular.z = 1.0 * math.atan2(dy, dx)
            pubs[i].publish(cmd)

        rate.sleep()


if __name__ == "__main__":
    try:
        swarm_controller()
    except rospy.ROSInterruptException:
        pass
