#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry

N_ROBOTS = 4  # adjust to how many heroes are spawned
odom_data = {}


def odom_callback(msg, robot_id):
    pos = msg.pose.pose.position
    odom_data[robot_id] = (pos.x, pos.y)


def odom_debug():
    rospy.init_node("odom_debug")

    # Subscribers
    for i in range(N_ROBOTS):
        rospy.Subscriber(f"/hero_{i}/odom", Odometry, odom_callback, i)

    rate = rospy.Rate(2)  # 2 Hz
    rospy.loginfo("Listening to odometry topics...")

    while not rospy.is_shutdown():
        for i in range(N_ROBOTS):
            if i in odom_data:
                x, y = odom_data[i]
                rospy.loginfo(f"[hero_{i}] Odom position -> x: {x:.2f}, y: {y:.2f}")
            else:
                rospy.logwarn(f"[hero_{i}] No odom data received yet.")
        rate.sleep()


if __name__ == "__main__":
    try:
        odom_debug()
    except rospy.ROSInterruptException:
        pass
