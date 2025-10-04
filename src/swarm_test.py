#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

N_ROBOTS = 4  # adjust depending on how many heroes you spawn
odom_data = {}


def odom_callback(msg, robot_id):
    pos = msg.pose.pose.position
    odom_data[robot_id] = (pos.x, pos.y)
    rospy.loginfo(f"[hero_{robot_id}] Odom position -> x: {pos.x:.2f}, y: {pos.y:.2f}")


def swarm_test():
    rospy.init_node("swarm_test")

    # Publishers
    pubs = {
        i: rospy.Publisher(f"/hero_{i}/cmd_vel", Twist, queue_size=1)
        for i in range(N_ROBOTS)
    }

    # Subscribers
    for i in range(N_ROBOTS):
        rospy.Subscriber(f"/hero_{i}/odom", Odometry, odom_callback, i)

    rospy.loginfo("Swarm test started... sending forward velocity commands.")
    rate = rospy.Rate(2)  # 2 Hz loop

    while not rospy.is_shutdown():
        cmd = Twist()
        cmd.linear.x = 0.2  # forward speed
        cmd.angular.z = 0.0

        for i in range(N_ROBOTS):
            pubs[i].publish(cmd)
            rospy.loginfo(
                f"[hero_{i}] Sent cmd_vel -> linear.x: {cmd.linear.x}, angular.z: {cmd.angular.z}"
            )

        rate.sleep()


if __name__ == "__main__":
    try:
        swarm_test()
    except rospy.ROSInterruptException:
        pass
