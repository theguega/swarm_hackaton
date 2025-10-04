#!/usr/bin/env python3
import rospy
from std_msgs.msg import String


def hello_world():
    rospy.init_node("hello_world_node")
    pub = rospy.Publisher("/hello_topic", String, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    rospy.loginfo("Hello world node started!")

    while not rospy.is_shutdown():
        msg = "Hello from swarm package!"
        rospy.loginfo(f"Publishing: {msg}")
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    try:
        hello_world()
    except rospy.ROSInterruptException:
        pass
