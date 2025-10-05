#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist


def go_straight(robot_name="hero_2", speed=0.1, duration=5.0):
    """
    Makes the robot go straight forward slowly for a few seconds.
    """
    pub = rospy.Publisher(
        f"/{robot_name}/velocity_controller/cmd_vel", Twist, queue_size=10
    )
    rospy.init_node(f"go_straight_{robot_name}", anonymous=True)
    rate = rospy.Rate(10)

    rospy.loginfo(f"[{robot_name}] Going straight at {speed} m/s for {duration}s")

    cmd = Twist()
    cmd.linear.x = speed
    cmd.angular.z = 0.0

    start_time = rospy.Time.now()
    while not rospy.is_shutdown():
        elapsed = (rospy.Time.now() - start_time).to_sec()
        if elapsed > duration:
            break
        pub.publish(cmd)
        rate.sleep()

    # stop the robot
    cmd.linear.x = 0.0
    pub.publish(cmd)
    rospy.loginfo(f"[{robot_name}] Stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make robot go straight slowly.")
    parser.add_argument(
        "--robot", type=str, default="hero_0", help="Robot name (e.g., hero_0)"
    )
    parser.add_argument("--speed", type=float, default=0.1, help="Forward speed (m/s)")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration (s)")
    args = parser.parse_args()

    go_straight(robot_name=args.robot, speed=args.speed, duration=args.duration)
