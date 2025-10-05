#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import ColorRGBA
import math, time, threading


class SwarmManager:
    def __init__(self):
        rospy.init_node("swarm_manager")

        # Parameters
        self.robot_names = rospy.get_param(
            "~robots", ["hero_0", "hero_1", "hero_2", "hero_3"]
        )
        self.timeout = rospy.get_param("~timeout", 5.0)
        self.radius = rospy.get_param("~formation_radius", 2.0)
        self.center = (0.0, 0.0)
        self.lock = threading.Lock()

        self.last_odom = {r: time.time() for r in self.robot_names}
        self.positions = {r: (0, 0) for r in self.robot_names}
        self.active_robots = list(self.robot_names)

        self.cmd_pubs = {
            r: rospy.Publisher(
                f"/{r}/velocity_controller/cmd_vel", Twist, queue_size=10
            )
            for r in self.robot_names
        }
        self.led_pubs = {
            r: rospy.Publisher(f"/{r}/led", ColorRGBA, queue_size=10)
            for r in self.robot_names
        }

        for r in self.robot_names:
            rospy.Subscriber(f"/{r}/odom", Odometry, self.odom_cb, callback_args=r)

        self.rate = rospy.Rate(2)
        rospy.loginfo("✅ Swarm Manager started with robots: %s", self.robot_names)

    def odom_cb(self, msg, robot):
        with self.lock:
            self.last_odom[robot] = time.time()
            self.positions[robot] = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def compute_formation(self):
        n = len(self.active_robots)
        formation = {}
        if n == 0:
            return formation
        for i, r in enumerate(self.active_robots):
            angle = 2 * math.pi * i / n
            fx = self.center[0] + self.radius * math.cos(angle)
            fy = self.center[1] + self.radius * math.sin(angle)
            formation[r] = (fx, fy)
        return formation

    def move_toward(self, robot, target):
        if robot not in self.positions:
            return
        x, y = self.positions[robot]
        tx, ty = target
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)

        cmd = Twist()
        if dist > 0.1:
            cmd.linear.x = 0.3
            cmd.angular.z = 2.0 * math.atan2(dy, dx)  # simple steering
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pubs[robot].publish(cmd)

    def set_led(self, robot, r, g, b):
        msg = ColorRGBA(r, g, b, 1.0)
        self.led_pubs[robot].publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            with self.lock:
                now = time.time()
                alive = [
                    r
                    for r in self.robot_names
                    if now - self.last_odom[r] < self.timeout
                ]
                if set(alive) != set(self.active_robots):
                    rospy.logwarn("⚠️ Swarm reconfiguring: Active robots = %s", alive)
                    self.active_robots = alive
                    # Flash LEDs yellow while reconfiguring
                    for r in alive:
                        self.set_led(r, 1.0, 1.0, 0.0)

                formation = self.compute_formation()

                if len(alive) == 1:
                    self.set_led(alive[0], 1.0, 0.0, 0.0)
                elif len(alive) == 0:
                    pass
                else:
                    for r in alive:
                        self.move_toward(r, formation[r])
                        self.set_led(r, 0.0, 0.0, 1.0)  # blue = stable

            self.rate.sleep()


if __name__ == "__main__":
    manager = SwarmManager()
    manager.run()
