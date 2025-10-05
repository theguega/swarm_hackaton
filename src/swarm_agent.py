#!/usr/bin/env python3
"""
swarm_agent.py

ROS1 node implementing sampled Velocity-Obstacle style collision avoidance
with golden-ratio disk sampling + penalty scoring + critically-damped smoothing.

Intended topics (per-robot namespace, default /hero_0):
  - subscribe: /<ns>/odom            (nav_msgs/Odometry)
  - subscribe (optional): /<ns>/laser (sensor_msgs/Range or LaserScan) -> not used by default
  - publish: /<ns>/velocity_controller/cmd_vel  (geometry_msgs/Twist)

Parameters (rosparam / private):
  robot_namespace (string) : e.g. "/hero_0"
  swarm_size (int)         : total robots in swarm (used to query other odoms)
  robot_radius (float)     : robot footprint radius (m)
  max_speed (float)        : max linear speed (m/s)
  samples (int)            : number of velocity samples
  evasion_strength (float) : weight for collision term in penalty
  dt_control (float)       : control loop dt (s)
  smoothing_k (float)      : spring constant for critically-damped smoothing
  smoothing_initial_v (float) : initial smoothing velocity (internal)
  use_tf (bool)            : whether to try reading other robots via TF (if false, uses odom topics)
  publish_topic (string)   : default "velocity_controller/cmd_vel"
  goal (list)              : [x,y] local goal in the robot's odom frame or global frame depending on usage
"""

import rospy
import math
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf
import threading
import time

# ---------------- math helpers ----------------


def norm2(v):
    return math.hypot(v[0], v[1])


def clamp_mag(v, max_mag):
    mag = norm2(v)
    if mag > max_mag:
        return (v[0] * max_mag / mag, v[1] * max_mag / mag)
    return v


# golden-ratio disk sampling: returns list of (vx, vy) points within disk radius r
def golden_disk_samples(radius, n):
    # uniform sampling on disk using golden angle
    samples = []
    phi = (1 + 5**0.5) / 2.0
    alpha = 2 * math.pi * (1 - 1 / phi)  # golden angle
    for i in range(n):
        r = math.sqrt(float(i + 1) / n) * radius
        theta = i * alpha
        samples.append((r * math.cos(theta), r * math.sin(theta)))
    return samples


# analytic time-to-collision between two disks with constant velocities.
# Solve for smallest t>=0 where | (p_rel + v_rel*t) | <= r_sum
# If never collides -> return math.inf
def time_to_collision(p_rel, v_rel, r_sum):
    # Solve |p + v t|^2 = r_sum^2
    px, py = p_rel
    vx, vy = v_rel
    a = vx * vx + vy * vy
    b = 2 * (px * vx + py * vy)
    c = px * px + py * py - r_sum * r_sum
    if a == 0.0:
        # relative velocity zero -> either already colliding or static
        if c <= 0:
            return 0.0
        else:
            return math.inf
    disc = b * b - 4 * a * c
    if disc < 0:
        return math.inf
    sqrt_d = math.sqrt(disc)
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)
    # collision interval is between t1 and t2 (t1 <= t2)
    if t2 < 0:
        return math.inf
    if t1 < 0 <= t2:
        return 0.0
    return t1 if t1 >= 0 else math.inf


# critically damped oscillator smoothing (2 state)
class CriticallyDampedSmoother:
    def __init__(self, k, dt, initial_val=(0.0, 0.0)):
        self.k = float(k)
        self.dt = float(dt)
        self.y = np.array(initial_val, dtype=float)
        self.v = np.zeros(2, dtype=float)

    # x is target (tuple/list/np)
    def step(self, x):
        x = np.array(x, dtype=float)
        # y_t = y_{t-1} + v_{t-1} * dt
        self.y = self.y + self.v * self.dt
        # v_t = v_{t-1} + (k*(x - y) - 2*sqrt(k)*v_{t-1}) * dt
        damp = 2.0 * math.sqrt(self.k)
        self.v = self.v + (self.k * (x - self.y) - damp * self.v) * self.dt
        return tuple(self.y)


# ---------------- ROS node ----------------


class SwarmAgentNode(object):
    def __init__(self):
        rospy.init_node("swarm_agent", anonymous=True)

        # parameters
        ns = rospy.get_param("~robot_namespace", rospy.get_namespace().rstrip("/"))
        self.ns = ns if ns.startswith("/") else "/" + ns
        self.swarm_size = rospy.get_param("~swarm_size", 4)
        self.robot_radius = rospy.get_param("~robot_radius", 0.12)
        self.max_speed = rospy.get_param("~max_speed", 0.4)
        self.samples = rospy.get_param("~samples", 500)
        self.evasion_strength = rospy.get_param("~evasion_strength", 1.0)
        self.dt = rospy.get_param("~dt_control", 0.1)
        self.smoothing_k = rospy.get_param("~smoothing_k", 10.0)
        self.publish_topic = rospy.get_param(
            "~publish_topic", "velocity_controller/cmd_vel"
        )
        goal = rospy.get_param("~goal", None)  # optional [x,y]
        self.goal = tuple(goal) if goal is not None else None
        self.use_tf = rospy.get_param("~use_tf", True)
        self.other_odom_topics = rospy.get_param(
            "~other_odom_topics", None
        )  # optional list

        # internal
        self.odom_lock = threading.Lock()
        self.odom = None  # our latest odom
        self.others = {}  # namespace -> {'pos':(x,y), 'vel':(vx,vy), 'time':t}

        # smoothing: we smooth commanded linear velocities (vx, vy)
        self.smoother = CriticallyDampedSmoother(
            self.smoothing_k, self.dt, initial_val=(0.0, 0.0)
        )

        # ROS interfaces
        self.cmd_pub = rospy.Publisher(
            self.ns + "/" + self.publish_topic, Twist, queue_size=1
        )
        rospy.Subscriber(self.ns + "/odom", Odometry, self.odom_cb, queue_size=1)

        self.tf_listener = tf.TransformListener() if self.use_tf else None

        # optional: subscribe to each other's odom topics for more accurate velocities
        if self.other_odom_topics is None:
            # default: assume hero_0..hero_{N-1}
            self.other_names = ["/hero_{}".format(i) for i in range(self.swarm_size)]
        else:
            self.other_names = self.other_odom_topics

        # subscribe to others' odoms (non blocking)
        for name in self.other_names:
            # skip ourselves
            if name.rstrip("/") == self.ns.rstrip("/"):
                continue
            topic = name + "/odom"
            rospy.Subscriber(
                topic, Odometry, self.other_odom_cb, callback_args=name, queue_size=1
            )

        rospy.loginfo(
            "Swarm agent initialized in namespace %s, swarm_size=%d",
            self.ns,
            self.swarm_size,
        )

        self.samples_cache = golden_disk_samples(self.max_speed, max(10, self.samples))

    def odom_cb(self, msg):
        with self.odom_lock:
            self.odom = msg

    def other_odom_cb(self, msg, ns):
        # store other robot pose and velocity in our map (in world frame)
        pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        # approximate planar velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        with self.odom_lock:
            self.others[ns] = {
                "pos": pos,
                "vel": (vx, vy),
                "time": rospy.Time.now().to_sec(),
            }

    def other_pose_from_tf(self, other_ns):
        # fallback method: use TF lookup from other_ns/odom to our odom frame (or directly to /map)
        try:
            target_frame = (
                other_ns.strip("/") + "/base_link"
                if False
                else other_ns.strip("/") + "/odom"
            )
            # We'll try to lookup transform from world (odom) to their odom
            # Simpler: try to read transform from their base_link to global 'odom'
            src = other_ns.strip("/") + "/base_link"
            self.tf_listener.waitForTransform(
                "odom", src, rospy.Time(0), rospy.Duration(0.1)
            )
            (trans, rot) = self.tf_listener.lookupTransform("odom", src, rospy.Time(0))
            vx = 0.0
            vy = 0.0
            return {
                "pos": (trans[0], trans[1]),
                "vel": (vx, vy),
                "time": rospy.Time.now().to_sec(),
            }
        except Exception as e:
            return None

    def compute_preferred_velocity(self, my_pos):
        # simple go-to-goal preferred velocity
        if self.goal is None:
            # default: stay put
            return (0.0, 0.0)
        dx = self.goal[0] - my_pos[0]
        dy = self.goal[1] - my_pos[1]
        vec = (dx, dy)
        vec = clamp_mag(vec, self.max_speed)
        return vec

    def find_best_sample(self, pos, vel):
        v_pref = self.compute_preferred_velocity(pos)
        best_sample = (0.0, 0.0)
        best_score = float("inf")

        for s in self.samples_cache:
            v_sample = s  # vx, vy in world frame
            # compute time to nearest collision with all others
            t_nearest = float("inf")
            for other_ns, info in list(self.others.items()):
                p_other = info["pos"]
                v_other = info["vel"]
                # relative pos = p_other - pos
                p_rel = (p_other[0] - pos[0], p_other[1] - pos[1])
                v_rel = (
                    v_other[0] - v_sample[0],
                    v_other[1] - v_sample[1],
                )  # relative velocity
                t_col = time_to_collision(p_rel, v_rel, self.robot_radius * 2.0)
                if t_col < t_nearest:
                    t_nearest = t_col
            # compute penalty
            dist_pref = math.hypot(v_pref[0] - v_sample[0], v_pref[1] - v_sample[1])
            # avoid division by zero
            tcol = t_nearest
            if tcol == 0.0:
                collision_penalty = 1e6
            elif tcol == float("inf"):
                collision_penalty = 0.0
            else:
                collision_penalty = self.evasion_strength / tcol
            score = dist_pref + collision_penalty
            if score < best_score:
                best_score = score
                best_sample = v_sample
        return best_sample, best_score

    def run(self):
        rate = rospy.Rate(1.0 / self.dt)
        while not rospy.is_shutdown():
            with self.odom_lock:
                od = self.odom
            if od is None:
                rate.sleep()
                continue
            # our current pose & vel (world frame via odom)
            pos = (od.pose.pose.position.x, od.pose.pose.position.y)
            vel = (od.twist.twist.linear.x, od.twist.twist.linear.y)

            # if TF mode and others empty, try TF
            if self.use_tf and len(self.others) == 0:
                for name in self.other_names:
                    if name.rstrip("/") == self.ns.rstrip("/"):
                        continue
                    info = self.other_pose_from_tf(name)
                    if info:
                        self.others[name] = info

            best_v, best_score = self.find_best_sample(pos, vel)

            # smoothing
            smooth_v = self.smoother.step(best_v)

            # publish Twist (map/same frame as odom)
            twist = Twist()
            twist.linear.x = smooth_v[0]
            twist.linear.y = smooth_v[1]
            # keep angular zero for holonomic; if non-holonomic convert later
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)

            rate.sleep()


if __name__ == "__main__":
    node = SwarmAgentNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
