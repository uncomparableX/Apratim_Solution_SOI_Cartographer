import math
import threading
from typing import Optional, Tuple
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from .utilities import quaternion_to_yaw, pose_distance, normalize_angle, ros_now


class RobotState:
    def __init__(self, robot_name: str, params):
        self.robot_name = robot_name
        self.params = params
        self._lock = threading.Lock()
        self._pose: Optional[Tuple[float, float, float]] = None
        self._linear_velocity = 0.0
        self._angular_velocity = 0.0
        self._latest_scan: Optional[LaserScan] = None
        self._latest_odom: Optional[Odometry] = None
        self._goal: Optional[Tuple[float, float]] = None
        self._status = "idle"
        self._last_odom_time = 0.0
        self._last_scan_time = 0.0
        self._last_position = (0.0, 0.0)
        self._last_move_time = 0.0
        self._stuck_distance_threshold = 0.05
        self._stuck_timeout = float(params.stuck_timeout)
        self._initialized = False

    def _now(self) -> float:
        """Sim-time-aware seconds -- see utilities.ros_now / SolutionNode._now().
        stuck-detection and scan/odom-staleness timeouts here need this
        instead of wall-clock time for the same reason as everywhere else in
        this solution: they're judging simulated elapsed time."""
        return ros_now(self.params.node)

    def update_odometry(self, odom_msg: Odometry):
        with self._lock:
            self._latest_odom = odom_msg
            x = odom_msg.pose.pose.position.x
            y = odom_msg.pose.pose.position.y
            yaw = quaternion_to_yaw(odom_msg.pose.pose.orientation)
            self._pose = (x, y, yaw)
            self._linear_velocity = math.hypot(
                odom_msg.twist.twist.linear.x,
                odom_msg.twist.twist.linear.y
            )
            self._angular_velocity = odom_msg.twist.twist.angular.z
            self._last_odom_time = self._now()
            if not self._initialized:
                self._last_position = (x, y)
                self._last_move_time = self._now()
                self._initialized = True
            else:
                dist = pose_distance(
                    (x, y, yaw),
                    (self._last_position[0], self._last_position[1], yaw)
                )
                if dist > self._stuck_distance_threshold:
                    self._last_position = (x, y)
                    self._last_move_time = self._now()

    def update_scan(self, scan_msg: LaserScan):
        with self._lock:
            self._latest_scan = scan_msg
            self._last_scan_time = self._now()

    def set_goal(self, goal: Tuple[float, float]):
        with self._lock:
            self._goal = goal

    def clear_goal(self):
        with self._lock:
            self._goal = None

    def set_status(self, status: str):
        with self._lock:
            self._status = status

    def get_pose(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            return self._pose

    def get_velocity(self) -> Tuple[float, float]:
        with self._lock:
            return (self._linear_velocity, self._angular_velocity)

    def get_scan(self) -> Optional[LaserScan]:
        with self._lock:
            return self._latest_scan

    def get_goal(self) -> Optional[Tuple[float, float]]:
        with self._lock:
            return self._goal

    def get_status(self) -> str:
        with self._lock:
            return self._status

    def has_scan(self) -> bool:
        with self._lock:
            return self._latest_scan is not None

    def has_pose(self) -> bool:
        with self._lock:
            return self._pose is not None

    def is_stuck(self) -> bool:
        with self._lock:
            if self._goal is None or self._pose is None:
                return False
            elapsed = self._now() - self._last_move_time
            return elapsed > self._stuck_timeout

    def distance_to_goal(self) -> float:
        with self._lock:
            if self._pose is None or self._goal is None:
                return float('inf')
            return math.hypot(self._goal[0] - self._pose[0], self._goal[1] - self._pose[1])

    def heading_to_goal(self) -> float:
        with self._lock:
            if self._pose is None or self._goal is None:
                return 0.0
            dx = self._goal[0] - self._pose[0]
            dy = self._goal[1] - self._pose[1]
            return normalize_angle(math.atan2(dy, dx) - self._pose[2])

    def time_since_scan(self) -> float:
        with self._lock:
            return self._now() - self._last_scan_time

    def time_since_odom(self) -> float:
        with self._lock:
            return self._now() - self._last_odom_time

    def ready(self) -> bool:
        with self._lock:
            return self._pose is not None and self._latest_scan is not None

    def reset(self):
        with self._lock:
            self._pose = None
            self._linear_velocity = 0.0
            self._angular_velocity = 0.0
            self._latest_scan = None
            self._latest_odom = None
            self._goal = None
            self._status = "idle"
            self._last_odom_time = 0.0
            self._last_scan_time = 0.0
            self._last_position = (0.0, 0.0)
            self._last_move_time = 0.0
            self._initialized = False