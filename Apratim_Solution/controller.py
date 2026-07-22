import math
from typing import List, Tuple, Optional
from geometry_msgs.msg import Twist
from .utilities import normalize_angle, clamp_value, pose_distance

class Controller:
    def __init__(self, params, robot_name=None):
        self.params = params
        self.robot_name = robot_name
        self.max_linear_speed = float(params.max_linear_speed)
        self.max_angular_speed = float(params.max_angular_speed)
        self.goal_tolerance = float(params.goal_tolerance)
        self._path: List[Tuple[float, float]] = []
        self._goal: Optional[Tuple[float, float]] = None
        self._path_index = 0

    def set_path(self, path: List[Tuple[float, float]]):
        self._path = list(path)
        self._path_index = 0

    def set_goal(self, goal: Tuple[float, float]):
        self._goal = goal

    def clear_goal(self):
        self._goal = None
        self._path = []
        self._path_index = 0

    def goal_reached(self, robot_pose: Tuple[float, float, float]) -> bool:
        if self._goal is None:
            return True
        return pose_distance(robot_pose, (self._goal[0], self._goal[1], 0.0)) < self.goal_tolerance

    def _get_lookahead(self, robot_pose: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
        if not self._path:
            return self._goal
            
        x, y, yaw = robot_pose
        # FIX: Increased lookahead distance to 0.6m. 
        # This prevents the robot from violently twitching left/right trying to reach a point under its wheels.
        lookahead_distance = 0.6 
        
        best_pt = self._path[-1]
        for i in range(self._path_index, len(self._path)):
            dist = math.hypot(self._path[i][0] - x, self._path[i][1] - y)
            if dist >= lookahead_distance:
                self._path_index = i
                return self._path[i]
                
        return best_pt

    def compute_velocity(self, robot_pose: Tuple[float, float, float], lookahead_point=None, grid=None):
        x, y, yaw = robot_pose
        target = lookahead_point if lookahead_point else self._get_lookahead(robot_pose)
        
        if target is None:
            return self.stop()
            
        tx, ty = target[0], target[1]
        dx, dy = tx - x, ty - y
        distance = math.hypot(dx, dy)

        goal_dist = pose_distance(robot_pose, (self._goal[0], self._goal[1], 0.0)) if self._goal else distance
        if goal_dist < self.goal_tolerance:
            return self.stop()

        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(target_heading - yaw)

        # FIX: Strict spin-in-place if the goal is > 30 degrees to the side.
        # This guarantees it won't drive forward while facing a wall.
        if abs(heading_error) > 0.5:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = math.copysign(self.max_angular_speed * 0.8, heading_error)
            return cmd

        # Pure Pursuit geometry
        if distance > 0.001:
            curvature = 2.0 * math.sin(heading_error) / distance
        else:
            curvature = 0.0

        # Slow down smoothly as we approach the final goal
        target_speed = self.max_linear_speed
        if goal_dist < 1.0:
            target_speed = max(0.1, self.max_linear_speed * goal_dist)

        angular = clamp_value(curvature * target_speed, -self.max_angular_speed, self.max_angular_speed)

        cmd = Twist()
        cmd.linear.x = target_speed
        cmd.angular.z = angular
        return cmd

    def stop(self, reason: Optional[str] = None) -> Twist:
        return Twist()

    def current_path(self) -> List[Tuple[float, float]]:
        return list(self._path)

    def valid_path(self) -> bool:
        return len(self._path) > 0

    def reset(self):
        self.clear_goal()