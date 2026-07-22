
import math
import threading
from typing import List, Tuple, Optional, Dict, Any
from .utilities import euclidean_distance, normalize_angle

try:
    from .algorithms.hungarian import HungarianAlgorithm
    _HAS_HUNGARIAN = True
except Exception:
    _HAS_HUNGARIAN = False


class Coordinator:
    def __init__(self, params):
        self.params = params
        self._lock = threading.Lock()
        self._robots: Dict[str, Dict[str, Any]] = {}
        self._assignments: Dict[str, Optional[Tuple[float, float]]] = {}
        self._heading_weight = 0.3
        self._busy_penalty = 5.0

    def register_robot(self, robot_name: str):
        with self._lock:
            self._robots[robot_name] = {
                'pose': None,
                'status': 'idle',
                'goal': None,
            }
            self._assignments[robot_name] = None

    def update_robot_pose(self, robot_name: str, pose: Tuple[float, float, float]):
        with self._lock:
            if robot_name not in self._robots:
                self.register_robot(robot_name)
            self._robots[robot_name]['pose'] = pose

    def update_robot_status(self, robot_name: str, status: str):
        with self._lock:
            if robot_name not in self._robots:
                self.register_robot(robot_name)
            self._robots[robot_name]['status'] = status

    def update_robot_goal(self, robot_name: str, goal: Optional[Tuple[float, float]]):
        with self._lock:
            if robot_name not in self._robots:
                self.register_robot(robot_name)
            self._robots[robot_name]['goal'] = goal
            self._assignments[robot_name] = goal

    def assign_frontiers(self, frontier_clusters: List[Tuple[float, float]], *robot_states) -> List[Optional[Tuple[float, float]]]:
        with self._lock:
            robots = []
            for rs in robot_states:
                name = getattr(rs, 'name', None)
                if name is None:
                    name = getattr(rs, 'robot_name', None)
                pose = getattr(rs, 'get_pose', lambda: None)()
                robots.append((name, pose))

            valid_robots = [(i, name, pose) for i, (name, pose) in enumerate(robots) if pose is not None]
            if not valid_robots or not frontier_clusters:
                return [None] * len(robots)

            n_robots = len(valid_robots)
            n_frontiers = len(frontier_clusters)

            def compute_cost(pose, fx, fy, name):
                dist = euclidean_distance((pose[0], pose[1]), (fx, fy))
                heading = math.atan2(fy - pose[1], fx - pose[0])
                heading_err = abs(normalize_angle(heading - pose[2]))
                heading_penalty = heading_err * self._heading_weight
                busy = self._busy_penalty if self._assignments.get(name) is not None else 0.0
                return dist + heading_penalty + busy

            cost_matrix = []
            for _, name, pose in valid_robots:
                row = [compute_cost(pose, fx, fy, name) for fx, fy in frontier_clusters]
                cost_matrix.append(row)

            assigned_indices = set()
            assignments = [None] * len(robots)

            if _HAS_HUNGARIAN and n_robots <= n_frontiers:
                try:
                    ha = HungarianAlgorithm()
                    if n_robots < n_frontiers:
                        padded = [row[:] for row in cost_matrix]
                        for row in padded:
                            while len(row) < n_frontiers:
                                row.append(float('inf'))
                        result = ha.solve(padded)
                    else:
                        result = ha.solve(cost_matrix)
                    for r_idx, f_idx in enumerate(result):
                        if f_idx >= 0 and f_idx < n_frontiers and f_idx not in assigned_indices:
                            robot_idx = valid_robots[r_idx][0]
                            assignments[robot_idx] = frontier_clusters[f_idx]
                            assigned_indices.add(f_idx)
                except Exception:
                    pass

            for r_idx, name, pose in valid_robots:
                if assignments[r_idx] is not None:
                    continue
                best_cost = float('inf')
                best_f = None
                for f_idx, (fx, fy) in enumerate(frontier_clusters):
                    if f_idx in assigned_indices:
                        continue
                    cost = compute_cost(pose, fx, fy, name)
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f_idx
                if best_f is not None:
                    assignments[r_idx] = frontier_clusters[best_f]
                    assigned_indices.add(best_f)

            return assignments

    def should_swap_goals(self, robot_a, robot_b, goal_a, goal_b) -> bool:
        with self._lock:
            pose_a = getattr(robot_a, 'get_pose', lambda: None)()
            pose_b = getattr(robot_b, 'get_pose', lambda: None)()
            if pose_a is None or pose_b is None or goal_a is None or goal_b is None:
                return False

            def cost(pose, goal):
                if goal is None:
                    return float('inf')
                dist = euclidean_distance((pose[0], pose[1]), (goal[0], goal[1]))
                heading = math.atan2(goal[1] - pose[1], goal[0] - pose[0])
                heading_err = abs(normalize_angle(heading - pose[2]))
                return dist + heading_err * self._heading_weight

            current_cost = cost(pose_a, goal_a) + cost(pose_b, goal_b)
            swapped_cost = cost(pose_a, goal_b) + cost(pose_b, goal_a)
            return swapped_cost < current_cost * 0.95

    def current_assignments(self) -> Dict[str, Optional[Tuple[float, float]]]:
        with self._lock:
            return dict(self._assignments)

    def robot_status(self, robot_name: str) -> Optional[str]:
        with self._lock:
            return self._robots.get(robot_name, {}).get('status')

    def robot_goal(self, robot_name: str) -> Optional[Tuple[float, float]]:
        with self._lock:
            return self._robots.get(robot_name, {}).get('goal')

    def robot_pose(self, robot_name: str) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            return self._robots.get(robot_name, {}).get('pose')

    def robot_names(self) -> List[str]:
        with self._lock:
            return list(self._robots.keys())

    def robot_count(self) -> int:
        with self._lock:
            return len(self._robots)

    def remove_assignment(self, robot_name: str):
        with self._lock:
            if robot_name in self._assignments:
                self._assignments[robot_name] = None
            if robot_name in self._robots:
                self._robots[robot_name]['goal'] = None

    def clear_assignments(self):
        with self._lock:
            for name in self._assignments:
                self._assignments[name] = None
                if name in self._robots:
                    self._robots[name]['goal'] = None

    def reset(self):
        with self._lock:
            self._robots.clear()
            self._assignments.clear()