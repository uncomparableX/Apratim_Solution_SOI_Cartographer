import math
import time
import numpy as np
from typing import List, Tuple, Optional
from nav_msgs.msg import OccupancyGrid
from .utilities import (
    world_to_grid, grid_to_world, euclidean_distance, compute_path_length,
)
from .algorithms.astar import AStar


class Planner:
    def __init__(self, params):
        self.params = params
        self.resolution = float(params.map_resolution)
        self.origin_x = float(params.map_origin_x)
        self.origin_y = float(params.map_origin_y)
        self.robot_radius = float(params.robot_radius)
        self.safety_margin = float(params.safety_margin)
        self.inflation_radius = self.robot_radius + self.safety_margin
        self.inflation_cells = max(1, int(math.ceil(self.inflation_radius / self.resolution)))
        self.max_path_length = float(params.max_path_length)
        self.unknown_penalty = 10.0
        self._cached_path: List[Tuple[float, float]] = []
        self._cached_grid: Optional[np.ndarray] = None
        self._cached_inflated: Optional[np.ndarray] = None
        self._astar = AStar()
        self.last_rejection_reason: Optional[str] = None

    def _logger(self):
        node = getattr(self.params, 'node', None)
        return node.get_logger() if node is not None else None

    def _reject(self, reason: str, level: str = 'info'):
        self.last_rejection_reason = reason

    def _extract_grid(self, occupancy_grid: OccupancyGrid) -> np.ndarray:
        w = occupancy_grid.info.width
        h = occupancy_grid.info.height
        data = np.array(occupancy_grid.data, dtype=np.int8)
        return data.reshape((h, w))

    def _inflate_obstacles(self, grid: np.ndarray) -> np.ndarray:
        if (
            self._cached_grid is not None
            and self._cached_inflated is not None
            and np.array_equal(grid, self._cached_grid)
        ):
            return self._cached_inflated
        h, w = grid.shape
        inflated = grid.copy()
        occupied_mask = grid > 50
        if not np.any(occupied_mask):
            self._cached_grid = grid.copy()
            self._cached_inflated = inflated
            return inflated
        occ_y, occ_x = np.where(occupied_mask)
        for ox, oy in zip(occ_x, occ_y):
            x_min = max(0, ox - self.inflation_cells)
            x_max = min(w, ox + self.inflation_cells + 1)
            y_min = max(0, oy - self.inflation_cells)
            y_max = min(h, oy + self.inflation_cells + 1)
            for dy in range(y_min, y_max):
                for dx in range(x_min, x_max):
                    if (dx - ox) ** 2 + (dy - oy) ** 2 <= self.inflation_cells ** 2:
                        if inflated[dy, dx] <= 50:
                            inflated[dy, dx] = 49
        self._cached_grid = grid.copy()
        self._cached_inflated = inflated
        return inflated

    def _is_valid_goal(self, gx: int, gy: int, inflated: np.ndarray) -> bool:
        h, w = inflated.shape
        if not (0 <= gx < w and 0 <= gy < h):
            return False
        if inflated[gy, gx] > 50:
            return False
        return True

    def _find_nearest_free(self, gx: int, gy: int, inflated: np.ndarray) -> Optional[Tuple[int, int]]:
        h, w = inflated.shape
        
        # FIX: ONLY snap to strictly FREE (0) cells. Prevents snapping into unreachable UNKNOWN (-1) space.
        for r in range(1, 40):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < w and 0 <= ny < h and inflated[ny, nx] == 0:
                        return nx, ny
                        
        # Fallback to any passable cell (<= 50) if absolutely no 0 is found
        for r in range(1, 40):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < w and 0 <= ny < h and inflated[ny, nx] <= 50:
                        return nx, ny
        return None

    def _grid_cost(self, gx: int, gy: int, inflated: np.ndarray) -> float:
        h, w = inflated.shape
        if not (0 <= gx < w and 0 <= gy < h):
            return float('inf')
        val = inflated[gy, gx]
        if val > 50:
            return float('inf')
        if val < 0:
            return self.unknown_penalty
        return 1.0 + (val / 100.0) * 5.0

    def _line_of_sight(self, p1: Tuple[int, int], p2: Tuple[int, int], inflated: np.ndarray) -> bool:
        x0, y0 = p1
        x1, y1 = p2
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if not (0 <= x < inflated.shape[1] and 0 <= y < inflated.shape[0]):
                return False
            if inflated[y, x] > 50:
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _smooth_path(self, path: List[Tuple[int, int]], inflated: np.ndarray) -> List[Tuple[int, int]]:
        if len(path) < 3:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._line_of_sight(path[i], path[j], inflated):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def plan(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float],
        occupancy_grid: OccupancyGrid
    ) -> List[Tuple[float, float]]:
        self.last_rejection_reason = None
        grid = self._extract_grid(occupancy_grid)
        inflated = self._inflate_obstacles(grid)
        sx, sy = world_to_grid(start_pose[0], start_pose[1], self.origin_x, self.origin_y, self.resolution)
        gx, gy = world_to_grid(goal_pose[0], goal_pose[1], self.origin_x, self.origin_y, self.resolution)
        h, w = inflated.shape
        goal_in_bounds = (0 <= gx < w and 0 <= gy < h)
        start_in_bounds = (0 <= sx < w and 0 <= sy < h)
        
        if not start_in_bounds:
            self._cached_path = []
            self._reject(f'start cell ({sx},{sy}) is outside the grid bounds')
            return []
            
        # FIX: Explicitly clear the robot's immediate base footprint so A* can always start
        for dy in range(max(0, sy - self.inflation_cells), min(h, sy + self.inflation_cells + 1)):
            for dx in range(max(0, sx - self.inflation_cells), min(w, sx + self.inflation_cells + 1)):
                if math.hypot(dx - sx, dy - sy) <= self.inflation_cells:
                    if grid[dy, dx] <= 50: 
                        inflated[dy, dx] = grid[dy, dx]
        inflated[sy, sx] = 0

        if not self._is_valid_goal(gx, gy, inflated):
            nearest = self._find_nearest_free(gx, gy, inflated)
            if nearest is None:
                self._cached_path = []
                self._reject(f'goal cell ({gx},{gy}) is inflated-occupied and no free cell exists')
                return []
            gx, gy = nearest

        def neighbors(node):
            x, y = node
            nbs = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        cost = self._grid_cost(nx, ny, inflated)
                        if cost < float('inf'):
                            nbs.append(((nx, ny), math.hypot(dx, dy) * cost))
            return nbs

        def heuristic(node):
            return math.hypot(node[0] - gx, node[1] - gy)

        path_grid = self._astar.search((sx, sy), (gx, gy), neighbors, heuristic)
        if path_grid is None or len(path_grid) == 0:
            self._cached_path = []
            self._reject(f'A* found no route from ({sx},{sy}) to ({gx},{gy}) through the inflated grid')
            return []
            
        smoothed_grid = self._smooth_path(path_grid, inflated)
        path_world = []
        for gx_pt, gy_pt in smoothed_grid:
            wx, wy = grid_to_world(gx_pt, gy_pt, self.origin_x, self.origin_y, self.resolution)
            path_world.append((wx, wy))
            
        path_length = compute_path_length(path_world)
        if path_length > self.max_path_length:
            self._cached_path = []
            self._reject(f'planned path length {path_length:.1f}m exceeds max_path_length {self.max_path_length:.1f}m')
            return []
            
        self._cached_path = path_world
        return path_world

    def is_path_valid(
        self,
        path: List[Tuple[float, float]],
        occupancy_grid: OccupancyGrid,
        start_pose: Optional[Tuple[float, float, float]] = None,
    ) -> bool:
        if len(path) == 0:
            return False
        grid = self._extract_grid(occupancy_grid)
        inflated = self._inflate_obstacles(grid)
        h, w = inflated.shape
        for wx, wy in path:
            if start_pose is not None:
                if math.hypot(wx - start_pose[0], wy - start_pose[1]) <= self.robot_radius:
                    continue
            gx, gy = world_to_grid(wx, wy, self.origin_x, self.origin_y, self.resolution)
            if not (0 <= gx < w and 0 <= gy < h):
                return False
            if inflated[gy, gx] > 50:
                return False
        return True

    def smooth_path(self, path: List[Tuple[float, float]], occupancy_grid: OccupancyGrid) -> List[Tuple[float, float]]:
        if len(path) < 3:
            return path
        grid = self._extract_grid(occupancy_grid)
        inflated = self._inflate_obstacles(grid)
        grid_path = []
        for wx, wy in path:
            gx, gy = world_to_grid(wx, wy, self.origin_x, self.origin_y, self.resolution)
            grid_path.append((gx, gy))
        smoothed = self._smooth_path(grid_path, inflated)
        return [grid_to_world(gx, gy, self.origin_x, self.origin_y, self.resolution) for gx, gy in smoothed]

    def replan(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float],
        occupancy_grid: OccupancyGrid
    ) -> List[Tuple[float, float]]:
        return self.plan(start_pose, goal_pose, occupancy_grid)

    def current_path(self) -> List[Tuple[float, float]]:
        return list(self._cached_path)

    def valid_path(self) -> bool:
        return len(self._cached_path) > 0

    def reset(self):
        self._cached_path = []
        self._cached_grid = None
        self._cached_inflated = None