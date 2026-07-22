
import math
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header
from .utilities import world_to_grid, grid_to_world


class OccupancyGridManager:
    def __init__(self, params):
        self.params = params
        self.resolution = float(params.map_resolution)
        self.width = int(params.map_width)
        self.height = int(params.map_height)
        self.origin_x = float(params.map_origin_x)
        self.origin_y = float(params.map_origin_y)
        self.robot_radius = float(params.robot_radius)
        self.safety_margin = float(params.safety_margin)
        self.inflation_radius = self.robot_radius + self.safety_margin
        self.inflation_cells = max(1, int(math.ceil(self.inflation_radius / self.resolution)))
        self._lock = threading.Lock()
        self._grids: Dict[str, np.ndarray] = {}
        self._global_grid: Optional[np.ndarray] = None
        self._global_ros: Optional[OccupancyGrid] = None

    def create_empty_grid(self) -> np.ndarray:
        return np.full((self.height, self.width), -1, dtype=np.int8)

    def from_ros(self, msg: OccupancyGrid) -> np.ndarray:
        if msg is None or not hasattr(msg, 'data') or msg.data is None:
            return self.create_empty_grid()
        h = msg.info.height
        w = msg.info.width
        return np.array(msg.data, dtype=np.int8).reshape((h, w))

    def to_ros(self, grid: np.ndarray) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.params.node.get_clock().now().to_msg()
        msg.header.frame_id = self.params.global_map_frame
        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width = int(grid.shape[1])
        msg.info.height = int(grid.shape[0])
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.ravel().tolist()
        return msg

    def inflate_obstacles(self, grid: np.ndarray) -> np.ndarray:
        if grid is None:
            return self.create_empty_grid()
        h, w = grid.shape
        inflated = grid.copy()
        occupied = grid > 50
        if not np.any(occupied):
            return inflated
        occ_y, occ_x = np.where(occupied)
        for ox, oy in zip(occ_x, occ_y):
            x_min = max(0, ox - self.inflation_cells)
            x_max = min(w, ox + self.inflation_cells + 1)
            y_min = max(0, oy - self.inflation_cells)
            y_max = min(h, oy + self.inflation_cells + 1)
            for dy in range(y_min, y_max):
                for dx in range(x_min, x_max):
                    if (dx - ox) ** 2 + (dy - oy) ** 2 <= self.inflation_cells ** 2:
                        inflated[dy, dx] = 100
        return inflated

    def clear_unknown(self, grid: np.ndarray) -> np.ndarray:
        if grid is None:
            return self.create_empty_grid()
        cleared = grid.copy()
        cleared[cleared == -1] = 0
        return cleared

    def copy(self, grid: np.ndarray) -> np.ndarray:
        if grid is None:
            return self.create_empty_grid()
        return grid.copy()

    def merge(self, grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
        if grid_a is None and grid_b is None:
            return self.create_empty_grid()
        if grid_a is None:
            return grid_b.copy()
        if grid_b is None:
            return grid_a.copy()
        if grid_a.shape != grid_b.shape:
            return grid_a.copy()
        merged = grid_a.copy()
        mask_a_unknown = grid_a == -1
        mask_b_free = grid_b == 0
        mask_b_occ = grid_b > 50
        merged[mask_a_unknown & mask_b_free] = 0
        merged[mask_a_unknown & mask_b_occ] = 100
        mask_a_free = grid_a == 0
        merged[mask_a_free & mask_b_occ] = 100
        return merged

    def crop(self, grid: np.ndarray) -> np.ndarray:
        if grid is None:
            return self.create_empty_grid()
        h, w = grid.shape
        known = grid != -1
        if not np.any(known):
            return grid.copy()
        rows = np.any(known, axis=1)
        cols = np.any(known, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return grid[rmin:rmax + 1, cmin:cmax + 1].copy()

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return world_to_grid(x, y, self.origin_x, self.origin_y, self.resolution)

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return grid_to_world(gx, gy, self.origin_x, self.origin_y, self.resolution)

    def is_occupied(self, gx: int, gy: int, grid: np.ndarray = None) -> bool:
        g = grid if grid is not None else self._get_default_grid()
        h, w = g.shape
        if not (0 <= gx < w and 0 <= gy < h):
            return False
        return g[gy, gx] > 50

    def is_free(self, gx: int, gy: int, grid: np.ndarray = None) -> bool:
        g = grid if grid is not None else self._get_default_grid()
        h, w = g.shape
        if not (0 <= gx < w and 0 <= gy < h):
            return False
        return g[gy, gx] == 0

    def is_unknown(self, gx: int, gy: int, grid: np.ndarray = None) -> bool:
        g = grid if grid is not None else self._get_default_grid()
        h, w = g.shape
        if not (0 <= gx < w and 0 <= gy < h):
            return True
        return g[gy, gx] == -1

    def set_occupied(self, gx: int, gy: int, grid: np.ndarray = None) -> np.ndarray:
        g = grid if grid is not None else self.create_empty_grid()
        h, w = g.shape
        if 0 <= gx < w and 0 <= gy < h:
            g[gy, gx] = 100
        return g

    def set_free(self, gx: int, gy: int, grid: np.ndarray = None) -> np.ndarray:
        g = grid if grid is not None else self.create_empty_grid()
        h, w = g.shape
        if 0 <= gx < w and 0 <= gy < h:
            g[gy, gx] = 0
        return g

    def set_unknown(self, gx: int, gy: int, grid: np.ndarray = None) -> np.ndarray:
        g = grid if grid is not None else self.create_empty_grid()
        h, w = g.shape
        if 0 <= gx < w and 0 <= gy < h:
            g[gy, gx] = -1
        return g

    def neighbors8(self, gx: int, gy: int, grid: np.ndarray = None) -> List[Tuple[int, int]]:
        g = grid if grid is not None else self._get_default_grid()
        h, w = g.shape
        nbs = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    nbs.append((nx, ny))
        return nbs

    def neighbors4(self, gx: int, gy: int, grid: np.ndarray = None) -> List[Tuple[int, int]]:
        g = grid if grid is not None else self._get_default_grid()
        h, w = g.shape
        nbs = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < w and 0 <= ny < h:
                nbs.append((nx, ny))
        return nbs

    def statistics(self, grid: np.ndarray = None) -> dict:
        g = grid if grid is not None else self._get_default_grid()
        h, w = g.shape
        occupied = int(np.count_nonzero(g > 50))
        free = int(np.count_nonzero(g == 0))
        unknown = int(np.count_nonzero(g == -1))
        return {
            "width": w,
            "height": h,
            "occupied": occupied,
            "free": free,
            "unknown": unknown,
            "resolution": self.resolution
        }

    def reset(self):
        with self._lock:
            self._grids.clear()
            self._global_grid = None
            self._global_ros = None

    def update_grid(self, robot_id: str, occupancy_grid: OccupancyGrid):
        with self._lock:
            self._grids[robot_id] = self.from_ros(occupancy_grid)

    def get_combined_grid(self) -> Optional[OccupancyGrid]:
        with self._lock:
            if self._global_ros is not None:
                return self._global_ros
            if not self._grids:
                return None
            combined = None
            for grid in self._grids.values():
                if combined is None:
                    combined = grid.copy()
                else:
                    combined = self.merge(combined, grid)
            if combined is not None:
                self._global_grid = combined
                self._global_ros = self.to_ros(combined)
                return self._global_ros
            return None

    def set_global_map(self, occupancy_grid: OccupancyGrid):
        with self._lock:
            self._global_ros = occupancy_grid
            if occupancy_grid is not None:
                self._global_grid = self.from_ros(occupancy_grid)

    def _get_default_grid(self) -> np.ndarray:
        if self._global_grid is not None:
            return self._global_grid
        if self._grids:
            return next(iter(self._grids.values()))
        return self.create_empty_grid()