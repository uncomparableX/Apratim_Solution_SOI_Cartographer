import math
import threading
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header
from .utilities import normalize_angle

try:
    from .algorithms.icp import ICP
    _HAS_ICP = True
except Exception:
    _HAS_ICP = False


class MapMerger:
    def __init__(self, params):
        self.params = params
        self.resolution = float(params.map_resolution)
        self.origin_x = float(params.map_origin_x)
        self.origin_y = float(params.map_origin_y)
        self.width = int(params.map_width)
        self.height = int(params.map_height)
        self._lock = threading.Lock()
        self._robot_maps: Dict[str, OccupancyGrid] = {}
        self._transforms: Dict[Tuple[str, str], Tuple[Tuple[float, float, float], float]] = {}
        self._global_map: Optional[OccupancyGrid] = None
        self._merge_count = 0
        self._aligned = False
        self._confidence = 0.0
        self._base_name: Optional[str] = None

    def _logger(self):
        node = getattr(self.params, 'node', None)
        return node.get_logger() if node is not None else None

    def update_robot_map(self, robot_name: str, occupancy_grid: OccupancyGrid):
        with self._lock:
            self._robot_maps[robot_name] = occupancy_grid

    def get_transform_to_base(self, robot_name: str) -> Optional[Tuple[float, float, float]]:
        """Rigid transform (dx, dy, dyaw) mapping robot_name's own odometry
        frame into the base robot's frame -- i.e. the frame /global_map and
        every canonical-frame frontier/grid coordinate in this solution is
        expressed in (merge_maps always uses the first-registered robot's
        map as that base). Returns (0,0,0) if robot_name IS the base robot,
        or None if no cross-robot alignment with fitness >= icp_fitness_threshold
        exists yet.

        Each robot's odometry zeroes at its own actual spawn point and the
        competition deliberately does not reveal the offset between those
        spawn points (the training world spawns them 2m apart), so a raw
        robot_2 pose is not directly comparable to a robot_1-frame/merged-map
        coordinate until this transform is applied.
        """
        with self._lock:
            if self._base_name is None:
                return None
            if robot_name == self._base_name:
                return (0.0, 0.0, 0.0)
            cached = self._transforms.get((self._base_name, robot_name))
            if cached is None or cached[1] < self.params.icp_fitness_threshold:
                return None
            return cached[0]

    def align_maps(self, map1: OccupancyGrid, map2: OccupancyGrid) -> Tuple[Tuple[float, float, float], float]:
        if _HAS_ICP:
            try:
                transform, fitness = self._align_with_icp(map1, map2)
                if fitness >= self.params.icp_fitness_threshold:
                    return transform, fitness
            except Exception:
                pass
        transform, fitness = self._align_occupancy(map1, map2)
        return transform, fitness

    def _align_with_icp(self, map1: OccupancyGrid, map2: OccupancyGrid) -> Tuple[Tuple[float, float, float], float]:
        pts1 = self._grid_to_points(map1)
        pts2 = self._grid_to_points(map2)
        if len(pts1) < 10 or len(pts2) < 10:
            return (0.0, 0.0, 0.0), 0.0
        icp = ICP(max_iterations=self.params.icp_max_iterations, tolerance=self.params.icp_tolerance)
        transform, fitness = icp.fit(pts1, pts2)
        if transform is None:
            return (0.0, 0.0, 0.0), 0.0
        return transform, fitness

    def _align_occupancy(self, map1: OccupancyGrid, map2: OccupancyGrid) -> Tuple[Tuple[float, float, float], float]:
        pts1 = self._grid_to_points(map1)
        pts2 = self._grid_to_points(map2)
        if len(pts1) < 5 or len(pts2) < 5:
            return (0.0, 0.0, 0.0), 0.0
        c1 = np.mean(pts1, axis=0)
        c2 = np.mean(pts2, axis=0)
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dyaw = 0.0
        fitness = 0.0
        if len(pts1) >= 3 and len(pts2) >= 3:
            try:
                cov1 = np.cov(pts1.T)
                cov2 = np.cov(pts2.T)
                if cov1.shape == (2, 2) and cov2.shape == (2, 2):
                    eig1 = np.linalg.eigvals(cov1)
                    eig2 = np.linalg.eigvals(cov2)
                    if np.all(np.isreal(eig1)) and np.all(np.isreal(eig2)) and np.max(eig1) > 1e-6 and np.max(eig2) > 1e-6:
                        _, ev1 = np.linalg.eig(cov1)
                        _, ev2 = np.linalg.eig(cov2)
                        v1 = ev1[:, np.argmax(eig1)].real
                        v2 = ev2[:, np.argmax(eig2)].real
                        angle1 = math.atan2(v1[1], v1[0])
                        angle2 = math.atan2(v2[1], v2[0])
                        dyaw = normalize_angle(angle1 - angle2)
                        cos_yaw = math.cos(dyaw)
                        sin_yaw = math.sin(dyaw)
                        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                        aligned_c2 = R @ c2
                        dx = c1[0] - aligned_c2[0]
                        dy = c1[1] - aligned_c2[1]
                        transformed = (R @ pts2.T).T + np.array([dx, dy])
                        distances = np.linalg.norm(transformed[:, None] - pts1[None, :], axis=2)
                        min_dists = np.min(distances, axis=1)
                        fitness = float(np.mean(min_dists < self.resolution * 2))
                        return (dx, dy, dyaw), fitness
            except Exception:
                pass
        max_dim = max(self.width, self.height) * self.resolution
        fitness = max(0.0, 1.0 - math.hypot(dx, dy) / max_dim) * 0.3
        return (dx, dy, dyaw), fitness

    def _grid_to_points(self, grid: OccupancyGrid) -> np.ndarray:
        data = np.array(grid.data, dtype=np.int8)
        h = grid.info.height
        w = grid.info.width
        grid_arr = data.reshape((h, w))
        occ_y, occ_x = np.where(grid_arr > 50)
        if len(occ_x) == 0:
            return np.empty((0, 2), dtype=np.float32)
        wx = occ_x.astype(np.float32) * grid.info.resolution + grid.info.origin.position.x + grid.info.resolution * 0.5
        wy = occ_y.astype(np.float32) * grid.info.resolution + grid.info.origin.position.y + grid.info.resolution * 0.5
        return np.column_stack((wx, wy))

    def merge_maps(self, map1: OccupancyGrid, map2: OccupancyGrid, transform: Tuple[float, float, float]) -> Optional[OccupancyGrid]:
        if map1 is None or map2 is None:
            return None
        dx, dy, dyaw = transform
        cos_yaw = math.cos(dyaw)
        sin_yaw = math.sin(dyaw)
        h1 = map1.info.height
        w1 = map1.info.width
        h2 = map2.info.height
        w2 = map2.info.width
        res1 = map1.info.resolution
        res2 = map2.info.resolution
        ox1 = map1.info.origin.position.x
        oy1 = map1.info.origin.position.y
        ox2 = map2.info.origin.position.x
        oy2 = map2.info.origin.position.y
        data1 = np.array(map1.data, dtype=np.int8).reshape((h1, w1))
        data2 = np.array(map2.data, dtype=np.int8).reshape((h2, w2))
        gx2, gy2 = np.meshgrid(np.arange(w2, dtype=np.float32), np.arange(h2, dtype=np.float32))
        wx2 = gx2 * res2 + ox2 + res2 * 0.5
        wy2 = gy2 * res2 + oy2 + res2 * 0.5
        wx1 = wx2 * cos_yaw - wy2 * sin_yaw + dx
        wy1 = wx2 * sin_yaw + wy2 * cos_yaw + dy
        gx1 = ((wx1 - ox1) / res1).astype(np.int32)
        gy1 = ((wy1 - oy1) / res1).astype(np.int32)
        valid = (gx1 >= 0) & (gx1 < w1) & (gy1 >= 0) & (gy1 < h1) & (data2 != -1)
        if not np.any(valid):
            return self._build_occupancy_grid(data1, map1.info)
        vgx1 = gx1[valid]
        vgy1 = gy1[valid]
        val2 = data2[valid]
        val1 = data1[vgy1, vgx1]
        fused = np.where(
            val1 == -1,
            val2,
            np.where(
                val2 == -1,
                val1,
                np.where((val1 > 50) | (val2 > 50), 100, 0)
            )
        )
        merged = data1.copy()
        merged[vgy1, vgx1] = fused
        return self._build_occupancy_grid(merged, map1.info)

    def _build_occupancy_grid(self, grid_data: np.ndarray, info_template: MapMetaData) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.params.node.get_clock().now().to_msg()
        grid.header.frame_id = self.params.global_map_frame
        grid.info = MapMetaData()
        grid.info.resolution = info_template.resolution
        grid.info.width = grid_data.shape[1]
        grid.info.height = grid_data.shape[0]
        grid.info.origin.position.x = info_template.origin.position.x
        grid.info.origin.position.y = info_template.origin.position.y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.x = 0.0
        grid.info.origin.orientation.y = 0.0
        grid.info.origin.orientation.z = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = grid_data.ravel().tolist()
        return grid

    def update(self) -> Optional[OccupancyGrid]:
        with self._lock:
            if len(self._robot_maps) < 2:
                return None
            names = list(self._robot_maps.keys())
            self._base_name = names[0]
            base_map = self._robot_maps[names[0]]
            merged = base_map
            worst_fitness = 1.0
            any_recomputed = False
            for i in range(1, len(names)):
                other_map = self._robot_maps[names[i]]
                key = (names[0], names[i])
                cached = self._transforms.get(key)
                # Reuse a cached transform only once it has met the fitness
                # bar. Previously *any* cached transform was reused forever,
                # so whatever alignment the very first attempt produced --
                # typically computed from just a handful of wall cells each
                # robot had seen moments after spawning -- was locked in for
                # the rest of the run even if it was a poor fit. Retrying each
                # cycle is cheap relative to the 1s+ merge period, and once a
                # good fit is found it is cached exactly as before.
                if cached is not None and cached[1] >= self.params.icp_fitness_threshold:
                    transform, fitness = cached
                else:
                    transform, fitness = self.align_maps(merged, other_map)
                    self._transforms[key] = (transform, fitness)
                    any_recomputed = True
                worst_fitness = min(worst_fitness, fitness)
                merged = self.merge_maps(merged, other_map, transform)
                if merged is None:
                    return None
            self._global_map = merged
            self._merge_count += 1
            self._aligned = worst_fitness >= self.params.icp_fitness_threshold
            self._confidence = worst_fitness
            logger = self._logger()
            if logger is not None:
                data = np.array(merged.data, dtype=np.int8)
                total = len(data)
                known = int(np.sum(data >= 0))
                occupied = int(np.sum(data > 50))
                free = known - occupied
                unknown = total - known
                coverage = 100.0 * known / total if total > 0 else 0.0
                logger.info(
                    f'[merger] merge #{self._merge_count}: fitness={worst_fitness:.2f} '
                    f'({"recomputed" if any_recomputed else "cached"}, '
                    f'threshold={self.params.icp_fitness_threshold:.2f}) '
                    f'known={known} free={free} occupied={occupied} unknown={unknown} '
                    f'coverage={coverage:.1f}%'
                )
            return merged

    def get_global_map(self) -> Optional[OccupancyGrid]:
        with self._lock:
            return self._global_map

    def merge_ready(self) -> bool:
        with self._lock:
            return len(self._robot_maps) >= 2

    def statistics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "merges": self._merge_count,
                "aligned": self._aligned,
                "confidence": self._confidence,
                "robot_maps": len(self._robot_maps)
            }

    def reset(self):
        with self._lock:
            self._robot_maps.clear()
            self._transforms.clear()
            self._global_map = None
            self._merge_count = 0
            self._aligned = False
            self._confidence = 0.0