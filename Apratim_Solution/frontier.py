import math
import numpy as np
from typing import List, Tuple, Optional
from nav_msgs.msg import OccupancyGrid
from .utilities import world_to_grid, grid_to_world, euclidean_distance, normalize_angle


class FrontierDetector:
    def __init__(self, params):
        self.params = params
        self.resolution = float(params.map_resolution)
        self.origin_x = float(params.map_origin_x)
        self.origin_y = float(params.map_origin_y)
        self.min_size = int(params.frontier_min_size)
        self.cluster_distance = float(params.frontier_cluster_distance)
        self.min_distance = float(getattr(params, 'frontier_min_distance', 0.5))
        self._last_frontiers: List[Tuple[float, float]] = []
        self._last_clusters: List[List[Tuple[float, float]]] = []
        self._last_grid: Optional[OccupancyGrid] = None

    def detect(self, occupancy_grid: OccupancyGrid, robot_poses: List[Tuple[float, float, float]] = None) -> List[Tuple[float, float]]:
        if occupancy_grid is None or not hasattr(occupancy_grid, 'data') or occupancy_grid.data is None:
            self._last_frontiers = []
            self._last_grid = None
            return []
        w = occupancy_grid.info.width
        h = occupancy_grid.info.height
        grid = np.array(occupancy_grid.data, dtype=np.int8).reshape((h, w))
        free_mask = grid == 0
        unknown_mask = grid == -1
        if not np.any(free_mask) or not np.any(unknown_mask):
            self._last_frontiers = []
            self._last_grid = occupancy_grid
            return []
        padded = np.pad(unknown_mask.astype(np.int8), 1, mode='constant', constant_values=0)
        neighbor_sum = np.zeros((h, w), dtype=np.int32)
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                neighbor_sum += padded[dy:dy + h, dx:dx + w]
        frontier_mask = free_mask & (neighbor_sum > 0)
        if not np.any(frontier_mask):
            self._last_frontiers = []
            self._last_grid = occupancy_grid
            return []
        fy, fx = np.where(frontier_mask)
        wx = fx.astype(np.float32) * self.resolution + self.origin_x + self.resolution * 0.5
        wy = fy.astype(np.float32) * self.resolution + self.origin_y + self.resolution * 0.5
        frontiers = list(zip(wx.tolist(), wy.tolist()))

        # Filter out frontiers too close to robots
        if robot_poses is not None:
            filtered = []
            for f in frontiers:
                too_close = False
                for pose in robot_poses:
                    if euclidean_distance((pose[0], pose[1]), f) < self.min_distance:
                        too_close = True
                        break
                if not too_close:
                    filtered.append(f)
            frontiers = filtered

        self._last_frontiers = frontiers
        self._last_grid = occupancy_grid
        return frontiers

    def cluster_frontiers(self, frontiers: List[Tuple[float, float]], cluster_distance: float = None) -> List[List[Tuple[float, float]]]:
        if not frontiers:
            self._last_clusters = []
            return []
        if cluster_distance is None:
            cluster_distance = self.cluster_distance
        connect_cells = max(1, int(round(cluster_distance / self.resolution)))
        grid_coords = []
        for wx, wy in frontiers:
            gx, gy = world_to_grid(wx, wy, self.origin_x, self.origin_y, self.resolution)
            grid_coords.append((gx, gy))
        coord_set = set(grid_coords)
        visited = set()
        clusters = []
        for start in grid_coords:
            if start in visited:
                continue
            cluster = []
            queue = [start]
            visited.add(start)
            while queue:
                cx, cy = queue.pop(0)
                cluster.append((cx, cy))
                for dx in range(-connect_cells, connect_cells + 1):
                    for dy in range(-connect_cells, connect_cells + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nb = (cx + dx, cy + dy)
                        if nb in coord_set and nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
            if len(cluster) >= self.min_size:
                world_cluster = []
                for gx, gy in cluster:
                    wx, wy = grid_to_world(gx, gy, self.origin_x, self.origin_y, self.resolution)
                    world_cluster.append((float(wx), float(wy)))
                clusters.append(world_cluster)
        self._last_clusters = clusters
        return clusters

    def get_frontier_size(self, frontier: Tuple[float, float]) -> int:
        """Approximate the cell count of the cluster a frontier point belongs to.

        solution_node.py scores individual frontier cells before this cycle's
        clustering pass has run (clustering happens afterwards, on the
        filtered set), so there is no exact cluster available yet for a given
        point. This looks up the previous cycle's clusters (self._last_clusters,
        populated by cluster_frontiers()) and returns the size of whichever
        cluster's centroid is nearest -- a one-cycle-stale but cheap and
        stable proxy for "how large is the region this frontier opens onto",
        used so larger frontiers are preferred over single-cell noise.
        """
        if not self._last_clusters:
            return 1
        best_size = 1
        best_dist = float('inf')
        for cluster in self._last_clusters:
            if not cluster:
                continue
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            dist = euclidean_distance((cx, cy), frontier)
            if dist < best_dist:
                best_dist = dist
                best_size = len(cluster)
        return best_size

    def compute_information_gain(self, cluster: List[Tuple[float, float]], occupancy_grid: OccupancyGrid) -> float:
        if not cluster or occupancy_grid is None:
            return 0.0
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        gx, gy = world_to_grid(cx, cy, self.origin_x, self.origin_y, self.resolution)
        grid = np.array(occupancy_grid.data, dtype=np.int8).reshape((occupancy_grid.info.height, occupancy_grid.info.width))
        h, w = grid.shape
        radius = 5
        x_min = max(0, gx - radius)
        x_max = min(w, gx + radius + 1)
        y_min = max(0, gy - radius)
        y_max = min(h, gy + radius + 1)
        if x_min >= x_max or y_min >= y_max:
            return 0.0
        window = grid[y_min:y_max, x_min:x_max]
        unknown_count = int(np.sum(window == -1))
        return float(unknown_count) * self.resolution * self.resolution

    def score_frontiers(self, clusters: List[List[Tuple[float, float]]], robot_pose: Tuple[float, float, float]) -> List[Tuple[Tuple[float, float], float]]:
        if not clusters:
            return []
        occupancy_grid = self._last_grid
        rx, ry, ryaw = robot_pose
        scores = []
        for cluster in clusters:
            if not cluster:
                continue
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            info_gain = 0.0
            if occupancy_grid is not None:
                info_gain = self.compute_information_gain(cluster, occupancy_grid)
            dist = euclidean_distance((rx, ry), (cx, cy))
            distance_cost = dist * 0.5
            heading = math.atan2(cy - ry, cx - rx)
            heading_error = abs(normalize_angle(heading - ryaw))
            heading_bonus = max(0.0, 1.0 - heading_error / math.pi) * 2.0
            score = info_gain - distance_cost + heading_bonus
            scores.append(((cx, cy), score))
        return scores

    def best_frontier(self, robot_pose: Tuple[float, float, float], occupancy_grid: OccupancyGrid) -> Optional[Tuple[float, float]]:
        frontiers = self.detect(occupancy_grid)
        if not frontiers:
            return None
        clusters = self.cluster_frontiers(frontiers)
        if not clusters:
            return None
        scores = self.score_frontiers(clusters, robot_pose)
        if not scores:
            return None
        best = max(scores, key=lambda x: x[1])
        return best[0]

    def remove_frontier(self, frontier: Tuple[float, float]):
        if self._last_frontiers and frontier in self._last_frontiers:
            self._last_frontiers.remove(frontier)

    def frontier_count(self) -> int:
        return len(self._last_frontiers)

    def cluster_count(self) -> int:
        return len(self._last_clusters)

    def reset(self):
        self._last_frontiers = []
        self._last_clusters = []
        self._last_grid = None