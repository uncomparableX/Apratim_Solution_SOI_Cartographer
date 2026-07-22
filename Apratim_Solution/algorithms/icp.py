
import math
import numpy as np
import threading
from typing import Tuple


class ICP:
    def __init__(self, max_iterations=50, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._lock = threading.Lock()
        self._last_transform = (0.0, 0.0, 0.0)
        self._last_error = float('inf')

    def transform_points(self, points: np.ndarray, transform: Tuple[float, float, float]) -> np.ndarray:
        if points.size == 0:
            return points
        dx, dy, yaw = transform
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
        return (points @ R.T) + np.array([dx, dy], dtype=np.float64)

    def nearest_neighbors(self, source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if source.size == 0 or target.size == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
        distances = np.linalg.norm(source[:, np.newaxis] - target[np.newaxis, :], axis=2)
        indices = np.argmin(distances, axis=1)
        min_dists = np.min(distances, axis=1)
        matched_target = target[indices]
        return matched_target, min_dists

    def estimate_transform(self, source: np.ndarray, target: np.ndarray) -> Tuple[Tuple[float, float, float], float]:
        if source.size == 0 or target.size == 0 or len(source) != len(target):
            return (0.0, 0.0, 0.0), float('inf')
        if len(source) < 2:
            dx = target[0, 0] - source[0, 0]
            dy = target[0, 1] - source[0, 1]
            return (dx, dy, 0.0), 0.0
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        H = source_centered.T @ target_centered
        try:
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            t = target_centroid - R @ source_centroid
            yaw = math.atan2(R[1, 0], R[0, 0])
            dx = float(t[0])
            dy = float(t[1])
            transformed = self.transform_points(source, (dx, dy, yaw))
            error = float(np.mean(np.linalg.norm(target - transformed, axis=1)))
            return (dx, dy, yaw), error
        except np.linalg.LinAlgError:
            dx = target_centroid[0] - source_centroid[0]
            dy = target_centroid[1] - source_centroid[1]
            transformed = source + np.array([dx, dy])
            error = float(np.mean(np.linalg.norm(target - transformed, axis=1)))
            return (dx, dy, 0.0), error

    def compute_error(self, source: np.ndarray, target: np.ndarray) -> float:
        if source.size == 0 or target.size == 0:
            return float('inf')
        matched_target, min_dists = self.nearest_neighbors(source, target)
        if len(min_dists) == 0:
            return float('inf')
        return float(np.mean(min_dists))

    def _compose_transform(self, total: Tuple[float, float, float], incremental: Tuple[float, float, float]) -> Tuple[float, float, float]:
        dx1, dy1, yaw1 = total
        dx2, dy2, yaw2 = incremental
        cos_y2 = math.cos(yaw2)
        sin_y2 = math.sin(yaw2)
        dx = cos_y2 * dx1 - sin_y2 * dy1 + dx2
        dy = sin_y2 * dx1 + cos_y2 * dy1 + dy2
        yaw = yaw1 + yaw2
        return (dx, dy, yaw)

    def _inverse_transform(self, transform: Tuple[float, float, float]) -> Tuple[float, float, float]:
        dx, dy, yaw = transform
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        dx_inv = -(dx * cos_y + dy * sin_y)
        dy_inv = dx * sin_y - dy * cos_y
        yaw_inv = -yaw
        return (dx_inv, dy_inv, yaw_inv)

    def fit(self, source_points: np.ndarray, target_points: np.ndarray) -> Tuple[Tuple[float, float, float], float]:
        with self._lock:
            if source_points.size == 0 or target_points.size == 0:
                return (0.0, 0.0, 0.0), 0.0
            source = np.asarray(source_points, dtype=np.float64)
            target = np.asarray(target_points, dtype=np.float64)
            if len(source) < 3 or len(target) < 3:
                if len(source) > 0 and len(target) > 0:
                    source_centroid = np.mean(source, axis=0)
                    target_centroid = np.mean(target, axis=0)
                    dx = target_centroid[0] - source_centroid[0]
                    dy = target_centroid[1] - source_centroid[1]
                    transformed = source + np.array([dx, dy])
                    error = self.compute_error(transformed, target)
                    max_dim = max(np.ptp(target[:, 0]), np.ptp(target[:, 1]))
                    fitness = max(0.0, 1.0 - error / max_dim) if max_dim > 0 else 0.0
                    return (-dx, -dy, 0.0), fitness
                return (0.0, 0.0, 0.0), 0.0
            current_source = source.copy()
            total_transform = (0.0, 0.0, 0.0)
            best_error = float('inf')
            prev_error = float('inf')
            for i in range(self.max_iterations):
                matched_target, _ = self.nearest_neighbors(current_source, target)
                if len(matched_target) == 0:
                    break
                transform, error = self.estimate_transform(current_source, matched_target)
                if math.isnan(error) or math.isinf(error):
                    break
                total_transform = self._compose_transform(total_transform, transform)
                current_source = self.transform_points(current_source, transform)
                if error < best_error:
                    best_error = error
                improvement = prev_error - error
                if abs(improvement) < self.tolerance:
                    break
                prev_error = error
            self._last_transform = total_transform
            self._last_error = best_error
            inv_transform = self._inverse_transform(total_transform)
            transformed_target = self.transform_points(target, inv_transform)
            distances = np.linalg.norm(transformed_target - source, axis=1)
            max_dim = max(np.ptp(target[:, 0]), np.ptp(target[:, 1]))
            if max_dim > 0:
                fitness = max(0.0, min(1.0, 1.0 - np.mean(distances) / max_dim))
            else:
                fitness = 0.0
            return inv_transform, fitness

    def reset(self):
        with self._lock:
            self._last_transform = (0.0, 0.0, 0.0)
            self._last_error = float('inf')