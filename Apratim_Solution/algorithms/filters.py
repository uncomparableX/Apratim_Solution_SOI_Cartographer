#!/usr/bin/env python3

import copy
import numpy as np
import threading
from sensor_msgs.msg import LaserScan


class ScanFilter:
    def __init__(self, params):
        self.params = params
        self.min_range = float(params.min_scan_range)
        self.max_range = float(params.max_scan_range)
        self.median_window = 3
        self.smooth_window = 3
        self.outlier_factor = 3.0
        self._lock = threading.Lock()

    def filter_scan(self, scan: LaserScan) -> LaserScan:
        with self._lock:
            if scan is None:
                return None
            filtered = copy.deepcopy(scan)
            ranges = np.array(filtered.ranges, dtype=np.float32)
            ranges = self._remove_nan_internal(ranges)
            ranges = self._range_filter_internal(
                ranges, filtered.range_min, filtered.range_max
            )
            ranges = self._remove_outliers_internal(ranges)
            ranges = self._median_filter_internal(ranges)
            ranges = self._smooth_internal(ranges)
            filtered.ranges = ranges.tolist()
            return filtered

    def remove_nan(self, scan: LaserScan) -> LaserScan:
        with self._lock:
            if scan is None:
                return None
            filtered = copy.deepcopy(scan)
            ranges = np.array(filtered.ranges, dtype=np.float32)
            ranges = self._remove_nan_internal(ranges)
            filtered.ranges = ranges.tolist()
            return filtered

    def remove_outliers(self, scan: LaserScan) -> LaserScan:
        with self._lock:
            if scan is None:
                return None
            filtered = copy.deepcopy(scan)
            ranges = np.array(filtered.ranges, dtype=np.float32)
            ranges = self._remove_outliers_internal(ranges)
            filtered.ranges = ranges.tolist()
            return filtered

    def median_filter(self, scan: LaserScan) -> LaserScan:
        with self._lock:
            if scan is None:
                return None
            filtered = copy.deepcopy(scan)
            ranges = np.array(filtered.ranges, dtype=np.float32)
            ranges = self._median_filter_internal(ranges)
            filtered.ranges = ranges.tolist()
            return filtered

    def range_filter(self, scan: LaserScan) -> LaserScan:
        with self._lock:
            if scan is None:
                return None
            filtered = copy.deepcopy(scan)
            ranges = np.array(filtered.ranges, dtype=np.float32)
            ranges = self._range_filter_internal(
                ranges, filtered.range_min, filtered.range_max
            )
            filtered.ranges = ranges.tolist()
            return filtered

    def smooth(self, scan: LaserScan) -> LaserScan:
        with self._lock:
            if scan is None:
                return None
            filtered = copy.deepcopy(scan)
            ranges = np.array(filtered.ranges, dtype=np.float32)
            ranges = self._smooth_internal(ranges)
            filtered.ranges = ranges.tolist()
            return filtered

    def _remove_nan_internal(self, ranges: np.ndarray) -> np.ndarray:
        valid = np.isfinite(ranges)
        if not np.any(valid):
            return ranges
        result = ranges.copy()
        result[~valid] = 0.0
        return result

    def _range_filter_internal(
        self, ranges: np.ndarray, scan_min: float, scan_max: float
    ) -> np.ndarray:
        result = ranges.copy()
        lower = max(self.min_range, scan_min)
        upper = min(self.max_range, scan_max)
        mask = (result < lower) | (result > upper)
        result[mask] = 0.0
        return result

    def _remove_outliers_internal(self, ranges: np.ndarray) -> np.ndarray:
        valid = ranges > 0.0
        if not np.any(valid):
            return ranges
        valid_values = ranges[valid]
        median = float(np.median(valid_values))
        mad = float(np.median(np.abs(valid_values - median)))
        if mad < 1e-6:
            return ranges
        threshold = self.outlier_factor * mad
        result = ranges.copy()
        outlier_mask = valid & (np.abs(ranges - median) > threshold)
        result[outlier_mask] = 0.0
        return result

    def _median_filter_internal(self, ranges: np.ndarray) -> np.ndarray:
        n = len(ranges)
        if n < self.median_window:
            return ranges

        result = ranges.copy()
        half = self.median_window // 2

        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)

            window = ranges[start:end]
            valid = window[window > 0.0]

            if len(valid) > 0:
                result[i] = float(np.median(valid))

        return result

    def _smooth_internal(self, ranges: np.ndarray) -> np.ndarray:
        n = len(ranges)
        if n < self.smooth_window:
            return ranges

        result = ranges.copy()
        half = self.smooth_window // 2

        weights = np.ones(self.smooth_window, dtype=np.float32)
        weights /= np.sum(weights)

        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)

            window = ranges[start:end]
            valid = window[window > 0.0]

            if len(valid) > 0:
                result[i] = float(np.mean(valid))

        return result

    def reset(self):
        pass


# ---------------------------------------------------------------------
# Standalone filter functions
# ---------------------------------------------------------------------

def median_filter(ranges, window_size=3):
    ranges = np.asarray(ranges, dtype=np.float32)

    if len(ranges) < window_size:
        return ranges

    result = ranges.copy()
    half = window_size // 2

    for i in range(len(ranges)):
        start = max(0, i - half)
        end = min(len(ranges), i + half + 1)

        window = ranges[start:end]
        valid = window[np.isfinite(window)]

        if len(valid):
            result[i] = np.median(valid)

    return result


def occupancy_filter(*args, **kwargs):
    return args[0] if args else None


def exp_decay_filter(*args, **kwargs):
    return args[0] if args else None


def gaussian_filter_1d(*args, **kwargs):
    return args[0] if args else None