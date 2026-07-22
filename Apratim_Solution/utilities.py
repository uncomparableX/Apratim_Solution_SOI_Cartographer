import math
import time
import numpy as np
from typing import Tuple, List, Optional, Union
from geometry_msgs.msg import Quaternion, Pose, Point


def ros_now(node) -> float:
    """Seconds from the node's ROS clock. Unlike time.time(), this reflects
    simulation time (from Gazebo's /clock) whenever the node's use_sim_time
    parameter is set -- which every timeout/staleness/blacklist-expiry check
    in this solution needs, since they are all reasoning about how much the
    robot's world has actually progressed, not how much wall-clock time has
    passed while waiting on a simulation that may be running far from
    real-time. Falls back to time.time() defensively if the clock is
    unavailable for any reason (e.g. during early construction)."""
    try:
        return node.get_clock().now().nanoseconds / 1e9
    except Exception:
        return time.time()


def quaternion_to_yaw(q: Union[Quaternion, Tuple[float, float, float, float]]) -> float:
    if isinstance(q, Quaternion):
        x, y, z, w = q.x, q.y, q.z, q.w
    else:
        x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def pose_distance(pose_a: Tuple[float, float, float], pose_b: Tuple[float, float, float]) -> float:
    return math.hypot(pose_a[0] - pose_b[0], pose_a[1] - pose_b[1])


def euclidean_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def scan_to_points(scan, pose: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    ranges = np.array(scan.ranges, dtype=np.float32)
    angles = scan.angle_min + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
    valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32)
    valid_ranges = ranges[valid]
    valid_angles = angles[valid]
    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)
    points = np.column_stack((x, y))
    if pose is not None:
        points = transform_points(points, pose)
    return points


def transform_points(points: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
    if points.size == 0:
        return points
    px, py, yaw = pose
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    return (points @ R.T) + np.array([px, py], dtype=np.float32)


def apply_transform_2d(x: float, y: float, transform: Tuple[float, float, float]) -> Tuple[float, float]:
    """Map a point from a robot's own odometry frame into the base/canonical frame.

    `transform` = (dx, dy, dyaw) is the rigid transform such that
    p_base = R(dyaw) @ p_own + (dx, dy) -- the same convention MapMerger uses
    to fold one robot's map into the other's frame when merging.
    """
    dx, dy, dyaw = transform
    cos_a, sin_a = math.cos(dyaw), math.sin(dyaw)
    return (x * cos_a - y * sin_a + dx, x * sin_a + y * cos_a + dy)


def apply_inverse_transform_2d(x: float, y: float, transform: Tuple[float, float, float]) -> Tuple[float, float]:
    """Inverse of apply_transform_2d: map a point from the base/canonical frame
    into a robot's own odometry frame."""
    dx, dy, dyaw = transform
    tx, ty = x - dx, y - dy
    cos_a, sin_a = math.cos(dyaw), math.sin(dyaw)
    return (tx * cos_a + ty * sin_a, -tx * sin_a + ty * cos_a)


def apply_transform_pose(pose: Tuple[float, float, float], transform: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y = apply_transform_2d(pose[0], pose[1], transform)
    return (x, y, normalize_angle(pose[2] + transform[2]))


def apply_inverse_transform_pose(pose: Tuple[float, float, float], transform: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y = apply_inverse_transform_2d(pose[0], pose[1], transform)
    return (x, y, normalize_angle(pose[2] - transform[2]))


def world_to_grid(wx: float, wy: float, origin_x: float, origin_y: float, resolution: float) -> Tuple[int, int]:
    # Use math.floor for correct negative coordinate handling
    gx = math.floor((wx - origin_x) / resolution)
    gy = math.floor((wy - origin_y) / resolution)
    return gx, gy


def grid_to_world(gx: int, gy: int, origin_x: float, origin_y: float, resolution: float) -> Tuple[float, float]:
    wx = gx * resolution + origin_x + resolution * 0.5
    wy = gy * resolution + origin_y + resolution * 0.5
    return wx, wy


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def wrap_angle(angle: float) -> float:
    return normalize_angle(angle)


def compute_path_length(path: List[Tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    path_arr = np.array(path, dtype=np.float32)
    diffs = np.diff(path_arr, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.sum(dists))


def interpolate_poses(pose_a: Tuple[float, float, float], pose_b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    x = pose_a[0] + t * (pose_b[0] - pose_a[0])
    y = pose_a[1] + t * (pose_b[1] - pose_a[1])
    yaw_diff = normalize_angle(pose_b[2] - pose_a[2])
    yaw = normalize_angle(pose_a[2] + t * yaw_diff)
    return x, y, yaw


def pose_to_tuple(pose: Union[Pose, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if isinstance(pose, Pose):
        x = pose.position.x
        y = pose.position.y
        yaw = quaternion_to_yaw(pose.orientation)
        return x, y, yaw
    return pose


def tuple_to_pose(pose_tuple: Tuple[float, float, float]) -> Pose:
    p = Pose()
    p.position.x = pose_tuple[0]
    p.position.y = pose_tuple[1]
    p.position.z = 0.0
    q = yaw_to_quaternion(pose_tuple[2])
    p.orientation = q
    return p


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    return min(max_val, max(min_val, value))