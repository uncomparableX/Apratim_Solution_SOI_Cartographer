import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
import numpy as np
import math
import threading
from enum import Enum, auto
from collections import deque
from tf2_ros import Buffer, TransformListener, TransformException

from .robot_state import RobotState
from .occupancy_grid import OccupancyGridManager
from .mapper import Mapper
from .frontier import FrontierDetector
from .planner import Planner
from .controller import Controller
from .coordinator import Coordinator
from .merger import MapMerger
from .visualization import Visualization
from .utilities import (
    quaternion_to_yaw, normalize_angle, clamp_value, euclidean_distance,
    apply_transform_pose, apply_inverse_transform_pose, apply_inverse_transform_2d,
    ros_now,
)
from .parameters import Parameters


class RobotMode(Enum):
    IDLE = auto()
    EXPLORING = auto()
    NAVIGATING = auto()
    RECOVERING = auto()
    RETURNING = auto()
    COMPLETE = auto()
    STUCK = auto()
    EMERGENCY = auto()


class RecoveryPhase(Enum):
    NONE = auto()
    STOP = auto()
    ROTATE_INPLACE = auto()
    BACKUP = auto()
    FORWARD_PROBE = auto()
    CLEAR_HISTORY = auto()


class _CanonicalPoseView:
    """Minimal duck-typed stand-in for RobotState, exposing only .name and
    .get_pose(), so Coordinator.assign_frontiers() (which only ever calls
    those two) can be handed a robot's pose already converted into the
    canonical/base-robot frame -- without changing RobotState itself, whose
    real get_pose() must keep returning that robot's raw odometry pose for
    its own planner/controller."""
    def __init__(self, name, pose):
        self.name = name
        self.robot_name = name
        self._pose = pose

    def get_pose(self):
        return self._pose


class SolutionNode(Node):
    def __init__(self):
        super().__init__('solution_node')

        self.params = Parameters(self)
        self._validate_parameters()

        self.callback_group = ReentrantCallbackGroup()

        self.robot1_state = RobotState('robot_1', self.params)
        self.robot2_state = RobotState('robot_2', self.params)

        self.mapper1 = Mapper(self.params, 'robot_1')
        self.mapper2 = Mapper(self.params, 'robot_2')
        self.grid_manager = OccupancyGridManager(self.params)

        self.frontier_detector = FrontierDetector(self.params)
        self.planner = Planner(self.params)
        self.controller1 = Controller(self.params, 'robot_1')
        self.controller2 = Controller(self.params, 'robot_2')

        self.coordinator = Coordinator(self.params)
        self.map_merger = MapMerger(self.params)
        self.visualizer = Visualization(self.params)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        self.map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )
        self.cmd_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.scan1_sub = self.create_subscription(
            LaserScan, '/robot_1/scan', self.scan1_callback,
            self.sensor_qos, callback_group=self.callback_group
        )
        self.odom1_sub = self.create_subscription(
            Odometry, '/robot_1/odom', self.odom1_callback,
            self.sensor_qos, callback_group=self.callback_group
        )
        self.scan2_sub = self.create_subscription(
            LaserScan, '/robot_2/scan', self.scan2_callback,
            self.sensor_qos, callback_group=self.callback_group
        )
        self.odom2_sub = self.create_subscription(
            Odometry, '/robot_2/odom', self.odom2_callback,
            self.sensor_qos, callback_group=self.callback_group
        )

        self.cmd_vel1_pub = self.create_publisher(
            Twist, '/robot_1/cmd_vel', self.cmd_qos
        )
        self.cmd_vel2_pub = self.create_publisher(
            Twist, '/robot_2/cmd_vel', self.cmd_qos
        )
        self.global_map_pub = self.create_publisher(
            OccupancyGrid, '/global_map', self.map_qos
        )

        self.mapping_timer = self.create_timer(
            self.params.mapping_dt, self.mapping_callback,
            callback_group=self.callback_group
        )
        self.exploration_timer = self.create_timer(
            self.params.exploration_dt, self.exploration_callback,
            callback_group=self.callback_group
        )
        self.control_timer = self.create_timer(
            self.params.control_dt, self.control_callback,
            callback_group=self.callback_group
        )
        self.coordination_timer = self.create_timer(
            self.params.coordination_dt, self.coordination_callback,
            callback_group=self.callback_group
        )
        self.merge_timer = self.create_timer(
            self.params.merge_dt, self.merge_callback,
            callback_group=self.callback_group
        )
        self.recovery_timer = self.create_timer(
            self.params.recovery_dt, self.recovery_callback,
            callback_group=self.callback_group
        )
        self.viz_timer = self.create_timer(
            self.params.viz_dt, self.visualization_callback,
            callback_group=self.callback_group
        )
        self.status_timer = self.create_timer(
            5.0, self.status_report_callback,
            callback_group=self.callback_group
        )

        self.data_lock = threading.Lock()

        self.scan1_buffer = None
        self.scan2_buffer = None
        self.odom1_buffer = None
        self.odom2_buffer = None
        self.scan1_time = 0.0
        self.scan2_time = 0.0
        self.odom1_time = 0.0
        self.odom2_time = 0.0

        self.frontiers = []
        self.frontier_clusters = []
        self.robot1_goal = None
        self.robot2_goal = None
        self.robot1_mode = RobotMode.IDLE
        self.robot2_mode = RobotMode.IDLE

        self.frontier_blacklist = set()
        self.frontier_attempts = {}
        self.frontier_timestamps = {}
        self.max_frontier_attempts = 3
        self.frontier_timeout = 45.0
        self.goal_assignment_time = 0.0
        self.blacklist_prune_interval = 120.0
        self.last_blacklist_prune = 0.0

        self.local_map1 = None
        self.local_map2 = None
        self.global_map = None
        self.map_transform_1to2 = None
        self.icp_fitness = 0.0
        self.map_aligned = False
        self.last_merge_time = 0.0
        self.merge_consecutive_failures = 0
        self.max_merge_failures = 10

        self.recovery1 = {
            'active': False,
            'phase': RecoveryPhase.NONE,
            'start_time': 0.0,
            'last_change': 0.0,
            'attempt_count': 0,
            'max_attempts': 5,
            'direction': 1.0
        }
        self.recovery2 = {
            'active': False,
            'phase': RecoveryPhase.NONE,
            'start_time': 0.0,
            'last_change': 0.0,
            'attempt_count': 0,
            'max_attempts': 5,
            'direction': -1.0
        }

        self.stuck_check_distance = 0.12
        self.stuck_check_time = 8.0
        self.pos_history1 = deque(maxlen=30)
        self.pos_history2 = deque(maxlen=30)
        self.vel_history1 = deque(maxlen=10)
        self.vel_history2 = deque(maxlen=10)

        self.coverage_percent = 0.0
        self.exploration_start_time = self._now()
        self.exploration_complete = False
        self.completion_reported = False
        self.last_coverage_increase_time = self._now()
        self.coverage_stall_threshold = 60.0

        self.min_robot_distance = 0.5
        self.safety_slowdown_distance = 1.2
        self.collision_avoidance_active = True

        self.loop_count = 0
        self.map_update_count = 0
        self.path_plan_count = 0
        self.merge_count = 0
        self.frontier_count = 0
        self.recovery_count = 0

        self.max_exploration_time = 600.0
        self.emergency_stop_triggered = False
        self.emergency_stop_time = 0.0

        self.scan_quality_threshold = 0.8
        self.min_valid_scan_points = 10

        self.get_logger().info('=' * 60)
        self.get_logger().info('SOI Cartographer Solution Node Initialized')
        self.get_logger().info(f'Mapping rate: {1.0/self.params.mapping_dt:.1f} Hz')
        self.get_logger().info(f'Control rate: {1.0/self.params.control_dt:.1f} Hz')
        self.get_logger().info(f'Exploration rate: {1.0/self.params.exploration_dt:.1f} Hz')
        self.get_logger().info(f'Merge rate: {1.0/self.params.merge_dt:.1f} Hz')
        self.get_logger().info(f'Recovery rate: {1.0/self.params.recovery_dt:.1f} Hz')
        self.get_logger().info('=' * 60)

    def _validate_parameters(self):
        if self.params.max_linear_speed <= 0:
            self.get_logger().warn('max_linear_speed invalid, using default 0.5')
            self.params.max_linear_speed = 0.5
        if self.params.max_angular_speed <= 0:
            self.get_logger().warn('max_angular_speed invalid, using default 1.0')
            self.params.max_angular_speed = 1.0
        if self.params.goal_tolerance < 0.05:
            self.get_logger().warn('goal_tolerance too small, using 0.25')
            self.params.goal_tolerance = 0.25
        if self.params.mapping_dt <= 0:
            self.params.mapping_dt = 0.2
        if self.params.control_dt <= 0:
            self.params.control_dt = 0.1
        if self.params.exploration_dt <= 0:
            self.params.exploration_dt = 1.0
        if self.params.merge_dt <= 0:
            self.params.merge_dt = 5.0
        if self.params.recovery_dt <= 0:
            self.params.recovery_dt = 0.5
        if self.params.viz_dt <= 0:
            self.params.viz_dt = 1.0

    def scan1_callback(self, msg):
        with self.data_lock:
            self.scan1_buffer = msg
            self.scan1_time = self._now()

    def scan2_callback(self, msg):
        with self.data_lock:
            self.scan2_buffer = msg
            self.scan2_time = self._now()

    def odom1_callback(self, msg):
        with self.data_lock:
            self.odom1_buffer = msg
            self.odom1_time = self._now()
        self._update_robot_state(self.robot1_state, msg)
        self._record_position(self.robot1_state, self.pos_history1)
        self._record_velocity(msg, self.vel_history1)

    def odom2_callback(self, msg):
        with self.data_lock:
            self.odom2_buffer = msg
            self.odom2_time = self._now()
        self._update_robot_state(self.robot2_state, msg)
        self._record_position(self.robot2_state, self.pos_history2)
        self._record_velocity(msg, self.vel_history2)

    def _update_robot_state(self, robot_state, odom_msg):
        try:
            robot_state.update_odometry(odom_msg)
        except Exception as e:
            self.get_logger().debug(f'State update error: {str(e)}')

    def _record_position(self, robot_state, history):
        pose = robot_state.get_pose()
        if pose is not None:
            history.append((pose[0], pose[1], self._now()))

    def _record_velocity(self, odom_msg, history):
        lin = odom_msg.twist.twist.linear
        ang = odom_msg.twist.twist.angular
        speed = math.hypot(lin.x, lin.y)
        # FIX: Removed absolute value which was destroying the angular velocity sign
        history.append((speed, ang.z, self._now()))

    def _now(self) -> float:
        """Sim-time-aware seconds (see utilities.ros_now). Every elapsed-time
        check in this node (scan/odom staleness, frontier cooldown/blacklist
        timing, stuck detection, exploration time budget, coverage-plateau
        detection) needs this instead of Python's wall-clock time module:
        those are all judgments about how far the *simulation* has
        progressed, and wall-clock time keeps advancing at real speed no
        matter how slowly Gazebo is actually stepping the simulation forward."""
        return ros_now(self)

    def _get_robot_pose(self, robot_state, robot_id):
        if self.params.use_tf_poses:
            try:
                tf_frame = f'{robot_id}/odom'
                base_frame = f'{robot_id}/base_link'
                transform = self.tf_buffer.lookup_transform(
                    tf_frame, base_frame, rclpy.time.Time()
                )
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                q = transform.transform.rotation
                yaw = quaternion_to_yaw(q)
                return (x, y, yaw)
            except (TransformException, Exception):
                pass
        return robot_state.get_pose()

    def _pose_to_canonical(self, robot_id, pose):
        """Express `pose` (that robot's own raw odometry pose) in the base
        robot's frame -- the frame /global_map, get_combined_grid(), and every
        frontier centroid derived from them are expressed in.

        Each robot's odometry zeroes at its own actual spawn point, and the
        competition does not reveal the offset between those spawn points, so
        robot_1's raw pose and robot_2's raw pose are two different coordinate
        systems even though solution_node.py previously compared them (and
        compared them to canonical-frame frontiers/grid coordinates) as if
        they were the same one. This returns None when no cross-robot
        alignment with adequate fitness exists yet, rather than guessing.
        """
        if pose is None:
            return None
        transform = self.map_merger.get_transform_to_base(robot_id)
        if transform is None:
            return None
        return apply_transform_pose(pose, transform)

    def _point_from_canonical(self, robot_id, x, y):
        """Inverse of _pose_to_canonical, for a bare (x, y) point (e.g. a
        frontier centroid chosen in canonical/base-robot frame) that needs to
        become a goal expressed in `robot_id`'s own frame before it can be
        handed to that robot's own controller/planner, which only ever see
        that robot's own raw odometry pose."""
        transform = self.map_merger.get_transform_to_base(robot_id)
        if transform is None:
            return None
        return apply_inverse_transform_2d(x, y, transform)

    def mapping_callback(self):
        with self.data_lock:
            scan1 = self.scan1_buffer
            scan2 = self.scan2_buffer
            odom1 = self.odom1_buffer
            odom2 = self.odom2_buffer

        now = self._now()
        if scan1 is not None and (now - self.scan1_time) < self.params.scan_buffer_timeout:
            pose1 = self._get_robot_pose(self.robot1_state, 'robot_1')
            if pose1 is not None:
                try:
                    if self._check_scan_quality(scan1):
                        known_before = self.mapper1.statistics()['known_cells']
                        self.local_map1 = self.mapper1.update(scan1, pose1)
                        if self.local_map1 is not None:
                            self.grid_manager.update_grid('robot_1', self.local_map1)
                            self.map_update_count += 1
                            stats1 = self.mapper1.statistics()
                            self.get_logger().debug(
                                f'[mapping] robot_1 update #{stats1["updates"]}: '
                                f'known={stats1["known_cells"]} (+{stats1["known_cells"]-known_before}) '
                                f'free={stats1["free_cells"]} occupied={stats1["occupied_cells"]} '
                                f'unknown={stats1["total_cells"]-stats1["known_cells"]}'
                            )
                except Exception as e:
                    self.get_logger().debug(f'Mapper1 error: {str(e)}')

        if scan2 is not None and (now - self.scan2_time) < self.params.scan_buffer_timeout:
            pose2 = self._get_robot_pose(self.robot2_state, 'robot_2')
            if pose2 is not None:
                try:
                    if self._check_scan_quality(scan2):
                        known_before = self.mapper2.statistics()['known_cells']
                        self.local_map2 = self.mapper2.update(scan2, pose2)
                        if self.local_map2 is not None:
                            self.grid_manager.update_grid('robot_2', self.local_map2)
                            self.map_update_count += 1
                            stats2 = self.mapper2.statistics()
                            self.get_logger().debug(
                                f'[mapping] robot_2 update #{stats2["updates"]}: '
                                f'known={stats2["known_cells"]} (+{stats2["known_cells"]-known_before}) '
                                f'free={stats2["free_cells"]} occupied={stats2["occupied_cells"]} '
                                f'unknown={stats2["total_cells"]-stats2["known_cells"]}'
                            )
                except Exception as e:
                    self.get_logger().debug(f'Mapper2 error: {str(e)}')

    def _check_scan_quality(self, scan_msg):
        try:
            ranges = np.array(scan_msg.ranges)
            # FIX: Do not reject .inf readings. Open space is valid and essential for mapping!
            valid = ranges > self.params.min_scan_range
            ratio = float(np.sum(valid)) / float(len(ranges))
            return ratio >= 0.05
        except Exception:
            return True

    def exploration_callback(self):
        """Assign new frontiers only when robots lack valid goals."""
        if self.exploration_complete or self.emergency_stop_triggered:
            return

        elapsed = self._now() - self.exploration_start_time
        if elapsed > self.max_exploration_time:
            self.get_logger().warn('Max exploration time reached, triggering emergency stop')
            self._trigger_emergency_stop()
            return

        grid = self.grid_manager.get_combined_grid()
        if grid is None:
            return

        pose1 = self._get_robot_pose(self.robot1_state, 'robot_1')
        pose2 = self._get_robot_pose(self.robot2_state, 'robot_2')
        # pose2 expressed in the canonical/base-robot frame that `grid` and
        # every frontier centroid derived from it are in -- or None if no
        # confident cross-robot alignment exists yet. Every comparison below
        # between a robot position and a canonical-frame coordinate uses this,
        # never the raw pose2 (see _pose_to_canonical's docstring for why).
        pose2_canonical = self._pose_to_canonical('robot_2', pose2)

        needs_new_goal_1 = (self.robot1_goal is None or
                          self.robot1_mode in [RobotMode.IDLE, RobotMode.COMPLETE] or
                          self._is_goal_stale(self.robot1_goal))
        needs_new_goal_2 = (self.robot2_goal is None or
                          self.robot2_mode in [RobotMode.IDLE, RobotMode.COMPLETE] or
                          self._is_goal_stale(self.robot2_goal))

        if not needs_new_goal_1 and not needs_new_goal_2:
            # Both robots have active goals; do not reassign
            return

        # Without a confident cross-robot alignment yet, there is no reliable
        # way to convert a canonical-frame frontier into robot_2's own frame,
        # so assigning it one here would reintroduce the exact bug this pass
        # fixed. Let it pick a goal directly from its own local map instead
        # of sitting idle -- which also gives ICP more scan overlap to work
        # with, helping alignment succeed sooner.
        if needs_new_goal_2 and pose2 is not None and pose2_canonical is None:
            if self._assign_independent_goal(2, self.local_map2, pose2, self.controller2, 'robot_2'):
                needs_new_goal_2 = False

        # Detect frontiers, filtering out those too close to either robot
        robot_poses = []
        if pose1 is not None:
            robot_poses.append(pose1)
        if pose2_canonical is not None:
            robot_poses.append(pose2_canonical)

        try:
            raw_frontiers = self.frontier_detector.detect(grid, robot_poses=robot_poses)
            self.frontier_count = len(raw_frontiers)
        except Exception as e:
            self.get_logger().debug(f'Frontier detection error: {str(e)}')
            return

        if not raw_frontiers:
            self._evaluate_completion(grid)
            self._prune_blacklist()
            return

        # needs_new_goal_2 may already be False (served independently above,
        # or no confident pose2_canonical to act on); only pursue the
        # canonical/coordinated path for what's left.
        if not needs_new_goal_1 and not needs_new_goal_2:
            self._evaluate_completion(grid)
            self._prune_blacklist()
            return

        filtered = self._filter_and_score_frontiers(raw_frontiers, pose1, pose2_canonical)
        self.frontiers = [f[0] for f in filtered]

        if not self.frontiers:
            self._evaluate_completion(grid)
            self._prune_blacklist()
            return

        self._build_frontier_clusters()
        self._assign_goals_to_robots(pose1, pose2_canonical, needs_new_goal_1, needs_new_goal_2)

        if self.coverage_percent > 0:
            self.last_coverage_increase_time = self._now()

        self._prune_blacklist()

    def _is_goal_stale(self, goal):
        """Check if a goal should be considered stale and replaced."""
        g_key = (round(goal[0], 2), round(goal[1], 2))
        if g_key in self.frontier_blacklist:
            return True
        if g_key in self.frontier_attempts and self.frontier_attempts[g_key] >= self.max_frontier_attempts:
            return True
        return False

    def _filter_and_score_frontiers(self, frontiers, pose1, pose2):
        filtered = []
        for f in frontiers:
            is_blacklisted = False
            
            # FIX: Use Euclidean distance to check blacklists! 
            for b_point in self.frontier_blacklist:
                if math.hypot(f[0] - b_point[0], f[1] - b_point[1]) < self.params.frontier_cluster_distance:
                    is_blacklisted = True
                    break
                    
            if not is_blacklisted:
                for failed_goal, count in self.frontier_attempts.items():
                    if math.hypot(f[0] - failed_goal[0], f[1] - failed_goal[1]) < self.params.frontier_cluster_distance:
                        if count >= self.max_frontier_attempts:
                            is_blacklisted = True
                            break
                            
            if is_blacklisted:
                continue

            in_cooldown = False
            for ts_goal, ts_time in self.frontier_timestamps.items():
                if math.hypot(f[0] - ts_goal[0], f[1] - ts_goal[1]) < self.params.frontier_cluster_distance:
                    if self._now() - ts_time < self.frontier_timeout:
                        in_cooldown = True
                        break
            if in_cooldown:
                continue

            d1 = float('inf') if pose1 is None else math.hypot(f[0]-pose1[0], f[1]-pose1[1])
            d2 = float('inf') if pose2 is None else math.hypot(f[0]-pose2[0], f[1]-pose2[1])
            if min(d1, d2) < self.params.goal_tolerance:
                continue
                
            score = self._score_frontier(f, d1, d2, pose1, pose2)
            filtered.append((f, score))
            
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered

    def _score_frontier(self, frontier, d1, d2, pose1, pose2):
        distance_score = 1.0 / (1.0 + min(d1, d2))
        size_score = 1.0
        if hasattr(self.frontier_detector, 'get_frontier_size'):
            try:
                size_score = self.frontier_detector.get_frontier_size(frontier) / 10.0
            except Exception:
                pass
        orientation_score = 1.0
        if pose1 is not None and pose2 is not None:
            nearest_pose = pose1 if d1 < d2 else pose2
            angle_to_frontier = math.atan2(frontier[1]-nearest_pose[1], frontier[0]-nearest_pose[0])
            orientation_diff = abs(normalize_angle(angle_to_frontier - nearest_pose[2]))
            orientation_score = 1.0 - (orientation_diff / math.pi)
        return distance_score * 0.5 + size_score * 0.3 + orientation_score * 0.2

    def _build_frontier_clusters(self):
        if not self.frontiers:
            self.frontier_clusters = []
            return
        try:
            self.frontier_clusters = self.frontier_detector.cluster_frontiers(
                self.frontiers, self.params.frontier_cluster_distance
            )
        except Exception:
            self.frontier_clusters = [[f] for f in self.frontiers]

    def _assign_goals_to_robots(self, pose1, pose2, needs1, needs2):
        if pose1 is None:
            return
        needs2 = needs2 and (pose2 is not None)

        centroids = []
        for cluster in self.frontier_clusters:
            if len(cluster) == 0:
                continue
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            
            # FIX: Explicitly check the calculated centroid against the blacklist.
            # This absolutely prevents the infinite assignment loop.
            c_key = (round(cx, 2), round(cy, 2))
            is_blacklisted = False
            for b_point in self.frontier_blacklist:
                if math.hypot(cx - b_point[0], cy - b_point[1]) <= self.params.frontier_cluster_distance:
                    is_blacklisted = True
                    break
                    
            if self.frontier_attempts.get(c_key, 0) >= self.max_frontier_attempts:
                is_blacklisted = True
                
            if is_blacklisted:
                continue
                
            centroids.append((cx, cy))

        if len(centroids) == 0:
            return

        available = list(centroids)
        if not needs1 and self.robot1_goal is not None:
            g1_rounded = (round(self.robot1_goal[0], 2), round(self.robot1_goal[1], 2))
            available = [c for c in available if (round(c[0], 2), round(c[1], 2)) != g1_rounded]
        if not needs2 and self.robot2_goal is not None:
            g2_canonical = self._pose_to_canonical('robot_2', (self.robot2_goal[0], self.robot2_goal[1], 0.0))
            if g2_canonical is not None:
                g2_rounded = (round(g2_canonical[0], 2), round(g2_canonical[1], 2))
                available = [c for c in available if (round(c[0], 2), round(c[1], 2)) != g2_rounded]

        if len(available) == 0:
            return

        robots_needing = []
        if needs1:
            robots_needing.append(('robot_1', pose1, self.controller1))
        if needs2:
            robots_needing.append(('robot_2', pose2, self.controller2))

        if len(robots_needing) == 0:
            return

        if len(available) == 1 and len(robots_needing) == 2:
            d1 = math.hypot(available[0][0]-pose1[0], available[0][1]-pose1[1])
            d2 = math.hypot(available[0][0]-pose2[0], available[0][1]-pose2[1])
            if d1 < d2:
                self._set_robot_goal(1, available[0])
            else:
                self._set_robot_goal_from_canonical(2, available[0])
            return

        if len(robots_needing) == 2 and len(available) >= 2:
            try:
                assignment = self.coordinator.assign_frontiers(
                    available, self.robot1_state, _CanonicalPoseView('robot_2', pose2)
                )
            except Exception as e:
                self.get_logger().debug(f'Coordination assignment error: {str(e)}')
                assignment = None

            if assignment is not None and len(assignment) >= 2:
                if needs1:
                    self._set_robot_goal(1, assignment[0])
                if needs2:
                    self._set_robot_goal_from_canonical(2, assignment[1])
            else:
                self._greedy_assignment(centroids, available, pose1, pose2, needs1, needs2)
        else:
            robot_name, robot_pose, _ = robots_needing[0]
            best = min(available, key=lambda c: math.hypot(c[0]-robot_pose[0], c[1]-robot_pose[1]))
            if robot_name == 'robot_1':
                self._set_robot_goal(1, best)
            else:
                self._set_robot_goal_from_canonical(2, best)

    def _set_robot_goal(self, rid, goal):
        """Set goal and initialize controller state consistently."""
        if rid == 1:
            self.robot1_goal = goal
            self.robot1_mode = RobotMode.EXPLORING
            self.robot1_state.set_goal(goal)
            self.controller1.set_goal(goal)
            self.controller1.set_path([])
        else:
            self.robot2_goal = goal
            self.robot2_mode = RobotMode.EXPLORING
            self.robot2_state.set_goal(goal)
            self.controller2.set_goal(goal)
            self.controller2.set_path([])
        self.goal_assignment_time = self._now()
        self.get_logger().info(f'Robot {rid} assigned goal: ({goal[0]:.2f}, {goal[1]:.2f})')

    def _set_robot_goal_from_canonical(self, rid, canonical_point):
        """Convert a frontier centroid chosen in canonical/base-robot frame
        into robot `rid`'s own odometry frame, then store it via
        _set_robot_goal. Every goal assigned to a non-base robot must go
        through this (never _set_robot_goal directly with a canonical-frame
        point), or that robot's own controller/planner -- which only ever see
        its own raw, differently-zeroed odometry pose -- will be steered
        toward coordinates that don't correspond to where it actually needs
        to go. For the base robot (robot_1) this is a no-op passthrough since
        its own frame already IS the canonical frame."""
        robot_name = 'robot_1' if rid == 1 else 'robot_2'
        point = self._point_from_canonical(robot_name, canonical_point[0], canonical_point[1])
        if point is None:
            self.get_logger().warn(
                f'{robot_name} lost cross-robot alignment while a goal was being assigned; skipping this cycle'
            )
            return
        self._set_robot_goal(rid, point)

    def _greedy_assignment(self, centroids, available, pose1, pose2, needs1, needs2):
        if len(available) >= 2:
            dists1 = [math.hypot(c[0]-pose1[0], c[1]-pose1[1]) for c in available]
            dists2 = [math.hypot(c[0]-pose2[0], c[1]-pose2[1]) for c in available]
            idx1 = int(np.argmin(dists1))
            idx2 = int(np.argmin(dists2))
            if idx1 == idx2:
                if dists1[idx1] < dists2[idx2]:
                    idx2 = int(np.argmin([d if i != idx1 else float('inf') for i, d in enumerate(dists2)]))
                else:
                    idx1 = int(np.argmin([d if i != idx2 else float('inf') for i, d in enumerate(dists1)]))
            if needs1:
                self._set_robot_goal(1, available[idx1])
            if needs2:
                self._set_robot_goal_from_canonical(2, available[idx2])
        elif len(available) == 1:
            d1 = math.hypot(available[0][0]-pose1[0], available[0][1]-pose1[1])
            d2 = math.hypot(available[0][0]-pose2[0], available[0][1]-pose2[1])
            if d1 < d2 and needs1:
                self._set_robot_goal(1, available[0])
            elif needs2:
                self._set_robot_goal_from_canonical(2, available[0])

    def _assign_independent_goal(self, rid, local_map, pose, controller, robot_name):
        if local_map is None or pose is None:
            return False
        try:
            raw = self.frontier_detector.detect(local_map, robot_poses=[pose])
        except Exception as e:
            return False
            
        if not raw:
            return False

        best = None
        best_score = -1.0
        for f in raw:
            is_blacklisted = False
            for b_point in self.frontier_blacklist:
                if math.hypot(f[0] - b_point[0], f[1] - b_point[1]) <= self.params.frontier_cluster_distance:
                    is_blacklisted = True
                    break
            if is_blacklisted:
                continue
                
            d = math.hypot(f[0]-pose[0], f[1]-pose[1])
            if d < self.params.goal_tolerance:
                continue
            score = 1.0 / (1.0 + d)
            if score > best_score:
                best_score = score
                best = f
                
        if best is None:
            return False
            
        self._set_robot_goal(rid, best)
        return True

    def _evaluate_completion(self, grid):
        coverage = self._compute_coverage(grid)
        self.coverage_percent = coverage

        if coverage > 95.0:
            self.exploration_complete = True
            self.robot1_mode = RobotMode.COMPLETE
            self.robot2_mode = RobotMode.COMPLETE
            if not self.completion_reported:
                elapsed = self._now() - self.exploration_start_time
                self.get_logger().info(
                    f'EXPLORATION COMPLETE: Coverage={coverage:.1f}% Time={elapsed:.1f}s'
                )
                self.completion_reported = True
                self._publish_cmd_vel(self.cmd_vel1_pub, 0.0, 0.0)
                self._publish_cmd_vel(self.cmd_vel2_pub, 0.0, 0.0)
        elif coverage > 85.0:
            time_since_increase = self._now() - self.last_coverage_increase_time
            if time_since_increase > self.coverage_stall_threshold:
                self.get_logger().info('Coverage stalled, marking complete')
                self.exploration_complete = True
                self.robot1_mode = RobotMode.COMPLETE
                self.robot2_mode = RobotMode.COMPLETE

    def control_callback(self):
        self.loop_count += 1

        if self.emergency_stop_triggered:
            self._publish_cmd_vel(self.cmd_vel1_pub, 0.0, 0.0)
            self._publish_cmd_vel(self.cmd_vel2_pub, 0.0, 0.0)
            return

        if self.recovery1['active']:
            self._execute_recovery1()
            return

        if self.recovery2['active']:
            self._execute_recovery2()
            return

        pose1 = self._get_robot_pose(self.robot1_state, 'robot_1')
        pose2 = self._get_robot_pose(self.robot2_state, 'robot_2')
        grid1, grid2 = self._select_planning_grids()

        # Collision avoidance needs both poses in ONE shared frame. pose1 and
        # pose2 come from two independent odometry frames (see
        # _pose_to_canonical's docstring), so comparing them directly -- as
        # this used to do -- produced a bogus "distance" between the robots
        # from the very first control tick (both frames read ~(0,0) near
        # start-up regardless of the real 2m+ spawn separation), which
        # permanently tripped the collision-avoidance branch and is why the
        # robots would only creep/oscillate instead of driving to their
        # goals. Transform each robot's pose into the OTHER's frame instead;
        # if no confident cross-robot alignment exists yet, skip the check
        # rather than act on numbers that don't mean what they look like.
        other_pose_for_1 = self._pose_to_canonical('robot_2', pose2)
        transform_2to1 = self.map_merger.get_transform_to_base('robot_2')
        other_pose_for_2 = (
            apply_inverse_transform_pose(pose1, transform_2to1)
            if (pose1 is not None and transform_2to1 is not None) else None
        )

        if pose1 is not None:
            self._control_robot(1, pose1, other_pose_for_1, grid1,
                              self.robot1_goal, self.controller1,
                              self.cmd_vel1_pub, 'robot_1')

        if pose2 is not None:
            self._control_robot(2, pose2, other_pose_for_2, grid2,
                              self.robot2_goal, self.controller2,
                              self.cmd_vel2_pub, 'robot_2')

    def _select_planning_grids(self):
        """Choose the occupancy grid each robot should plan and validate paths against.

        robot_1's own odometry frame IS the canonical/base frame that
        merge_maps() and get_combined_grid() are expressed in (merging always
        folds the other robot's map onto the first-registered robot's), so
        robot_1 can safely use the richer merged grid once one exists.

        robot_2's pose, goal, and controller all operate in robot_2's OWN
        odometry frame. Handing it the canonical/merged grid -- as this used
        to do once a merge existed -- would plan a path in one frame while
        executing it against a pose in another, which is a second instance of
        the same cross-frame mismatch described in _pose_to_canonical.
        Because every goal assigned to robot_2 is now converted into its own
        frame at assignment time (see _assign_goals_to_robots /
        _assign_independent_goal), robot_2 must always plan against its own
        local map so that goal, pose, and grid all agree on what frame they're in.
        """
        combined = self.grid_manager.get_combined_grid()
        grid1 = combined if combined is not None else self.local_map1
        grid2 = self.local_map2
        return grid1, grid2

    def _control_robot(self, rid, my_pose, other_pose, grid, goal,
                       controller, cmd_pub, robot_name):
        if goal is None:
            self._publish_cmd_vel(cmd_pub, 0.0, 0.0)
            if rid == 1:
                self.robot1_mode = RobotMode.IDLE
            else:
                self.robot2_mode = RobotMode.IDLE
            return

        if self._is_goal_reached(my_pose, goal):
            self.get_logger().info(f'{robot_name} reached goal')
            self._mark_frontier_success(goal)
            if rid == 1:
                self.robot1_goal = None
                self.robot1_mode = RobotMode.IDLE
            else:
                self.robot2_goal = None
                self.robot2_mode = RobotMode.IDLE
            controller.clear_goal()
            self._publish_cmd_vel(cmd_pub, 0.0, 0.0)
            return

        if grid is None:
            self._publish_cmd_vel(cmd_pub, 0.0, 0.0)
            return

        try:
            if not controller.valid_path():
                self._replan_robot(rid, my_pose, goal, grid)

            if not controller.valid_path():
                self.get_logger().warn(
                    f'{robot_name} path planning failed: '
                    f'{self.planner.last_rejection_reason or "no free route"}'
                )
                self._mark_frontier_failure(goal)
                if rid == 1:
                    self.robot1_goal = None
                    self.robot1_mode = RobotMode.IDLE
                else:
                    self.robot2_goal = None
                    self.robot2_mode = RobotMode.IDLE
                controller.clear_goal()
                self._publish_cmd_vel(cmd_pub, 0.0, 0.0)
                return

            current_path = controller.current_path()
            if not self.planner.is_path_valid(current_path, grid, start_pose=my_pose):
                self._replan_robot(rid, my_pose, goal, grid)
                if not controller.valid_path():
                    self._publish_cmd_vel(cmd_pub, 0.0, 0.0)
                    return
        except Exception as e:
            self.get_logger().warn(f'{robot_name} planning error: {str(e)}')
            self._publish_cmd_vel(cmd_pub, 0.0, 0.0)
            return

        if other_pose is not None and self.collision_avoidance_active:
            dist = math.hypot(my_pose[0]-other_pose[0], my_pose[1]-other_pose[1])
            if dist < self.min_robot_distance:
                self._execute_collision_avoidance(rid, my_pose, other_pose, cmd_pub)
                return
            elif dist < self.safety_slowdown_distance:
                self._apply_safety_limits(rid)

        try:
            cmd = controller.compute_velocity(my_pose)
            cmd = self._enforce_velocity_limits(cmd, rid)
            
            # FIX: REMOVED _apply_acceleration_limits here. 
            # controller.py handles physics safely. Doing it twice broke the physics engine.
            cmd_pub.publish(cmd)
            
            if rid == 1:
                self.robot1_mode = RobotMode.NAVIGATING
            else:
                self.robot2_mode = RobotMode.NAVIGATING
        except Exception as e:
            self._publish_cmd_vel(cmd_pub, 0.0, 0.0)

    def _is_goal_reached(self, pose, goal):
        return math.hypot(pose[0]-goal[0], pose[1]-goal[1]) < self.params.goal_tolerance

    def _replan_robot(self, rid, pose, goal, grid):
        try:
            new_path = self.planner.plan(pose, goal, grid)
            if new_path is not None and len(new_path) > 0:
                if rid == 1:
                    self.controller1.set_path(new_path)
                else:
                    self.controller2.set_path(new_path)
                self.path_plan_count += 1
            else:
                if rid == 1:
                    self.controller1.set_path([])
                else:
                    self.controller2.set_path([])
        except Exception as e:
            self.get_logger().debug(f'Replan error: {str(e)}')
            if rid == 1:
                self.controller1.set_path([])
            else:
                self.controller2.set_path([])

    def _execute_collision_avoidance(self, rid, my_pose, other_pose, cmd_pub):
        angle_to_other = math.atan2(other_pose[1]-my_pose[1], other_pose[0]-my_pose[0])
        my_yaw = my_pose[2]
        rel_angle = normalize_angle(angle_to_other - my_yaw)

        cmd = Twist()
        if abs(rel_angle) < math.pi / 2:
            cmd.angular.z = 0.6 if rel_angle > 0 else -0.6
            cmd.linear.x = 0.05
        else:
            cmd.linear.x = 0.15
        cmd_pub.publish(cmd)

    def _apply_safety_limits(self, rid):
        """Cap this robot's speed for the next command while the other robot
        is within safety_slowdown_distance but not yet close enough to need
        full collision-avoidance maneuvering. Previously a no-op stub, so the
        `elif dist < safety_slowdown_distance` branch in _control_robot did
        nothing at all; consumed once by _enforce_velocity_limits below."""
        setattr(self, f'_safety_slowdown_{rid}', True)

    def _enforce_velocity_limits(self, cmd, rid):
        max_lin = self.params.max_linear_speed
        max_ang = self.params.max_angular_speed
        if getattr(self, f'_safety_slowdown_{rid}', False):
            max_lin *= 0.5
            setattr(self, f'_safety_slowdown_{rid}', False)
        cmd.linear.x = clamp_value(cmd.linear.x, -max_lin, max_lin)
        cmd.linear.y = clamp_value(cmd.linear.y, -max_lin, max_lin)
        cmd.angular.z = clamp_value(cmd.angular.z, -max_ang, max_ang)
        return cmd

    def _apply_acceleration_limits(self, cmd, rid):
        history = self.vel_history1 if rid == 1 else self.vel_history2
        if len(history) < 2:
            return cmd
        last_speed = history[-1][0]
        last_ang = history[-1][1]
        dt = self.params.control_dt
        max_lin_acc = getattr(self.params, 'max_acceleration_linear', 1.0)
        max_ang_acc = getattr(self.params, 'max_acceleration_angular', 2.0)
        desired_speed = cmd.linear.x
        desired_ang = cmd.angular.z
        speed_diff = desired_speed - last_speed
        ang_diff = desired_ang - last_ang
        cmd.linear.x = last_speed + clamp_value(speed_diff, -max_lin_acc*dt, max_lin_acc*dt)
        cmd.angular.z = last_ang + clamp_value(ang_diff, -max_ang_acc*dt, max_ang_acc*dt)
        return cmd

    def _publish_cmd_vel(self, pub, lin, ang):
        cmd = Twist()
        cmd.linear.x = float(lin)
        cmd.angular.z = float(ang)
        pub.publish(cmd)

    def _mark_frontier_success(self, goal):
        g = (round(goal[0], 2), round(goal[1], 2))
        if g in self.frontier_attempts:
            del self.frontier_attempts[g]
        self.frontier_timestamps[g] = self._now()

    def _mark_frontier_failure(self, goal):
        g = (round(goal[0], 2), round(goal[1], 2))
        self.frontier_attempts[g] = self.frontier_attempts.get(g, 0) + 1
        if self.frontier_attempts[g] >= self.max_frontier_attempts:
            self.frontier_blacklist.add(g)
            self.get_logger().info(f'Blacklisted frontier {g} after {self.max_frontier_attempts} failures')

    def _prune_blacklist(self):
        now = self._now()
        if now - self.last_blacklist_prune < self.blacklist_prune_interval:
            return
        self.last_blacklist_prune = now
        to_remove = []
        for g in self.frontier_blacklist:
            if g in self.frontier_timestamps:
                if now - self.frontier_timestamps[g] > self.blacklist_prune_interval:
                    to_remove.append(g)
        for g in to_remove:
            self.frontier_blacklist.discard(g)
            if g in self.frontier_attempts:
                del self.frontier_attempts[g]

    def coordination_callback(self):
        if self.exploration_complete or self.emergency_stop_triggered:
            return

        if self.robot1_goal is None or self.robot2_goal is None:
            return

        try:
            should_swap = self.coordinator.should_swap_goals(
                self.robot1_state, self.robot2_state,
                self.robot1_goal, self.robot2_goal
            )
            if should_swap:
                self.get_logger().info('Swapping goals for efficiency')
                self.robot1_goal, self.robot2_goal = self.robot2_goal, self.robot1_goal
                self.robot1_state.set_goal(self.robot1_goal)
                self.robot2_state.set_goal(self.robot2_goal)
                self.controller1.set_goal(self.robot1_goal)
                self.controller2.set_goal(self.robot2_goal)
                self.controller1.set_path([])
                self.controller2.set_path([])
                self.goal_assignment_time = self._now()
        except Exception as e:
            self.get_logger().debug(f'Coordination error: {str(e)}')

        if self.robot1_mode in [RobotMode.IDLE, RobotMode.COMPLETE] and self.robot2_mode not in [RobotMode.IDLE, RobotMode.COMPLETE]:
            self._reassign_idle_robot(1, self.robot1_state)
        elif self.robot2_mode in [RobotMode.IDLE, RobotMode.COMPLETE] and self.robot1_mode not in [RobotMode.IDLE, RobotMode.COMPLETE]:
            self._reassign_idle_robot(2, self.robot2_state)

        if self._now() - self.goal_assignment_time > self.frontier_timeout:
            self.get_logger().warn('Goal assignment timeout, reassigning')
            self.controller1.set_path([])
            self.controller2.set_path([])
            self.goal_assignment_time = self._now()

    def _reassign_idle_robot(self, rid, robot_state):
        if not self.frontiers:
            return
        pose = robot_state.get_pose()
        if pose is None:
            return
        best = min(self.frontiers, key=lambda f: math.hypot(f[0]-pose[0], f[1]-pose[1]))
        self._set_robot_goal(rid, best)

    def merge_callback(self):
        if self.local_map1 is None or self.local_map2 is None:
            return

        try:
            transform, fitness = self.map_merger.align_maps(self.local_map1, self.local_map2)
            self.map_transform_1to2 = transform
            self.icp_fitness = fitness
            self.map_aligned = fitness > self.params.icp_fitness_threshold

            merged = self.map_merger.merge_maps(self.local_map1, self.local_map2, transform)
            if merged is not None:
                self.global_map = self._stamp_map(merged)
                self.grid_manager.set_global_map(self.global_map)
                self.global_map_pub.publish(self.global_map)
                self.merge_count += 1
                self.last_merge_time = self._now()
                self.merge_consecutive_failures = 0

                new_coverage = self._compute_coverage(self.global_map)
                if new_coverage > self.coverage_percent:
                    self.last_coverage_increase_time = self._now()
                self.coverage_percent = new_coverage
            else:
                self.merge_consecutive_failures += 1
        except Exception as e:
            self.get_logger().debug(f'Merge error: {str(e)}')
            self.merge_consecutive_failures += 1
            if self.merge_consecutive_failures >= self.max_merge_failures:
                self.get_logger().warn('Too many merge failures, publishing local map 1 as global')
                if self.local_map1 is not None:
                    self.global_map = self._stamp_map(self.local_map1)
                    self.global_map_pub.publish(self.global_map)

    def _stamp_map(self, grid_msg):
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = self.params.global_map_frame
        return grid_msg

    def _compute_coverage(self, grid_msg):
        if grid_msg is None or not hasattr(grid_msg, 'data') or grid_msg.data is None:
            return 0.0
        try:
            data = np.array(grid_msg.data, dtype=np.int8)
            # CONDITION 5: Count ALL known cells (both free >=0 and occupied >50)
            known = np.sum(data >= 0)
            total = len(data)
            return 100.0 * float(known) / float(total) if total > 0 else 0.0
        except Exception:
            return 0.0

    def recovery_callback(self):
        if self.exploration_complete or self.emergency_stop_triggered:
            return

        self._detect_stuck_robot(self.robot1_state, self.pos_history1, self.recovery1, 'robot_1')
        self._detect_stuck_robot(self.robot2_state, self.pos_history2, self.recovery2, 'robot_2')

    def _detect_stuck_robot(self, robot_state, history, recovery_state, robot_name):
        if recovery_state['active']:
            return

        pose = robot_state.get_pose()
        if pose is None or len(history) < 10:
            return

        recent = history[-1]
        older = history[max(0, len(history)-10)]
        dist = math.hypot(recent[0]-older[0], recent[1]-older[1])
        time_diff = recent[2] - older[2]

        # CONDITION 6: Check if robot has active goal via robot_state.get_goal()
        if robot_state.get_goal() is not None and dist < self.stuck_check_distance and time_diff > self.stuck_check_time:
            self.get_logger().warn(
                f'{robot_name} STUCK detected (dist={dist:.3f}m over {time_diff:.1f}s)'
            )
            recovery_state['active'] = True
            recovery_state['phase'] = RecoveryPhase.STOP
            recovery_state['start_time'] = self._now()
            recovery_state['last_change'] = self._now()
            recovery_state['attempt_count'] += 1
            robot_state.set_status('recovering')
            self.recovery_count += 1

            if robot_name == 'robot_1':
                self.robot1_mode = RobotMode.RECOVERING
            else:
                self.robot2_mode = RobotMode.RECOVERING

    def _execute_recovery1(self):
        self._execute_recovery(self.recovery1, self.robot1_state, self.cmd_vel1_pub, 'robot_1', 1)

    def _execute_recovery2(self):
        self._execute_recovery(self.recovery2, self.robot2_state, self.cmd_vel2_pub, 'robot_2', 2)

    def _execute_recovery(self, recovery_state, robot_state, cmd_pub, robot_name, rid):
        now = self._now()
        phase = recovery_state['phase']
        dt_phase = now - recovery_state['last_change']

        if phase == RecoveryPhase.STOP:
            self._publish_cmd_vel(cmd_pub, 0.0, 0.0)
            if dt_phase > 0.3:
                recovery_state['phase'] = RecoveryPhase.ROTATE_INPLACE
                recovery_state['last_change'] = now

        elif phase == RecoveryPhase.ROTATE_INPLACE:
            cmd = Twist()
            cmd.angular.z = self.params.recovery_rotation_speed * recovery_state['direction']
            cmd_pub.publish(cmd)
            if dt_phase > 2.5:
                recovery_state['phase'] = RecoveryPhase.BACKUP
                recovery_state['last_change'] = now

        elif phase == RecoveryPhase.BACKUP:
            cmd = Twist()
            cmd.linear.x = -self.params.recovery_backup_speed
            cmd_pub.publish(cmd)
            if dt_phase > 1.5:
                if recovery_state['attempt_count'] < 3:
                    recovery_state['phase'] = RecoveryPhase.FORWARD_PROBE
                else:
                    recovery_state['phase'] = RecoveryPhase.CLEAR_HISTORY
                recovery_state['last_change'] = now

        elif phase == RecoveryPhase.FORWARD_PROBE:
            cmd = Twist()
            cmd.linear.x = 0.2
            cmd.angular.z = 0.2 * recovery_state['direction']
            cmd_pub.publish(cmd)
            if dt_phase > 2.0:
                recovery_state['phase'] = RecoveryPhase.CLEAR_HISTORY
                recovery_state['last_change'] = now

        elif phase == RecoveryPhase.CLEAR_HISTORY:
            self._publish_cmd_vel(cmd_pub, 0.0, 0.0)
            recovery_state['active'] = False
            recovery_state['phase'] = RecoveryPhase.NONE
            robot_state.set_status('idle')
            robot_state.clear_goal()

            if rid == 1:
                self.robot1_goal = None
                self.robot1_mode = RobotMode.IDLE
                self.pos_history1.clear()
                self.controller1.clear_goal()
            else:
                self.robot2_goal = None
                self.robot2_mode = RobotMode.IDLE
                self.pos_history2.clear()
                self.controller2.clear_goal()

            self.get_logger().info(f'{robot_name} recovery sequence complete')

    def visualization_callback(self):
        try:
            pose1 = self._get_robot_pose(self.robot1_state, 'robot_1')
            pose2 = self._get_robot_pose(self.robot2_state, 'robot_2')

            if pose1 is not None and pose2 is not None:
                self.visualizer.publish_robot_poses(pose1, pose2)

            if self.global_map is not None:
                self.visualizer.publish_global_map(self.global_map)

            if self.frontiers:
                self.visualizer.publish_frontiers(self.frontiers)

            if self.robot1_goal is not None and self.robot2_goal is not None:
                self.visualizer.publish_goals(self.robot1_goal, self.robot2_goal)

            path1 = self.controller1.current_path()
            path2 = self.controller2.current_path()
            if path1:
                self.visualizer.publish_path(path1, 'robot_1')
            if path2:
                self.visualizer.publish_path(path2, 'robot_2')
        except Exception as e:
            self.get_logger().debug(f'Viz error: {str(e)}')

    def status_report_callback(self):
        elapsed = self._now() - self.exploration_start_time
        grid_for_stats = self.global_map if self.global_map is not None else self.grid_manager.get_combined_grid()
        occ_desc = 'known=0 free=0 occupied=0 unknown=0'
        if grid_for_stats is not None:
            data = np.array(grid_for_stats.data, dtype=np.int8)
            total = len(data)
            known = int(np.sum(data >= 0))
            occupied = int(np.sum(data > 50))
            free = known - occupied
            unknown = total - known
            occ_desc = f'known={known} free={free} occupied={occupied} unknown={unknown}'
        self.get_logger().info(
            f'[STATUS] Coverage={self.coverage_percent:.1f}% | {occ_desc} | '
            f'Maps={self.map_update_count} Plans={self.path_plan_count} '
            f'Merges={self.merge_count} Recovers={self.recovery_count} | '
            f'R1={self.robot1_mode.name} R2={self.robot2_mode.name} | '
            f'Time={elapsed:.1f}s Frontiers={self.frontier_count}'
        )

    def _trigger_emergency_stop(self):
        self.emergency_stop_triggered = True
        self.emergency_stop_time = self._now()
        self._publish_cmd_vel(self.cmd_vel1_pub, 0.0, 0.0)
        self._publish_cmd_vel(self.cmd_vel2_pub, 0.0, 0.0)
        self.get_logger().error('EMERGENCY STOP TRIGGERED')

    def destroy_node(self):
        self.get_logger().info('Shutting down SolutionNode')
        self._publish_cmd_vel(self.cmd_vel1_pub, 0.0, 0.0)
        self._publish_cmd_vel(self.cmd_vel2_pub, 0.0, 0.0)
        if self.global_map is not None:
            self._stamp_map(self.global_map)
            self.global_map_pub.publish(self.global_map)
        super().destroy_node()