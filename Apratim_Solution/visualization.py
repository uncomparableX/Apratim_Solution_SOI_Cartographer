
import math
import threading
from typing import List, Tuple, Optional, Dict, Any
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA, Header
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy


class Visualization:
    def __init__(self, params):
        self.params = params
        self.node = params.node
        self._lock = threading.Lock()
        self._marker_id = 0
        self._namespaces = {
            'robot': 'robot',
            'path': 'path',
            'frontier': 'frontier',
            'cluster': 'cluster',
            'selected': 'selected',
            'goal': 'goal',
            'text': 'text',
            'map': 'map'
        }
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.marker_pub = self.node.create_publisher(MarkerArray, '/visualization_markers', qos)
        self.text_pub = self.node.create_publisher(MarkerArray, '/visualization_text', qos)

    def _next_id(self) -> int:
        with self._lock:
            self._marker_id += 1
            return self._marker_id

    def _create_header(self, frame_id='map') -> Header:
        h = Header()
        h.stamp = self.node.get_clock().now().to_msg()
        h.frame_id = frame_id
        return h

    def _create_marker(self, ns: str, mtype: int, mid: int, frame_id='map', lifetime=0.0):
        m = Marker()
        m.header = self._create_header(frame_id)
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.lifetime.sec = int(lifetime)
        m.lifetime.nanosec = int((lifetime - int(lifetime)) * 1e9)
        m.pose.orientation.w = 1.0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        return m

    def _color(self, r, g, b, a=1.0):
        c = ColorRGBA()
        c.r = float(r)
        c.g = float(g)
        c.b = float(b)
        c.a = float(a)
        return c

    def publish_robot_poses(self, pose1: Tuple[float, float, float], pose2: Tuple[float, float, float]):
        if pose1 is None and pose2 is None:
            return
        markers = []
        if pose1 is not None:
            m = self._create_marker(self._namespaces['robot'], Marker.ARROW, self._next_id())
            m.pose.position.x = pose1[0]
            m.pose.position.y = pose1[1]
            m.pose.position.z = 0.1
            yaw = pose1[2]
            m.pose.orientation.z = math.sin(yaw * 0.5)
            m.pose.orientation.w = math.cos(yaw * 0.5)
            m.scale = Vector3(x=0.4, y=0.08, z=0.08)
            m.color = self._color(0.0, 0.0, 1.0)
            markers.append(m)
            t = self._create_marker(self._namespaces['robot'], Marker.TEXT_VIEW_FACING, self._next_id())
            t.pose.position.x = pose1[0]
            t.pose.position.y = pose1[1]
            t.pose.position.z = 0.5
            t.text = 'robot_1'
            t.scale.z = 0.2
            t.color = self._color(0.0, 0.0, 1.0)
            markers.append(t)
        if pose2 is not None:
            m = self._create_marker(self._namespaces['robot'], Marker.ARROW, self._next_id())
            m.pose.position.x = pose2[0]
            m.pose.position.y = pose2[1]
            m.pose.position.z = 0.1
            yaw = pose2[2]
            m.pose.orientation.z = math.sin(yaw * 0.5)
            m.pose.orientation.w = math.cos(yaw * 0.5)
            m.scale = Vector3(x=0.4, y=0.08, z=0.08)
            m.color = self._color(0.0, 0.4, 1.0)
            markers.append(m)
            t = self._create_marker(self._namespaces['robot'], Marker.TEXT_VIEW_FACING, self._next_id())
            t.pose.position.x = pose2[0]
            t.pose.position.y = pose2[1]
            t.pose.position.z = 0.5
            t.text = 'robot_2'
            t.scale.z = 0.2
            t.color = self._color(0.0, 0.4, 1.0)
            markers.append(t)
        if markers:
            self.marker_pub.publish(MarkerArray(markers=markers))

    def publish_robot_pose(self, robot_name: str, pose: Tuple[float, float, float]):
        if pose is None:
            return
        markers = []
        m = self._create_marker(self._namespaces['robot'], Marker.ARROW, self._next_id())
        m.pose.position.x = pose[0]
        m.pose.position.y = pose[1]
        m.pose.position.z = 0.1
        yaw = pose[2]
        m.pose.orientation.z = math.sin(yaw * 0.5)
        m.pose.orientation.w = math.cos(yaw * 0.5)
        m.scale = Vector3(x=0.4, y=0.08, z=0.08)
        m.color = self._color(0.0, 0.0, 1.0)
        markers.append(m)
        t = self._create_marker(self._namespaces['robot'], Marker.TEXT_VIEW_FACING, self._next_id())
        t.pose.position.x = pose[0]
        t.pose.position.y = pose[1]
        t.pose.position.z = 0.5
        t.text = robot_name
        t.scale.z = 0.2
        t.color = self._color(0.0, 0.0, 1.0)
        markers.append(t)
        self.marker_pub.publish(MarkerArray(markers=markers))

    def publish_path(self, path: List[Tuple[float, float]], robot_name: str = 'robot'):
        if not path:
            return
        markers = []
        m = self._create_marker(self._namespaces['path'], Marker.LINE_STRIP, self._next_id())
        m.scale.x = 0.03
        m.color = self._color(0.0, 1.0, 0.0)
        for wx, wy in path:
            p = Point()
            p.x = float(wx)
            p.y = float(wy)
            p.z = 0.05
            m.points.append(p)
        markers.append(m)
        self.marker_pub.publish(MarkerArray(markers=markers))

    def publish_frontiers(self, frontiers: List[Tuple[float, float]]):
        if not frontiers:
            return
        markers = []
        m = self._create_marker(self._namespaces['frontier'], Marker.POINTS, self._next_id())
        m.scale.x = 0.08
        m.scale.y = 0.08
        m.color = self._color(1.0, 1.0, 0.0)
        for wx, wy in frontiers:
            p = Point()
            p.x = float(wx)
            p.y = float(wy)
            p.z = 0.05
            m.points.append(p)
        markers.append(m)
        self.marker_pub.publish(MarkerArray(markers=markers))

    def publish_goals(self, goal1: Optional[Tuple[float, float]], goal2: Optional[Tuple[float, float]]):
        markers = []
        if goal1 is not None:
            m = self._create_marker(self._namespaces['goal'], Marker.SPHERE, self._next_id())
            m.pose.position.x = goal1[0]
            m.pose.position.y = goal1[1]
            m.pose.position.z = 0.05
            m.scale = Vector3(x=0.15, y=0.15, z=0.15)
            m.color = self._color(1.0, 0.0, 1.0)
            markers.append(m)
        if goal2 is not None:
            m = self._create_marker(self._namespaces['goal'], Marker.SPHERE, self._next_id())
            m.pose.position.x = goal2[0]
            m.pose.position.y = goal2[1]
            m.pose.position.z = 0.05
            m.scale = Vector3(x=0.15, y=0.15, z=0.15)
            m.color = self._color(1.0, 0.0, 1.0)
            markers.append(m)
        if markers:
            self.marker_pub.publish(MarkerArray(markers=markers))

    def publish_global_map(self, global_map: OccupancyGrid):
        if global_map is None:
            return
        markers = []
        t = self._create_marker(self._namespaces['map'], Marker.TEXT_VIEW_FACING, self._next_id())
        t.pose.position.x = float(self.params.map_origin_x)
        t.pose.position.y = float(self.params.map_origin_y) + 0.5
        t.pose.position.z = 1.0
        t.text = f'Global Map: {global_map.info.width}x{global_map.info.height}'
        t.scale.z = 0.15
        t.color = self._color(1.0, 1.0, 1.0)
        markers.append(t)
        self.text_pub.publish(MarkerArray(markers=markers))

    def publish_clusters(self, clusters: List[List[Tuple[float, float]]]):
        if not clusters:
            return
        markers = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            m = self._create_marker(self._namespaces['cluster'], Marker.SPHERE, self._next_id())
            m.pose.position.x = cx
            m.pose.position.y = cy
            m.pose.position.z = 0.1
            m.scale = Vector3(x=0.12, y=0.12, z=0.12)
            m.color = self._color(0.0, 1.0, 1.0)
            markers.append(m)
        if markers:
            self.marker_pub.publish(MarkerArray(markers=markers))

    def publish_selected_frontier(self, frontier: Optional[Tuple[float, float]]):
        if frontier is None:
            return
        markers = []
        m = self._create_marker(self._namespaces['selected'], Marker.SPHERE, self._next_id())
        m.pose.position.x = frontier[0]
        m.pose.position.y = frontier[1]
        m.pose.position.z = 0.1
        m.scale = Vector3(x=0.18, y=0.18, z=0.18)
        m.color = self._color(1.0, 0.0, 0.0)
        markers.append(m)
        self.marker_pub.publish(MarkerArray(markers=markers))

    def publish_global_map_status(self, statistics: Dict[str, Any]):
        if not statistics:
            return
        markers = []
        t = self._create_marker(self._namespaces['text'], Marker.TEXT_VIEW_FACING, self._next_id())
        t.pose.position.x = float(self.params.map_origin_x)
        t.pose.position.y = float(self.params.map_origin_y) + 1.0
        t.pose.position.z = 1.0
        t.text = f'Merges: {statistics.get("merges", 0)} Aligned: {statistics.get("aligned", False)}'
        t.scale.z = 0.15
        t.color = self._color(1.0, 1.0, 1.0)
        markers.append(t)
        self.text_pub.publish(MarkerArray(markers=markers))

    def publish_text(self, text: str):
        if not text:
            return
        markers = []
        t = self._create_marker(self._namespaces['text'], Marker.TEXT_VIEW_FACING, self._next_id())
        t.pose.position.x = float(self.params.map_origin_x)
        t.pose.position.y = float(self.params.map_origin_y) + 1.5
        t.pose.position.z = 1.0
        t.text = text
        t.scale.z = 0.15
        t.color = self._color(1.0, 1.0, 1.0)
        markers.append(t)
        self.text_pub.publish(MarkerArray(markers=markers))

    def publish_all(self, robot_states, paths, frontiers, clusters, selected_frontier, merger_statistics):
        if robot_states:
            for name, state in robot_states.items():
                pose = state.get_pose() if hasattr(state, 'get_pose') else None
                if pose is not None:
                    self.publish_robot_pose(name, pose)
        if paths:
            for robot_name, path in paths.items():
                if path:
                    self.publish_path(path, robot_name)
        if frontiers:
            self.publish_frontiers(frontiers)
        if clusters:
            self.publish_clusters(clusters)
        if selected_frontier is not None:
            self.publish_selected_frontier(selected_frontier)
        if merger_statistics:
            self.publish_global_map_status(merger_statistics)

    def clear(self):
        markers = []
        for ns in self._namespaces.values():
            m = self._create_marker(ns, Marker.CUBE, 0)
            m.action = Marker.DELETEALL
            markers.append(m)
        self.marker_pub.publish(MarkerArray(markers=markers))
        self.text_pub.publish(MarkerArray(markers=markers))

    def reset(self):
        with self._lock:
            self._marker_id = 0
        self.clear()