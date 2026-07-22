import numpy as np
import math
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header

class Mapper:
    def __init__(self, params, robot_name):
        self.params = params
        self.robot_name = robot_name
        self.resolution = float(params.map_resolution)
        self.width = int(params.map_width)
        self.height = int(params.map_height)
        self.origin_x = float(params.map_origin_x)
        self.origin_y = float(params.map_origin_y)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.l_occ = 1.2
        self.l_free = -0.7
        self.l_max = 5.0
        self.l_min = -5.0
        self.occ_threshold = 0.5
        self.free_threshold = -0.5
        self.update_count = 0
        self.ray_count = 0
        self.lidar_offset_x = float(getattr(params, 'lidar_offset_x', 0.2))

    def world_to_grid(self, wx, wy):
        gx = math.floor((wx - self.origin_x) / self.resolution)
        gy = math.floor((wy - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = gx * self.resolution + self.origin_x + self.resolution * 0.5
        wy = gy * self.resolution + self.origin_y + self.resolution * 0.5
        return wx, wy

    def bresenham(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        pts = []
        while True:
            pts.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return pts

    def update_ray(self, x0, y0, x1, y1, is_hit=True):
        w = self.width
        h = self.height
        lo = self.log_odds
        l_free = self.l_free
        l_occ = self.l_occ
        l_min = self.l_min
        l_max = self.l_max
        cells = self.bresenham(x0, y0, x1, y1)
        n = len(cells)
        if n == 0:
            return
            
        for i in range(n - 1):
            cx, cy = cells[i]
            if 0 <= cx < w and 0 <= cy < h:
                lo[cy, cx] += l_free
                if lo[cy, cx] < l_min:
                    lo[cy, cx] = l_min
                    
        cx, cy = cells[-1]
        if 0 <= cx < w and 0 <= cy < h:
            if is_hit:
                lo[cy, cx] += l_occ
                if lo[cy, cx] > l_max:
                    lo[cy, cx] = l_max
            else:
                lo[cy, cx] += l_free
                if lo[cy, cx] < l_min:
                    lo[cy, cx] = l_min

    def build_ros_map(self):
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.params.node.get_clock().now().to_msg()
        grid.header.frame_id = self.params.global_map_frame
        grid.info = MapMetaData()
        grid.info.resolution = self.resolution
        grid.info.width = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        
        occ = np.zeros((self.height, self.width), dtype=np.int8)
        occ[self.log_odds > self.occ_threshold] = 100
        occ[self.log_odds < self.free_threshold] = 0
        mask = (self.log_odds >= self.free_threshold) & (self.log_odds <= self.occ_threshold)
        occ[mask] = -1
        grid.data = occ.ravel().tolist()
        return grid

    def reset(self):
        self.log_odds.fill(0.0)
        self.update_count = 0
        self.ray_count = 0

    def statistics(self):
        total = self.width * self.height
        occupied = int(np.count_nonzero(self.log_odds > self.occ_threshold))
        free = int(np.count_nonzero(self.log_odds < self.free_threshold))
        known = occupied + free
        return {
            'total_cells': total,
            'known_cells': known,
            'occupied_cells': occupied,
            'free_cells': free,
            'updates': self.update_count,
            'rays': self.ray_count
        }

    def update(self, scan, pose):
        x, y, yaw = pose
        sensor_x = x + self.lidar_offset_x * math.cos(yaw)
        sensor_y = y + self.lidar_offset_x * math.sin(yaw)
        robot_gx, robot_gy = self.world_to_grid(sensor_x, sensor_y)
        
        if not (0 <= robot_gx < self.width and 0 <= robot_gy < self.height):
            return self.build_ros_map()
            
        ranges = np.array(scan.ranges, dtype=np.float32)
        n = len(ranges)
        if n == 0:
            return self.build_ros_map()
            
        # FIX: Process 'inf' rays to aggressively map open space!
        inf_mask = np.isinf(ranges) | np.isnan(ranges) | (ranges >= scan.range_max)
        ranges[inf_mask] = scan.range_max - 0.1
        
        angles = scan.angle_min + np.arange(n, dtype=np.float32) * scan.angle_increment
        cos_a = np.cos(angles + yaw)
        sin_a = np.sin(angles + yaw)
        
        valid = (ranges > scan.range_min)
        if not np.any(valid):
            return self.build_ros_map()
            
        valid_ranges = ranges[valid]
        valid_cos = cos_a[valid]
        valid_sin = sin_a[valid]
        is_hit_array = ~inf_mask[valid]
        
        end_x = sensor_x + valid_ranges * valid_cos
        end_y = sensor_y + valid_ranges * valid_sin
        end_gx = np.floor((end_x - self.origin_x) / self.resolution).astype(np.int32)
        end_gy = np.floor((end_y - self.origin_y) / self.resolution).astype(np.int32)
        
        for i in range(len(valid_ranges)):
            self.update_ray(robot_gx, robot_gy, end_gx[i], end_gy[i], is_hit_array[i])
            
        self.update_count += 1
        self.ray_count += len(valid_ranges)
        return self.build_ros_map()