

"""
occupancy_grid.py

Global Occupancy Grid Manager

Stores:
- Robot 1 local map
- Robot 2 local map
- Global merged map
"""

from __future__ import annotations

import copy
import threading
import numpy as np

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose


class OccupancyGridManager:

    def __init__(self, params):

        self.params = params

        self.lock = threading.RLock()

        ############################################################
        # Robot Maps
        ############################################################

        self.robot_maps = {
            "robot_1": None,
            "robot_2": None,
        }

        ############################################################
        # Global Map
        ############################################################

        self.global_map = None

        ############################################################
        # Grid Information
        ############################################################

        self.width = params.map_width
        self.height = params.map_height
        self.resolution = params.map_resolution

        self.origin_x = params.map_origin_x
        self.origin_y = params.map_origin_y

        ############################################################

        self.grid = np.full(

            (self.height, self.width),

            -1,

            dtype=np.int8,

        )

    ############################################################
    # Robot Map Storage
    ############################################################

    def update_grid(

        self,

        robot_name,

        occupancy_grid,

    ):

        with self.lock:

            self.robot_maps[robot_name] = copy.deepcopy(

                occupancy_grid

            )

    ############################################################

    def get_robot_grid(

        self,

        robot_name,

    ):

        with self.lock:

            if robot_name not in self.robot_maps:

                return None

            return copy.deepcopy(

                self.robot_maps[robot_name]

            )

    ############################################################
    # Global Map
    ############################################################

    def set_global_map(

        self,

        occupancy_grid,

    ):

        with self.lock:

            self.global_map = copy.deepcopy(

                occupancy_grid

            )

    ############################################################

    def get_global_map(self):

        with self.lock:

            if self.global_map is None:

                return None

            return copy.deepcopy(

                self.global_map

            )

    ############################################################
    # Combined Grid
    ############################################################

    def get_combined_grid(self):

        with self.lock:

            if self.global_map is not None:

                return copy.deepcopy(

                    self.global_map

                )

            if (

                self.robot_maps["robot_1"]

                is not None

            ):

                return copy.deepcopy(

                    self.robot_maps["robot_1"]

                )

            if (

                self.robot_maps["robot_2"]

                is not None

            ):

                return copy.deepcopy(

                    self.robot_maps["robot_2"]

                )

            return None

    ############################################################
    # Empty Grid
    ############################################################

    def create_empty_grid(self):

        msg = OccupancyGrid()

        msg.info.width = self.width

        msg.info.height = self.height

        msg.info.resolution = self.resolution

        msg.info.origin = Pose()

        msg.info.origin.position.x = self.origin_x

        msg.info.origin.position.y = self.origin_y

        msg.data = [-1] * (

            self.width * self.height

        )

        return msg

    ############################################################
    # Reset
    ############################################################

    def reset(self):

        with self.lock:

            self.robot_maps = {

                "robot_1": None,

                "robot_2": None,

            }

            self.global_map = None

            self.grid.fill(-1)

    ############################################################
    # Statistics
    ############################################################

    def statistics(self):

        grid = self.get_combined_grid()

        if grid is None:

            return {

                "known": 0,

                "unknown": self.width * self.height,

                "occupied": 0,

                "free": 0,

                "coverage": 0.0,

            }

        data = np.array(

            grid.data,

            dtype=np.int16,

        )

        known = np.sum(data >= 0)

        occupied = np.sum(data > 50)

        free = np.sum(

            (data >= 0)

            &

            (data <= 50)

        )

        unknown = np.sum(

            data < 0

        )

        coverage = (

            100.0 * known

            /

            len(data)

        )

        return {

            "known": int(known),

            "unknown": int(unknown),

            "occupied": int(occupied),

            "free": int(free),

            "coverage": float(coverage),

        }

    ############################################################

    def print_statistics(self):

        stats = self.statistics()

        print()

        print("=" * 50)

        print("Occupancy Grid")

        print("=" * 50)

        print(f"Known      : {stats['known']}")

        print(f"Unknown    : {stats['unknown']}")

        print(f"Occupied   : {stats['occupied']}")

        print(f"Free       : {stats['free']}")

        print(f"Coverage   : {stats['coverage']:.2f}%")

        print("=" * 50)