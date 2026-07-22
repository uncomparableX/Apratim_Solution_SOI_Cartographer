class Parameters:
    def __init__(self, node):
        self.node = node

        def get_param(name, default):
            if not node.has_parameter(name):
                node.declare_parameter(name, default)
            return node.get_parameter(name).value

        self.mapping_rate = get_param("mapping_rate", 5.0)
        self.exploration_rate = get_param("exploration_rate", 1.0)
        self.control_rate = get_param("control_rate", 10.0)
        self.coordination_rate = get_param("coordination_rate", 0.5)
        self.merge_rate = get_param("merge_rate", 0.2)
        self.recovery_rate = get_param("recovery_rate", 2.0)
        self.viz_rate = get_param("viz_rate", 1.0)

        self.mapping_dt = 1.0 / self.mapping_rate if self.mapping_rate > 0.0 else 0.2
        self.control_dt = 1.0 / self.control_rate if self.control_rate > 0.0 else 0.1
        self.exploration_dt = 1.0 / self.exploration_rate if self.exploration_rate > 0.0 else 1.0
        self.merge_dt = 1.0 / self.merge_rate if self.merge_rate > 0.0 else 5.0
        self.recovery_dt = 1.0 / self.recovery_rate if self.recovery_rate > 0.0 else 0.5
        self.coordination_dt = 1.0 / self.coordination_rate if self.coordination_rate > 0.0 else 2.0
        self.viz_dt = 1.0 / self.viz_rate if self.viz_rate > 0.0 else 1.0

        self.map_resolution = get_param("map_resolution", 0.05)
        self.map_width = get_param("map_width", 400)
        self.map_height = get_param("map_height", 400)
        self.map_origin_x = get_param("map_origin_x", -10.0)
        self.map_origin_y = get_param("map_origin_y", -10.0)
        self.global_map_frame = get_param("global_map_frame", "map")

        self.max_linear_speed = get_param("max_linear_speed", 0.5)
        self.max_angular_speed = get_param("max_angular_speed", 1.0)
        self.robot_radius = get_param("robot_radius", 0.25)
        self.safety_margin = get_param("safety_margin", 0.15)

        self.goal_tolerance = get_param("goal_tolerance", 0.25)
        self.path_lookahead = get_param("path_lookahead", 0.8)
        self.path_resolution = get_param("path_resolution", 0.1)
        self.max_path_length = get_param("max_path_length", 50.0)
        self.replan_threshold = get_param("replan_threshold", 5)

        self.frontier_min_size = get_param("frontier_min_size", 5)
        self.frontier_cluster_distance = get_param("frontier_cluster_distance", 0.5)
        self.frontier_min_distance = get_param("frontier_min_distance", 0.5)

        self.icp_max_iterations = get_param("icp_max_iterations", 50)
        self.icp_tolerance = get_param("icp_tolerance", 1e-6)
        self.icp_fitness_threshold = get_param("icp_fitness_threshold", 0.7)

        self.stuck_timeout = get_param("stuck_timeout", 10.0)

        self.scan_buffer_timeout = get_param("scan_buffer_timeout", 2.0)
        self.odom_buffer_timeout = get_param("odom_buffer_timeout", 2.0)

        self.max_acceleration_linear = get_param("max_acceleration_linear", 1.0)
        self.max_acceleration_angular = get_param("max_acceleration_angular", 2.0)

        self.recovery_rotation_speed = get_param("recovery_rotation_speed", 0.8)
        self.recovery_backup_speed = get_param("recovery_backup_speed", 0.25)

        self.min_scan_range = get_param("min_scan_range", 0.1)
        self.max_scan_range = get_param("max_scan_range", 30.0)

        self.use_tf_poses = get_param("use_tf_poses", False)
        self.tf_timeout = get_param("tf_timeout", 0.1)

        self.unknown_penalty = get_param("unknown_penalty", 1.5)

        # Forward offset of the LiDAR from base_link, along the robot's own
        # x-axis (see challenge_bridge/models/robot.sdf: lidar_link is at
        # <pose>0.2 0 0.1 0 0 0</pose> relative to base_link, and the static
        # transform publishes the same 0.2m offset). Overridable in case a
        # different robot model is ever used, but the competition's provided
        # robot model is fixed by the organisers.
        self.lidar_offset_x = get_param("lidar_offset_x", 0.2)