# SOI Cartographer

## Team Solution

Competition submission for the SOI Cartographer Autonomous Multi-Robot Mapping Challenge.

---

## Features

- Multi-Robot SLAM
- Cooperative Exploration
- Frontier Detection
- Frontier Assignment
- Occupancy Grid Mapping
- ICP Map Alignment
- Occupancy Grid Merging
- A* Navigation
- Obstacle Avoidance
- Information Gain Frontier Selection
- Multi-Robot Communication
- Dynamic Task Allocation
- Recovery Behaviors

---

## Folder Structure

```
launch/
config/
resource/
team_solution/
```

---

## Build

```bash
cd ~/cartographer

colcon build --symlink-install

source install/setup.bash
```

---

## Launch

Terminal 1

```bash
ros2 launch challenge_bridge competition.launch.py
```

Terminal 2

```bash
ros2 launch team_solution solution.launch.py
```

---

## Algorithms

- Frontier Exploration
- Information Gain
- A*
- Occupancy Grid Mapping
- ICP
- Map Correlation
- Cooperative Task Allocation

---

## Dependencies

- Ubuntu 24.04
- ROS2 Jazzy
- Gazebo Harmonic
- Python 3.12