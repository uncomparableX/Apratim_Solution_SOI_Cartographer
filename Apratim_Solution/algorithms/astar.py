
import heapq
import math
import threading
from typing import List, Tuple, Callable, Any, Dict


class AStar:
    def __init__(self):
        self._lock = threading.Lock()

    def search(
        self,
        start: Any,
        goal: Any,
        neighbors_fn: Callable[[Any], List[Tuple[Any, float]]],
        heuristic_fn: Callable[[Any], float]
    ) -> List[Any]:
        with self._lock:
            open_set: List[Tuple[float, int, Any]] = []
            counter = 0
            heapq.heappush(open_set, (heuristic_fn(start), counter, start))
            came_from: Dict[Any, Any] = {}
            g_score: Dict[Any, float] = {start: 0.0}
            closed_set: set = set()
            while open_set:
                _, _, current = heapq.heappop(open_set)
                if current in closed_set:
                    continue
                if current == goal:
                    return self.reconstruct_path(came_from, current)
                closed_set.add(current)
                for neighbor, edge_cost in neighbors_fn(current):
                    if neighbor in closed_set:
                        continue
                    tentative_g = g_score[current] + edge_cost
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic_fn(neighbor)
                        counter += 1
                        heapq.heappush(open_set, (f_score, counter, neighbor))
            return []

    def reconstruct_path(self, came_from: Dict[Any, Any], current: Any) -> List[Any]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def path_cost(self, path: List[Any]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                total += math.hypot(
                    float(b[0]) - float(a[0]),
                    float(b[1]) - float(a[1])
                )
            else:
                total += 1.0
        return total

    def reset(self):
        pass