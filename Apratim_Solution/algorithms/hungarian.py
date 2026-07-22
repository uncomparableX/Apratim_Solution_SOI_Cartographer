
import numpy as np
import threading


class HungarianAlgorithm:
    def __init__(self):
        self._lock = threading.Lock()

    def solve(self, cost_matrix):
        with self._lock:
            if cost_matrix is None:
                return []

            C = np.asarray(cost_matrix, dtype=np.float64)

            if C.size == 0:
                return []

            if C.ndim == 1:
                C = C.reshape(1, -1)
            elif C.ndim != 2:
                return []

            n_rows, n_cols = C.shape

            if n_rows == 0 or n_cols == 0:
                return []

            finite_mask = np.isfinite(C)
            if np.any(finite_mask):
                max_finite = float(np.max(C[finite_mask]))
            else:
                max_finite = 0.0
            large_value = max_finite + 1e6
            C = np.where(finite_mask, C, large_value)

            n = max(n_rows, n_cols)
            padded = np.zeros((n, n), dtype=np.float64)
            padded[:n_rows, :n_cols] = C

            assignment = self._hungarian_square(padded)

            return assignment[:n_rows]

    def _hungarian_square(self, C):
        n = len(C)
        u = [0.0] * (n + 1)
        v = [0.0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float('inf')] * (n + 1)
            used = [False] * (n + 1)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = C[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        result = [-1] * n
        for j in range(1, n + 1):
            if p[j] != 0:
                result[p[j] - 1] = j - 1

        return result

    def reset(self):
        pass