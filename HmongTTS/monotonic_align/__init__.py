
import numpy as np
import torch

# Try to import the C extension, fall back to pure Python if not available
try:
    from .core import maximum_path_c
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


def maximum_path_numpy(paths, values, t_ys, t_xs):
    """Pure Python/NumPy fallback for maximum_path_c"""
    b = values.shape[0]
    for i in range(b):
        t_y = t_ys[i]
        t_x = t_xs[i]

        # Initialize
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = -1e9 if x > 0 else 0
                else:
                    v_cur = values[i, y-1, x]
                if x == 0:
                    v_prev = 0 if y == 0 else -1e9
                else:
                    v_prev = values[i, y-1, x-1]
                values[i, y, x] += max(v_cur, v_prev)

        # Backtrack
        index = t_x - 1
        for y in range(t_y - 1, -1, -1):
            paths[i, y, index] = 1
            if index > 0 and (index == y or values[i, y-1, index] < values[i, y-1, index-1]):
                index -= 1


def maximum_path(neg_cent, mask):
    device = neg_cent.device
    dtype = neg_cent.dtype

    # 1. Move data to CPU for the C++ engine
    # The C++ code requires float32 input for costs and int32 for output
    neg_cent_cpu = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent_cpu.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)

    # 2. Run the Alignment Search (C++ if available, else pure Python)
    if USE_CYTHON:
        maximum_path_c(path, neg_cent_cpu, t_t_max, t_s_max)
    else:
        maximum_path_numpy(path, neg_cent_cpu, t_t_max, t_s_max)

    # 3. Move result back to GPU/MPS and CONVERT TO FLOAT
    # The network needs Floats (0.0, 1.0) not Integers (0, 1)
    return torch.from_numpy(path).to(device=device, dtype=torch.float32)
