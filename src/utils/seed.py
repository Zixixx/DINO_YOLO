from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    # 固定常见随机源，降低不同运行之间的数据顺序和初始化差异。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
