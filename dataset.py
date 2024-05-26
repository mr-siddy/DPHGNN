import dhg
from dhg.datapipe import (
    load_from_pickle,
    norm_ft,
    to_tensor,
    to_long_tensor,
    to_bool_tensor,
)
from dhg.data import BaseData
from typing import Optional
from functools import partial
import numpy as np
import pandas as pd

class CoRTO(BaseData):
    r"""
    The content of the CoRTO dataset includes the following:

    - ``num_classes``: The number of classes: :math:`2`.
    - ``num_vertices``: The number of vertices: :math:`66,790`.
    - ``num_edges``: The number of edges: :math:`27528`.
    - ``dim_features``: The dimension of features: :math:`10`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(66, 790 \times 10)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`27528`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(66,790, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(66,790, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(66,790, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(66,790, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """
    def __init__(self, data_root: "/home/siddy/DPHGNN/data_root/"):
        super().__init__("CoRTO", data_root)
        self._content = {
            "num_classes": 2,
            "num_vertices": 66790,
            "num_edges": 27528,
            "dim_features": 10,
            "features": {
                "upon": [{"filename": "/home/siddy/DPHGNN/data_root/name/filename/features.pkl", 
                          "md5": "f5070e8810bc0e6577ac0ff4022f5bfb"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "/home/siddy/DPHGNN/data_root/name/filename/edge_list.pkl", 
                          "md5": "4a0caa73e265e3b87697efab42da486a"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "/home/siddy/DPHGNN/data_root/name/filename/labels.pkl", 
                          "md5": "931f196b3f97d5e67ea684142421d541"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [{"filename": "/home/siddy/DPHGNN/data_root/name/filename/train_mask.pkl",
                          "md5": "971ea98e98112731efef799b9deac968"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [{"filename": "/home/siddy/DPHGNN/data_root/name/filename/val_mask.pkl", 
                          "md5": "e30b2503523ef31d68e38d709eacb9c3"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [{"filename": "/home/siddy/DPHGNN/data_root/name/filename/test_mask.pkl", 
                          "md5": "c275e26c6321b58f7f4a00b0ba997902"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }

