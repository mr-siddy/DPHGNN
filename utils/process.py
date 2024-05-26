import torch
import numpy as np

def _format_inputs(y_true, y_pred):
    assert y_true.dim() == 1
    assert y_pred.dim() in (1, 2)
    y_true = y_true.cpu().detach()
    if y_pred.dim() == 2:
        y_pred = y_pred.argmax(dim=1)
    y_pred = y_pred.cpu().detach()
    assert y_true.shape == y_pred.shape
    return (y_true, y_pred)

