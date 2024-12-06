import torch
import pytest
from model import MNISTNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = MNISTNet()
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, should be less than 20000"

def test_batch_norm_usage():
    model = MNISTNet()
    has_batch_norm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model should use BatchNormalization"

def test_dropout_usage():
    model = MNISTNet()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_gap_usage():
    model = MNISTNet()
    has_gap = any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules())
    assert has_gap, "Model should use Global Average Pooling"

def test_forward_pass():
    model = MNISTNet()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}" 