import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mlp import MLP


def test_mlp_init():
    model = MLP(input_dim=8, output_dim=4, hidden_dims=(128, 128))

    assert isinstance(model, MLP)


def test_mlp_forward_single_input_shape():
    model = MLP(input_dim=8, output_dim=4, hidden_dims=(128, 128))

    x = torch.randn(8)
    y = model(x)

    assert y.shape == (4,)


def test_mlp_forward_batch_input_shape():
    model = MLP(input_dim=8, output_dim=4, hidden_dims=(128, 128))

    x = torch.randn(32, 8)
    y = model(x)

    assert y.shape == (32, 4)


def test_mlp_output_dtype():
    model = MLP(input_dim=8, output_dim=4, hidden_dims=(128, 128))

    x = torch.randn(16, 8)
    y = model(x)

    assert y.dtype == torch.float32


def test_mlp_invalid_input_dim():
    with pytest.raises(ValueError, match="input_dim must be > 0"):
        MLP(input_dim=0, output_dim=4, hidden_dims=(128, 128))


def test_mlp_invalid_output_dim():
    with pytest.raises(ValueError, match="output_dim must be > 0"):
        MLP(input_dim=8, output_dim=0, hidden_dims=(128, 128))


def test_mlp_empty_hidden_dims():
    with pytest.raises(ValueError, match="hidden_dims must contain at least one layer"):
        MLP(input_dim=8, output_dim=4, hidden_dims=())


def test_mlp_invalid_hidden_dim_value():
    with pytest.raises(ValueError, match="all hidden_dims must be > 0"):
        MLP(input_dim=8, output_dim=4, hidden_dims=(128, -1))


def test_mlp_has_parameters():
    model = MLP(input_dim=8, output_dim=4, hidden_dims=(128, 128))

    num_params = sum(p.numel() for p in model.parameters())

    assert num_params > 0


def test_mlp_forward_is_deterministic_for_same_input_and_weights():
    model = MLP(input_dim=8, output_dim=4, hidden_dims=(128, 128))

    x = torch.randn(8)

    y1 = model(x)
    y2 = model(x)

    assert torch.allclose(y1, y2)