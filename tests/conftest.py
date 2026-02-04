"""Pytest configuration and fixtures."""
import torch
import pytest
import numpy as np

@pytest.fixture
def device():
    """Return the device to use for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    return 42
