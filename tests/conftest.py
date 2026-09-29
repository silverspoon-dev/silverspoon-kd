"""
Root conftest.py for silverspoon-kd tests.

Note: When running tests, ensure silverspoon-kd is installed in editable mode:
    pip install -e /path/to/silverspoon-kd

Device selection:
    pytest tests/                   # Runs on CPU (default)
    pytest tests/ --device=cuda     # Runs on CUDA (requires GPU with sufficient memory)
"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run tests on: cpu (default) or cuda",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    # Suppress PyTorch TF32 performance hint — enabling TF32 would change
    # float32 precision and risk test failures with tight tolerances
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*TensorFloat32 tensor cores.*:UserWarning",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--device") != "cuda":
        skip_cuda = pytest.mark.skip(reason="CUDA tests require --device=cuda")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


@pytest.fixture
def device(request):
    """Get the device to use for testing, controlled by --device flag."""
    requested = request.config.getoption("--device")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
