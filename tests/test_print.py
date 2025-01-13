import pytest
import os

@pytest.fixture
def model(allow_cuda):
    """Get the model instance."""
    val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    assert val is None, f"{val}"
    return "MODEL"


# @pytest.mark.xdist_group(name="gpu")
# def test_print1():
#     val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
#     assert val == "0", f"{val}"
#
# def test_print2():
#     val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
#     assert val == "0", f"{val}"

# def test_print3(model):
#     print("Awesome3")
#     assert 2 == 3

@pytest.mark.xdist_group(name="gpu")
def test_open_file(model):
    assert 2==2