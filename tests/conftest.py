import os
import pytest

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Add CPU group by default
    """

    # put all CPU tests in a single xdist_group
    cpu_group = pytest.mark.xdist_group(name="cpu")
    for item in items:
        if not any(is_gpu_mark(mark) for mark in item.iter_markers()):
            item.add_marker(cpu_group)


def is_gpu_mark(mark: pytest.Mark) -> bool:
    """Only return True for `@pytest.mark.xdist_group(name="gpu")`"""
    if mark.name != "xdist_group":
        return False
    return mark.kwargs.get("name", None) == "gpu"


# by default we disable CUDA, and only enable it for tests that declare cuda usage
# by using decorator @pytest.mark.xdist_group(name="gpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture(autouse=True)
def allow_cuda(request):
    """Only allow CUDA usage to tests decorated with `@pytest.mark.xdist_group(name="gpu")`.

    This is needed to allow multi-processed execution of tests.
    Without this, it would be hard to trace which test led to concurrent usage of GPU.
    Allowing concurrent usage of GPU is error-prone due to running out of memory.

    When user forgets to use the gpu decorator but uses GPU,
    some of the following errors are expected:
        * "Attempting to deserialize object on CUDA device 0 but torch.cuda.device_count() is 0",
        * "No CUDA GPUs are available"
        * "invalid literal for int() with base 10: ''"
    """

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "", "expected CUDA to be disabled by default."

    # Check any of the marks is the target mark
    if not any(is_gpu_mark(mark) for mark in request.node.iter_markers()):
        # Not a gpu marked test. Leave cuda disabled.
        yield
    else:
        # enable CUDA for the test, disable it later
        try:
            os.environ.pop("CUDA_VISIBLE_DEVICES")
            yield
        finally:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
