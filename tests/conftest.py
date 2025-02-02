import pytest

import torch


@pytest.hookimpl(optionalhook=True)
def pytest_html_report_title(report):
    """
    A pytest-html hook to add custom fields to the report.
    """
    # Customize the title of the html report (if used) to include version.
    device_count = torch.cuda.device_count()
    if device_count == 0:
        gpu_info = "no GPU"
    else:
        gpu_info = f"{device_count} {torch.cuda.get_device_name(0)} GPU(s)"
    report.title = f"ParaAttention Tests with {gpu_info}"
