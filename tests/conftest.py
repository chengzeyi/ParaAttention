import para_attn
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
    para_attn_version = para_attn.__version__
    report.title = f"ParaAttention {para_attn_version} Tests with {gpu_info}"


@pytest.hookimpl(optionalhook=True)
def pytest_html_results_summary(prefix, summary, postfix):
    prefix.extend(["<p>System Environment Information</p>"])

    # Get system information
    env_info = torch.utils.collect_env.get_pretty_env_info()

    # Add system information to the summary
    prefix.extend([f"<pre>{env_info}</pre>"])
