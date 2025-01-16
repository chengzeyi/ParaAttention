from typing import Optional

import importlib
import logging
import torch
import typer
import torch.multiprocessing as mp

from para_attn.server import ParaAttentionServer

current_method = mp.get_start_method(allow_none=True)
if current_method != "spawn":
    mp.set_start_method('spawn', force=True)

app = typer.Typer()

logging_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

def load_model(model_path: str):
    module_name, class_name = model_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

@app.command()
def serve(
    model: str = typer.Option(..., "--model", help="The model to serve"),
    host: Optional[str] = typer.Option("127.0.0.1", "--host", help="The hostname to serve the model on"),
    port: Optional[int] = typer.Option(5000, "--port", help="The port number to serve the model on"),
    num_devices: Optional[int] = typer.Option(torch.cuda.device_count(), "--num-devices", help="The number of devices to serve the model on"),
    log_level: Optional[str] = typer.Option("INFO", "--log-level", help="The log level to serve the model on"),
):
    """
    Serve the model on a specified host and port.

    Parameters:
    - model (str): module path of the model, in the format of module.submodule:ModelClassName
    - host (Optional[str]): The hostname to serve the model on. Defaults to "127.0.0.1".
    - port (Optional[int]): The port number to serve the model on. Defaults to 5000.
    - num_devices (Optional[int]): The number of devices to serve the model on. Defaults to all devices.
    """
    if log_level.lower() not in logging_levels:
        raise ValueError(f"Invalid log level: {log_level}. Valid levels are: {logging_levels.keys()}")
    log_level = logging_levels[log_level.lower()]
    
    model_class = load_model(model)
    server = ParaAttentionServer(model_class(), host, port, num_devices, log_level)
    server.run()

if __name__ == "__main__":
    app()