# ParaAttention Server

This folder contains a server implemented with python/flask that can be used to generate content with parallelized diffusers pipelines. I implemented this server because I couldn't find any good existing solution for serving parallelized diffusers pipelines in a production environment. This server is in its early stages, and improvements are welcomed and appreciated!

## HTTP Endpoints

- **`/generate` (POST)**: Accepts a generation request, validates parameters, and ensures all workers are ready. Returns a unique request ID for tracking.
  - **202 ACCEPTED**: Request accepted and is being processed. Request ID is also returned
  - **400 BAD_REQUEST**: Request contains invalid arguments or output directory does not exist.
  - **500 INTERNAL_SERVER_ERROR**: An internal error occurred during request processing.

- **`/get_output/<request_id>` (POST)**: Checks the status of a specific generation request using the provided `request_id`. Determines if the request is in flight, completed, or not found. If request is completed, the output is returned along with the file type
  - **200 OK**: Request is completed and output is available.
  - **202 ACCEPTED**: Request is still in flight.
  - **404 NOT_FOUND**: No request currently in flight with the provided request ID.
  - **500 INTERNAL_SERVER_ERROR**: An internal error occurred while checking request status.

- **`/cancel` (POST)**: Cancels an ongoing generation request by setting a cancellation event for worker processes.
  - **200 OK**: Request successfully cancelled.
  - **404 NOT_FOUND**: No request in flight to cancel.
  - **500 INTERNAL_SERVER_ERROR**: An internal error occurred during cancellation.

- **`/server_status` (GET)**: Checks the health of the server and verifies that all worker processes are functioning correctly.
  - **200 OK**: Server is running and all worker processes are functioning correctly.
  - **500 INTERNAL_SERVER_ERROR**: An internal error occurred while checking server status.


## Usage
### Setup
The server can be used either by importing the `ParaAttentionServer` class, instantiating it with the necessary arguments, and calling its `run` method:
```python
from para_attn.server import ParaAttentionServer
from para_attn.server.predictors import FluxPredictor

host = "127.0.0.1"
port = 5000
num_devices = 2

server = ParaAttentionServer(FluxPredictor(), host, port, num_devices)
server.run()
```

Or by running the `para-attn-server` command:
```bash
para-attn-server --model para_attn.server.predictors:FluxPredictor --host 127.0.0.1 --port 5000 --num-devices 2
```
In this later usage, the model path must be a module path that can be imported by `importlib.import_module` (see [below](#implementing-a-model-predictor) for more details about predictors).

### Server Status
To check whether the server has started sucessfully, and check whether all processes are ready and in a good state, you can post a request to the `/server_status` endpoint:
```bash
curl -X POST http://localhost:5000/server_status
```

### Generating Content
After the server is running, you can post a request to the `/generate` endpoint with model arguments to generate an image:

```bash
curl -X POST http://localhost:5000/generate  -H "Content-Type: application/json"  -d '{
           "prompt": "a cat in a hat",
           "negative_prompt": "a cat in a hat",
           "guidance_scale": 7,
           "num_inference_steps": 50,
           "output_path": "cat_in_hat.png",
           "height": 1024,
           "width": 1024
         }'
# response:{"request_id":"c28e0fa3-4360-4d12-8832-847a2e8e6ad7"}
```
If the request is successful, this endpoint will return immediately with a `request_id` that can be used to check the status of the request or cancel it. Once the request is complete, the server will write the output image to the specified path.

### Cancelling Requests
If the request is taking a long time and you want to cancel it, you can post a request to the `/cancel` endpoint, which will cause the worker processes to abort once they are done with the current denoising step
```bash
curl -X POST http://localhost:5000/cancel
```

### Checking Request Status
To check whether the request is complete, you can post a request to the `/request_status/<request_id>` endpoint:
```bash
curl -X POST http://localhost:5000/request_status/c28e0fa3-4360-4d12-8832-847a2e8e6ad7
```

**IMPORTANT** if any of these endpoints return status 500 (Internal Server Error) at any point, something has gone wrong and you should restart the server. Please open an issue if this happens.


## Implementing a model predictor
The implementation of this server is meant to seperate model specific logic away from the server implementation. In order to serve a new model, you must implement a thin wrapper around model intialization and prediction. See [here](predictors.py#L10) for examples of how to do this. The server imposes a [few requirements](inference_worker.py#L76) on what must happen in the `setup` method
- `dist.init_process_group()` must be called
- the model must have an attribute `pipe` which should be the diffusers pipeline containing the inference logic. This is needed for request cancellation to work.

The `predict` method
- [must](server.py:L208) have an an argument `output_path`
- once inference is complete, one process (canonically the one with rank zero like [here](predictors.py:L50)) must write the model output to disk after inference is finished.

When the server recieves keyword arguments at the `/generate` endpoint, the server will [validate](server.py#L239) that all parameters of the `predict` method have a matching kwarg. In order words, the keys in the dictionary you provide to `/generate` must exactly match the parameter names of predict.

Behind the scenes, the server will spawn `num_devices` processes, assign each one a rank, have each process call `setup`, and then on each prediction request, have each process call `predict` with matching arguments. For a correctly implemented `setup`/`predict`, if you were to comment out the couple of calls to `torch.distributed` and `para_attn`, the remaining code should look no different than it would for single GPU inference.

## Future Work
- currently if an error is thrown in the model `predict` method, the server goes into an error state
  and must be restarted by user. This simplifies exception handling, but may not be the best long term
  approach for production.
- choose a more production oriented backend than flask
