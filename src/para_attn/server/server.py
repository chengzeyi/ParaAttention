import inspect
import os
import signal
import socket
import sys
import time
import uuid

import torch.distributed as dist
import torch.multiprocessing as mp
from flask import Flask, jsonify, request

from .inference_worker import inference_worker

from .logger_factory import LoggerFactory

HTTP_STATUS = {
    "OK": 200,
    "ACCEPTED": 202,
    "NOT_FOUND": 404,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "INTERNAL_SERVER_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503,
}

logger = None


class ParaAttentionServer:
    def __init__(self, model, host: str, port: int, num_devices: int, log_level: int):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.torch_distributed_port = 29500
        self.world_size = dist.get_world_size() if num_devices is None else num_devices
        self.log_level = log_level

        # register signal handler for SIGTERM/SIGINT
        signal.signal(signal.SIGTERM, self._create_signal_handler())
        signal.signal(signal.SIGINT, self._create_signal_handler())

        global logger
        logger = LoggerFactory.create_logger(name="Server Process", level=log_level)
        logger.info(f"running on {self.host}:{self.port} with {self.world_size} devices")

        if not self._is_port_available(self.port):
            raise ValueError(f"port {self.port} is already in use")

        # look for a free port for torch distributed. This is a less than ideal hack
        # its mostly required for testing, sometimes running inference on a model will
        # result in worker processes being uninterruptable, which prevents the signal handler
        # from firing when the server terminates the worker process at the end of the class
        # lifetime. This prevents the subprocess from being able to call dist.destroy_process_group()
        # which is required if we want to resuse the port. This shouldn't be an issue in production, because
        # the server wont be terminated/restarted, but it is a problem for testing.
        while not self._is_port_available(self.torch_distributed_port):
            self.torch_distributed_port += 1

        # for sending requests to workers
        self.in_queue = [mp.Queue() for _ in range(self.world_size)]

        # for rank 0 to receive images from workers
        self.out_queue = mp.Queue()

        # for workers communicating status and errors back to host
        self.worker_error_msg_queue = [mp.Queue() for _ in range(self.world_size)]

        # worker_status[i] == 0 means worker with rank i is good
        # any worker_status[i] != 0 means error occured
        self.worker_status = [mp.Value("i", 0) for _ in range(self.world_size)]

        # an event that is used to communicate to workers that a request has been cancelled
        # it is set by the server, and waited on by a dedicated thread within the worker processes
        self.cancel_event = mp.Event()

        # the model to be used for inference with setup() and predict() methods implemented by user
        # we will throw an exception here is the model is not valid
        self.model = model
        self._validate_model(self.model)

        # get arguments names of all non-optional arguments of self.model.predict
        self.predict_args = inspect.signature(self.model.predict).parameters.keys()

        # an unsigned integer counter to track how many workers are available
        # to handle an incoming request
        self.ready_workers = mp.Value("i", 0)

        # bookkeeping for current request status
        self.current_request_id = None

        # set mp start method to spawn, required if subprocesses use CUDA
        self._init_mp()

        # spawn worker processes
        self.worker_processes = []
        for i in range(self.world_size):

            worker_args = (
                self.in_queue[i],
                self.out_queue if i == 0 else None,
                self.worker_error_msg_queue[i],
                self.worker_status[i],
                self.cancel_event,
                i,
                self.world_size,
                self.model,
                self.ready_workers,
                self.torch_distributed_port,
                self.log_level,
            )

            worker_process = mp.Process(target=inference_worker, args=worker_args)
            self.worker_processes.append(worker_process)
            worker_process.start()

        # wait for workers to initialize
        while True:
            time.sleep(1)
            logger.debug("waiting for workers to initialize ...")

            # check if any workers have encountered an error
            # if an error occurs in a worker process during setup, we will rethrow it here
            # and the server will exit
            self._are_workers_good()

            # check if all workers are ready
            if self.ready_workers.value == self.world_size:
                logger.debug("workers initialized!")
                break

        # register routes
        self.app.route("/generate", methods=["POST"])(self.generate)
        self.app.route("/cancel", methods=["POST"])(self.cancel)
        self.app.route("/server_status", methods=["GET"])(self.server_status)
        self.app.route("/get_output/<request_id>", methods=["GET"])(self.get_output)

    # destructor
    def __del__(self):
        logger.debug(f"server {os.getpid()} destructor called")
        self._cleanup()

    def _create_signal_handler(self):
        """
        create a signal handler for the server
        which is just _cleanup() + exit(0)
        """

        def handler(signum, frame):
            logger.debug("signal handler called")
            self._cleanup()
            sys.exit(0)

        return handler

    def _cleanup(self):
        """
        cleanup function for the server, send SIGTERM, and join all worker processes
        should be idempotent, doesn't matter if it is called multiple times
        """

        if hasattr(self, "worker_processes") and self.worker_processes:
            logger.debug("cleaning up workers")
            for worker in self.worker_processes:
                worker.terminate()

            for ix, worker in enumerate(self.worker_processes):
                worker.join(timeout=1)
                if worker.is_alive():
                    logger.warning(f"worker {ix} unresponsive to SIGTERM, sending SIGKILL")
                    os.kill(worker.pid, signal.SIGKILL)
            logger.debug("workers cleaned up")
        self.worker_processes = None

    def _are_workers_good(self):
        """
        check whether all workers are alive and well, if not will raise an exception
        if so will return True
        """
        logger.debug("checking worker statuses")
        for i in range(self.world_size):
            if not self.worker_processes[i].is_alive():
                raise ValueError(f"worker {i} is not alive, restart the server")
            if self.worker_status[i].value != 0:
                error_msg = self.worker_error_msg_queue[i].get_nowait()
                raise ValueError(error_msg)
        logger.debug("worker statuses are good")
        return True

    def _is_port_available(self, port, host="localhost"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            return True
        except socket.error:
            return False
        finally:
            sock.close()

    def _validate_request_args(self, request_kwargs):
        """
        all keys in request_kwargs must be arguments to predict
        """
        return all(k in self.predict_args for k in request_kwargs.keys())

    def _validate_model(self, model):
        """
        model must have methods setup and predict
        """
        if not hasattr(model, "setup"):
            raise ValueError("model must have a setup method")
        if not hasattr(model, "predict"):
            raise ValueError("model must have a predict method")

    def _init_mp(self):
        """
        set mp start method to spawn, required if subprocesses use CUDA
        """
        current_method = mp.get_start_method(allow_none=True)
        if current_method != "spawn":
            logger.debug(f"{os.getpid()} setting start method to spawn")
            mp.set_start_method("spawn", force=True)

    def generate(self):
        """
        Handle generation requests.
        Returns a request ID that can be used to check status.
        """
        try:
            self._are_workers_good()

            # validate kwargs
            kwargs = request.get_json()
            if not self._validate_request_args(kwargs):
                # bad request, request contains invalid arguments
                return jsonify({"error": "request contains invalid arguments"}), HTTP_STATUS["BAD_REQUEST"]

            # if workers are not all ready, sleep for 100ms and check again
            # if we just finished serving the previous request, it may take an additional instant
            # or two for the workers to all be ready again
            seconds_to_wait = 0
            while self.ready_workers.value != self.world_size and seconds_to_wait < 5:
                time.sleep(0.1)
                seconds_to_wait += 0.1

            # if workers are not all ready after waiting for 5 seconds, one of them probably hung
            if self.ready_workers.value != self.world_size:
                raise ValueError(
                    f"server {os.getpid()}: only {self.ready_workers.value} workers ready, this is probably a bug"
                )

            # request ids are used for tracking of in flight requests
            # bookkeep the request status
            self.current_request_id = str(uuid.uuid4())

            # ensure all in_queues are empty
            # since this server only handles one request at a time
            # TODO if we are going to allow the server to queue requests get rid of this check
            # right now server is handling only one request at a time
            for ix in range(self.world_size):
                if not self.in_queue[ix].empty():
                    raise ValueError(f"in_queue {ix} is not empty")

            # send the request to all workers and check that all workers are good
            for i in range(self.world_size):
                self.in_queue[i].put((self.current_request_id, kwargs))

            logger.debug(f"request {self.current_request_id} accepted")
            return jsonify({"request_id": self.current_request_id}), HTTP_STATUS["ACCEPTED"]
        except Exception as e:
            logger.error(f"internal server error: {e}")
            return jsonify({"internal server error": str(e)}), HTTP_STATUS["INTERNAL_SERVER_ERROR"]

    def get_output(self, request_id):
        """
        Get the output of a generation request.
        """
        try:

            logger.debug(
                f"getting output for request {request_id}, current request id is {self.current_request_id}, qsize is {self.out_queue.qsize()}"
            )

            self._are_workers_good()

            if self.current_request_id is None:
                return jsonify({"error": "no request in flight"}), HTTP_STATUS["NOT_FOUND"]

            elif self.current_request_id != request_id:
                return (
                    jsonify(
                        {"error": f"request {request_id} not found, current request id is {self.current_request_id}"}
                    ),
                    HTTP_STATUS["NOT_FOUND"],
                )

            elif self.out_queue.qsize() == 0:
                assert (
                    self.ready_workers.value != self.world_size
                ), "workers all ready but in flight request is not complete, this should never happen"
                return jsonify({"status": f"request {request_id} is in flight"}), HTTP_STATUS["ACCEPTED"]

            elif self.out_queue.qsize() == 1:
                request_id, data, format = self.out_queue.get()
                if request_id != self.current_request_id:
                    return (
                        jsonify(
                            {
                                "error": "model request id does not equal server request id, this is a bug in the server implementation"
                            }
                        ),
                        HTTP_STATUS["INTERNAL_SERVER_ERROR"],
                    )
                self.current_request_id = None
                return jsonify({"output": data, "file_type": format}), HTTP_STATUS["OK"]
            else:
                raise ValueError(
                    f"out_queue.qsize() == {self.out_queue.qsize()}, current request id is {self.current_request_id}, this should never happen"
                )

        except Exception as e:
            logger.error(f"internal server error: {e}")
            return jsonify({"error": str(e)}), HTTP_STATUS["INTERNAL_SERVER_ERROR"]

    def server_status(self):
        """
        check whether the server is here or not
        """
        try:
            self._are_workers_good()
            return jsonify({"status": "server is running"}), HTTP_STATUS["OK"]
        except Exception as e:
            logger.error(f"internal server error: {e}")
            return jsonify({"error": str(e)}), HTTP_STATUS["INTERNAL_SERVER_ERROR"]

    def cancel(self):
        """
        Cancel an ongoing generation request.
        """
        try:
            if self.current_request_id is None:
                return (
                    jsonify({"error": "there is currently no request in flight to cancel"}),
                    HTTP_STATUS["NOT_FOUND"],
                )
            else:
                request_id = self.current_request_id

            self.cancel_event.set()
            while self.ready_workers.value < self.world_size:
                time.sleep(1)
                self._are_workers_good()
            self.cancel_event.clear()

            # set current request id to None and clear the out_queue
            self.current_request_id = None
            if self.out_queue.qsize() > 0:
                self.out_queue.get()

            return jsonify({"status": f"request {request_id} cancelled"}), HTTP_STATUS["OK"]

        except Exception as e:
            logger.error(f"internal server error: {e}")
            return jsonify({"error": str(e)}), HTTP_STATUS["INTERNAL_SERVER_ERROR"]

    def run(self):
        """
        Start the Flask server.
        """
        self.app.run(host=self.host, port=self.port)
