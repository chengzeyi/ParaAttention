import os
import signal
import sys
import threading
import time

import torch.distributed as dist

from .logger_factory import LoggerFactory

global logger


# cleanup function for the worker, destroy process group
def _worker_cleanup():
    """
    cleanup function for the worker, destroy process group
    """
    if dist.is_initialized():
        dist.destroy_process_group()


# create a signal handler for the worker, which is just _worker_cleanup() + exit(0)
def _create_worker_signal_handler(rank):
    def handler(signum, frame):
        logger.debug("signal handler called")
        _worker_cleanup()
        logger.debug("signal handler done")
        sys.exit(0)

    return handler


def inference_worker(
    in_queue,
    out_queue,
    error_queue,
    worker_status,
    cancel_event,
    rank,
    world_size,
    model,
    ready_workers,
    torch_distributed_port,
    log_level,
):
    # register signal handler for SIGTERM/SIGINT
    signal.signal(signal.SIGTERM, _create_worker_signal_handler(rank))
    signal.signal(signal.SIGINT, _create_worker_signal_handler(rank))

    def _request_cancellation_monitor():
        while True:
            cancel_event.wait()
            logger.debug("received cancel event")
            model.pipe._interrupt = True
            while cancel_event.is_set():
                time.sleep(1)  # wait for main process to clear the cancel event

    # start a thread to monitor for request cancellations
    cancellation_thread = threading.Thread(target=_request_cancellation_monitor)
    cancellation_thread.daemon = (
        True  # set the thread to daemon, so it will be killed immediately when the main thread exits
    )
    cancellation_thread.start()

    try:
        # these environment variables need to be set in advance of calling dist.init_process_group
        # which we require to be called by the model setup() method
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(torch_distributed_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)

        global logger
        logger = LoggerFactory.create_logger(name=f"Worker-{rank}", level=log_level)
        model.setup()

        # we expect 2 things from setup()
        # 1. dist.is_initialized() is true
        # 2. self.pipe is a callable, and it should be a DiffusersPipeline
        assert dist.is_initialized()
        assert model.pipe is not None
        assert callable(model.pipe)

        while True:
            # increment the ready workers counter to indicate that this worker is ready to handle an incoming request
            with ready_workers.get_lock():
                ready_workers.value += 1

            logger.debug(f"ready workers: {ready_workers.value}")
            id, request = in_queue.get()

            # decrement the ready workers counter to indicate that this worker is no longer ready to handle an incoming request
            with ready_workers.get_lock():
                ready_workers.value -= 1

            output = model.predict(**request)

            # rank 0 is the only process that receives model output and passes it back to the main process
            if out_queue is not None:
                assert output is not None
                assert len(output) == 2
                image, format = output
                out_queue.put((id, image, format))

            logger.debug("prediction complete")

            # make sure cancellation thread is still active
            if not cancellation_thread.is_alive():
                raise Exception(f"worker {rank} cancellation thread is not alive, exiting")

    except Exception as e:
        logger.error(f"exception: {e}")
        with worker_status.get_lock():
            worker_status.value = 1
            error_queue.put(f"worker {rank} exception: error {e}")
            _worker_cleanup()
            exit(1)
