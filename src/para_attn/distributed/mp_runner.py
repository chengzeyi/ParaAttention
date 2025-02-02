import contextlib
import os
import sys
import pickle
import socket
import signal
import threading
import time
import traceback
from multiprocessing.reduction import ForkingPickler

import torch.distributed as dist
import torch.multiprocessing as mp

class ExceptionOutput(Exception):
    pass

class MPDistRunner:
    def __init__(self, *, rank=None, persist_attrs=None, temp_attrs=None):
        self.rank = rank
        self.persist_attrs = persist_attrs or {}
        self.temp_attrs = temp_attrs or {}

        self.array_size = None
        self.shared_array = None
        self.input_queue = None
        self.output_queue = None
        self.status = None
        self.lock = None

        self.processes = []

    def __del__(self):
        self.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    @property
    def world_size(self):
        raise NotImplementedError

    @property
    def start_method(self):
        return "spawn"

    @property
    def max_input_size(self):
        return 1 << 30

    def clone(self, *, rank=None):
        return self.__class__(rank=rank, persist_attrs={**self.persist_attrs})

    def init_env(self, *, master_addr=None, master_port=None):
        if master_addr is None:
            if "MASTER_ADDR" not in os.environ:
                master_addr = "127.0.0.1"
                os.environ["MASTER_ADDR"] = master_addr
        else:
            os.environ["MASTER_ADDR"] = master_addr
        if master_port is None:
            if "MASTER_PORT" not in os.environ:
                master_port = find_free_port()
                os.environ["MASTER_PORT"] = str(master_port)
        else:
            os.environ["MASTER_PORT"] = str(master_port)

    def start(self, args=(), kwargs=None, *, timeout=None):
        if self.processes:
            raise RuntimeError("Processes are already started")

        if kwargs is None:
            kwargs = {}

        self.init_env()

        world_size = self.world_size
        if world_size <= 0:
            raise ValueError(f"World size {world_size} is invalid")

        start_method = self.start_method
        mp_ = mp.get_context(start_method)

        barrier = mp_.Barrier(world_size)
        array_size = mp_.Value("q", 0, lock=False)
        shared_array = mp_.Array("c", self.max_input_size, lock=False)
        input_queue = mp_.JoinableQueue(maxsize=1)
        output_queue = mp_.Queue(maxsize=1)
        exception_queues = [mp_.Queue(maxsize=1) for _ in range(world_size)]
        status = mp_.Value("i", 0, lock=True)
        processes = []

        try:
            for rank in range(world_size):
                process = mp_.Process(
                    target=self.worker,
                    args=(
                        self.clone(rank=rank),
                        args,
                        kwargs,
                        barrier,
                        array_size,
                        shared_array,
                        input_queue if rank == 0 else None,
                        output_queue if rank == 0 else None,
                        exception_queues[rank] if rank == 0 else None,
                        status if rank == 0 else None,
                    ),
                    daemon=True,
                )
                process.start()
                processes.append(process)

            output_queue.get(timeout=timeout)
            exceptions = []
            for rank, exception_queue in enumerate(exception_queues):
                if not exception_queue.empty():
                    exceptions.append((rank, exception_queue.get()))
            if exceptions:
                msg = "\n".join(f"Rank {rank}: {exception}" for rank, exception in exceptions)
                raise RuntimeError(f"Exceptions occurred:\n{msg}")
        except Exception:
            self.terminate()
            raise

        self.array_size = array_size
        self.shared_array = shared_array
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.exception_queues = exception_queues
        self.status = status
        self.lock = threading.RLock()
        self.processes = processes

        return self

    def terminate(self):
        self.array_size = None
        self.shared_array = None
        self.input_queue = None
        self.output_queue = None
        self.exception_queues = None
        self.status = None

        for process in self.processes:
            if process.is_alive():
                process.terminate()
        self.processes = []

    def restart(self, *args, **kwargs):
        self.terminate()
        self.start(*args, **kwargs)

    def is_alive(self):
        return self.processes and all(process.is_alive() for process in self.processes)

    def is_idle(self):
        if not self.is_alive():
            return False

        with self.status.get_lock():
            return self.status.value == 0

    def is_almost_idle(self):
        if not self.is_alive():
            return False

        with self.status.get_lock():
            return self.status.value in (0, 2)

    def __call__(self, args=(), kwargs=None, *, timeout=None):
        if kwargs is None:
            kwargs = {}

        with self.lock:
            if not self.processes:
                raise RuntimeError("Processes are not started")
            for rank, process in enumerate(self.processes):
                if not process.is_alive():
                    raise RuntimeError(f"Process {rank} is not alive")

            data = ForkingPickler.dumps((args, kwargs), pickle.HIGHEST_PROTOCOL)
            data_size = len(data)
            if data_size > self.max_input_size:
                raise RuntimeError(f"Data size {data_size} exceeds maximum size {self.max_input_size}")
            self.array_size.value = data_size
            self.shared_array[:data_size] = data
            if timeout is not None:
                begin_time = time.time()
            self.input_queue.put(True, timeout=timeout)
            self.input_queue.join()
            if timeout is not None:
                end_time = time.time()
                duration = end_time - begin_time
                timeout = max(0, timeout - duration)
            output = self.output_queue.get(timeout=timeout)
            exceptions = []
            for rank, exception_queue in enumerate(self.exception_queues):
                if (rank == 0 and isinstance(output, ExceptionOutput)) or not exception_queue.empty():
                    exceptions.append((rank, exception_queue.get()))
            if exceptions:
                msg = "\n".join(f"Rank {rank}: {exception}" for rank, exception in exceptions)
                raise RuntimeError(f"Exceptions occurred:\n{msg}")
            return output

    def init_process_group(self):
        if dist.is_initialized():
            return
        dist.init_process_group(world_size=self.world_size, rank=self.rank)

    def destroy_process_group(self):
        if not dist.is_initialized():
            return
        dist.destroy_process_group()

    def init_processor(self):
        pass

    def process_task(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def worker(
        cls, runner, args, kwargs, barrier, array_size, shared_array, input_queue, output_queue, exception_queue, status,
    ):
        runner.init_process_group()

        try:
            try:
                runner.init_processor(*args, **kwargs)
            except Exception as e:
                exception = RuntimeError(f"Failed to initialize processor: {e}\n{traceback.format_exc()}")
                if exception_queue is not None:
                    exception_queue.put(exception)
            barrier.wait()
            if output_queue is not None:
                output_queue.put(True)

            while True:
                if input_queue is None:
                    barrier.wait()
                else:
                    input_queue.get()
                    barrier.wait()

                if status is not None:
                    with status.get_lock():
                        status.value = 1

                data_size = array_size.value
                value_bytes = shared_array[:data_size]

                output = None
                exception = None

                try:
                    input_args, input_kwargs = ForkingPickler.loads(value_bytes)
                except Exception as e:
                    exception = RuntimeError(f"Failed to unpickle data: {e}\n{traceback.format_exc()}")

                if output_queue is not None:
                    while not output_queue.empty():
                        output_queue.get()
                if exception_queue is not None:
                    while not exception_queue.empty():
                        exception_queue.get()
                if input_queue is not None:
                    input_queue.task_done()

                if exception is None:
                    try:
                        output = runner.process_task(*input_args, **input_kwargs)
                    except Exception as e:
                        exception = RuntimeError(f"Failed to process task: {e}\n{traceback.format_exc()}")

                if status is not None:
                    with status.get_lock():
                        status.value = 2

                if exception_queue is not None:
                    if exception is not None:
                        exception_queue.put(exception)

                if output_queue is not None:
                    if exception is None:
                        output_queue.put(output)
                    else:
                        output_queue.put(ExceptionOutput())

                barrier.wait()

                if status is not None:
                    with status.get_lock():
                        status.value = 0
        finally:
            runner.destroy_process_group()


def find_free_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
