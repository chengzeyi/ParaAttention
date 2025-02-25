# test_server.py
import base64
import glob
import multiprocessing
import multiprocessing as mp
import os
import time

import pytest
import requests
from para_attn.server import (
    FluxPredictor,
    MockPredictor,
    MockPredictorPredictError,
    MockPredictorSetupError,
    ParaAttentionServer,
)

current_method = mp.get_start_method(allow_none=True)
if current_method != "spawn":
    print(f"{os.getpid()} setting start method to spawn")
    mp.set_start_method("spawn", force=True)


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    """
    Cleanup function to be run after all tests have completed.
    """
    yield
    png_files = glob.glob("*.png")
    for file in png_files:
        os.remove(file)


def does_endpoint_exist(url):
    """
    check if an endpoint exists
    """
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# Define a function to run the server
def run_server(model, host, port, num_devices, log_level):
    server = ParaAttentionServer(model, host=host, port=port, num_devices=num_devices, log_level=log_level)
    server.run()


# Fixture to start the server in a separate process
@pytest.fixture(scope="function")
def server_model_fixture(request):
    model = request.param  # Get the model from the parameterized test

    if hasattr(model, "download_model"):
        model.download_model()

    port = 5000
    host = "localhost"
    num_devices = 4
    log_level = "DEBUG"

    # Start the server in a separate process
    process = multiprocessing.Process(target=run_server, args=(model, host, port, num_devices, log_level))
    process.start()

    yield process, model

    # Terminate the server process after tests
    process.terminate()
    process.join()


@pytest.mark.parametrize("server_model_fixture", [MockPredictorSetupError()], indirect=True)
def test_failed_setup(server_model_fixture):
    """
    test that the server will exit if a worker process throws an exception during setup
    """
    server_process, model = server_model_fixture
    server_process.join()
    assert not server_process.is_alive()


@pytest.mark.parametrize("server_model_fixture", [MockPredictorPredictError()], indirect=True)
def test_failed_predict(server_model_fixture):
    """
    test that if a prediction fails, the server will return status 500
    """
    server_process, model = server_model_fixture
    # get sample predict arguments from the model

    # 1. wait until the server is ready
    server_status_url = "http://localhost:5000/server_status"
    assert server_process.is_alive()
    while not does_endpoint_exist(server_status_url):
        print("Client: server not ready yet")
        time.sleep(1)

    response = requests.get(server_status_url)
    assert response.status_code == 200

    # 2. send a request to the server
    predict_args = model.get_test_predict_args()
    generate_url = "http://localhost:5000/generate"
    response = requests.post(generate_url, json=predict_args)

    # expect status 202 Accepted, along with a request_id
    assert response.status_code == 202
    assert response.json()["request_id"] is not None
    request_id = response.json()["request_id"]

    for i in range(10):
        time.sleep(1)
        get_output_url = f"http://localhost:5000/get_output/{request_id}"
        response = requests.get(get_output_url)
        if response.status_code != 202:
            break
    assert response.status_code == 500


# @pytest.mark.parametrize("server_process", [MockPredictor(), FluxPredictor()], indirect=True)
@pytest.mark.parametrize("server_model_fixture", [FluxPredictor(), MockPredictor()], indirect=True)
def test_startup(server_model_fixture):
    """
    can the server process start up and exit with a variety of predictors
    """
    server_process, model = server_model_fixture
    # poll server until it is ready
    url = "http://localhost:5000/server_status"
    assert server_process.is_alive()
    while not does_endpoint_exist(url):
        print("Client: server not ready yet")
        time.sleep(1)

    response = requests.get(url)
    assert response.status_code == 200


@pytest.mark.parametrize("server_model_fixture", [FluxPredictor(), MockPredictor()], indirect=True)
def test_predict(server_model_fixture):
    """
    can the server process start up, take 2 predicion requests, and return 2 responses
    """
    server_process, model = server_model_fixture
    # get sample predict arguments from the model

    # 1. wait until the server is ready
    server_status_url = "http://localhost:5000/server_status"
    assert server_process.is_alive()
    while not does_endpoint_exist(server_status_url):
        print("Client: server not ready yet")
        time.sleep(1)

    response = requests.get(server_status_url)
    assert response.status_code == 200

    # 2. send a request to the server
    predict_args = model.get_test_predict_args()
    for i in range(2):
        generate_url = "http://localhost:5000/generate"
        response = requests.post(generate_url, json=predict_args)

        # expect status 202 Accepted, along with a request_id
        assert response.status_code == 202
        assert response.json()["request_id"] is not None
        request_id = response.json()["request_id"]

        # 3. poll the server until the request is complete
        while True:
            get_output_url = f"http://localhost:5000/get_output/{request_id}"
            response = requests.get(get_output_url)
            if response.status_code == 202:
                time.sleep(1)
                continue
            elif response.status_code == 200:
                response_json = response.json()
                assert response_json["output"] is not None
                assert response_json["file_type"] is not None
                assert response_json["file_type"] in ["png", "jpg", "jpeg", "mp4"]
                try:
                    data = base64.b64decode(response_json["output"])
                    with open(f"{request_id}.{response_json['file_type']}", "wb") as f:
                        f.write(data)
                except Exception as e:
                    raise Exception(f"Failed to decode output: {e}")
                break

            else:
                raise Exception(f"Unexpected status code {response.status_code}")


@pytest.mark.parametrize("server_model_fixture", [FluxPredictor(), MockPredictor()], indirect=True)
def test_get_output(server_model_fixture):
    """
    the server starts up
    recieves a prediction request
    client queries for the status of another request, server returns 404
    client queries for the status of the first request, server returns 202 or 200
    """
    server_process, model = server_model_fixture
    # get sample predict arguments from the model
    # clean up the file if it already exists
    predict_args = model.get_test_predict_args()

    # 1. wait until the server is ready
    server_status_url = "http://localhost:5000/server_status"
    assert server_process.is_alive()
    while not does_endpoint_exist(server_status_url):
        print("Client: server not ready yet")
        time.sleep(1)

    response = requests.get(server_status_url)
    assert response.status_code == 200

    # 2. send a request to the server
    generate_url = "http://localhost:5000/generate"
    response = requests.post(generate_url, json=predict_args)

    # expect status 202 Accepted, along with a request_id
    assert response.status_code == 202
    assert response.json()["request_id"] is not None
    request_id = response.json()["request_id"]

    incorrect_status_url = f"http://localhost:5000/get_output/{request_id}x"
    response = requests.get(incorrect_status_url)
    assert response.status_code == 404

    # 3. poll the server until the request is complete
    while True:
        status_url = f"http://localhost:5000/get_output/{request_id}"
        response = requests.get(status_url)
        if response.status_code == 202:
            time.sleep(1)
            continue
        elif response.status_code == 200:
            break
        else:
            raise Exception(f"Unexpected status code {response.status_code}")


@pytest.mark.parametrize("server_model_fixture", [FluxPredictor(), MockPredictor()], indirect=True)
def test_cancel(server_model_fixture):
    """
    test cancel

    """
    server_process, model = server_model_fixture
    # get sample predict arguments from the model
    # clean up the file if it already exists

    # 1. wait until the server is ready
    server_status_url = "http://localhost:5000/server_status"
    assert server_process.is_alive()
    while not does_endpoint_exist(server_status_url):
        print("Client: server not ready yet")
        time.sleep(1)

    response = requests.get(server_status_url)
    assert response.status_code == 200

    # 2. send a request to the server
    predict_args = model.get_test_predict_args()
    generate_url = "http://localhost:5000/generate"
    response = requests.post(generate_url, json=predict_args)

    # expect status 202 Accepted, along with a request_id
    assert response.status_code == 202
    assert response.json()["request_id"] is not None
    request_id = response.json()["request_id"]

    time.sleep(1)

    # 3. cancel the request
    cancel_url = "http://localhost:5000/cancel"
    response = requests.post(cancel_url)
    assert response.status_code == 200
    assert response.json()["status"] == f"request {request_id} cancelled"

    # 4. send another request
    predict_args = model.get_test_predict_args()
    response = requests.post(generate_url, json=predict_args)
    assert response.status_code == 202
    assert response.json()["request_id"] is not None
    request_id = response.json()["request_id"]

    # 5. poll the server until the request is complete
    while True:
        status_url = f"http://localhost:5000/get_output/{request_id}"
        response = requests.get(status_url)
        if response.status_code == 202:
            time.sleep(1)
            continue
        elif response.status_code == 200:
            response_json = response.json()
            assert response_json["output"] is not None
            assert response_json["file_type"] is not None
            assert response_json["file_type"] in ["png", "jpg", "jpeg", "mp4"]
            break
        else:
            raise Exception(f"Unexpected status code {response.status_code}")


# @pytest.mark.parametrize("server_model_fixture", [LargeOutputPredictor()], indirect=True)
# def test_large_output(server_model_fixture):
#     """
#     test that the server can handle large outputs
#     """
#     server_process, model = server_model_fixture
#     TODO: implement this test
