import pytest

from typing import List
from ray.tests.test_autoscaler import MockProvider, MockProcessRunner
from ray.autoscaler._private.gcp.tpu_command_runner import TPUCommandRunner
from getpass import getuser
import hashlib

auth_config = {
    "ssh_user": "ray",
    "ssh_private_key": "8265.pem",
}


class MockTpuInstance:
    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers

    def get_internal_ip(self, worker_index: int) -> str:
        return "0.0.0.{}".format(worker_index)

    def get_external_ip(self, worker_index: int) -> str:
        return "0.0.0.{}".format(worker_index)


def test_tpu_ssh_command_runner():
    num_workers = 2
    process_runner = MockProcessRunner()
    provider = MockProvider()
    instance = MockTpuInstance(num_workers=num_workers)
    provider.create_node({}, {}, 1)
    cluster_name = "cluster"
    ssh_control_hash = hashlib.md5(cluster_name.encode()).hexdigest()
    ssh_user_hash = hashlib.md5(getuser().encode()).hexdigest()
    ssh_control_path = "/tmp/ray_ssh_{}/{}".format(
        ssh_user_hash[:10], ssh_control_hash[:10]
    )
    args = {
        "instance": instance,
        "log_prefix": "prefix",
        "node_id": "abc",
        "provider": provider,
        "auth_config": auth_config,
        "cluster_name": cluster_name,
        "process_runner": process_runner,
        "use_internal_ip": False,
    }
    env_vars = {"var1": 'quote between this " and this', "var2": "123"}
    cmd_runner = TPUCommandRunner(**args)
    cmd_runner.run(
        "echo helloo", port_forward=[(8265, 8265)], environment_variables=env_vars
    )

    def get_expected(worker_id: int) -> List[str]:
        return [
            "ssh",
            "-tt",
            "-L",
            "8265:localhost:8265",
            "-i",
            "8265.pem",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPath={}/%C".format(ssh_control_path),
            "-o",
            "ControlPersist=10s",
            "-o",
            "ConnectTimeout=120s",
            "ray@0.0.0.{}".format(worker_id),
            "bash",
            "--login",
            "-c",
            "-i",
            """'source ~/.bashrc; export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && (export var1='"'"'"quote between this \\" and this"'"'"';export var2='"'"'"123"'"'"';echo helloo)'""",  # noqa: E501
        ]

    calls = process_runner.calls

    assert len(process_runner.calls) == num_workers

    # Much easier to debug this loop than the function call.
    for i in range(num_workers):
        expected = get_expected(i)
        for x, y in zip(calls[i], expected):
            assert x == y
        process_runner.assert_has_call("1.2.3.4", exact=expected)


def test_tpu_docker_command_runner():
    num_workers = 1
    process_runner = MockProcessRunner()
    provider = MockProvider()
    instance = MockTpuInstance(num_workers=num_workers)
    provider.create_node({}, {}, 1)
    cluster_name = "cluster"
    ssh_control_hash = hashlib.md5(cluster_name.encode()).hexdigest()
    ssh_user_hash = hashlib.md5(getuser().encode()).hexdigest()
    ssh_control_path = "/tmp/ray_ssh_{}/{}".format(
        ssh_user_hash[:10], ssh_control_hash[:10]
    )
    docker_config = {"container_name": "container"}
    args = {
        "instance": instance,
        "log_prefix": "prefix",
        "node_id": "0",
        "provider": provider,
        "auth_config": auth_config,
        "cluster_name": cluster_name,
        "process_runner": process_runner,
        "use_internal_ip": False,
        "docker_config": docker_config,
    }
    cmd_runner = TPUCommandRunner(**args)

    env_vars = {"var1": 'quote between this " and this', "var2": "123"}
    cmd_runner.run("echo hello", environment_variables=env_vars)

    # This string is insane because there are an absurd number of embedded
    # quotes. While this is a ridiculous string, the escape behavior is
    # important and somewhat difficult to get right for environment variables.
    cmd = """'source ~/.bashrc; export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && (docker exec -it  container /bin/bash -c '"'"'bash --login -c -i '"'"'"'"'"'"'"'"'source ~/.bashrc; export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore && (export var1='"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"quote between this \\" and this"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"';export var2='"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"123"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"'"';echo hello)'"'"'"'"'"'"'"'"''"'"' )'"""  # noqa: E501

    def get_expected(worker_id: int) -> List[str]:
        return [
            "ssh",
            "-tt",
            "-i",
            "8265.pem",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPath={}/%C".format(ssh_control_path),
            "-o",
            "ControlPersist=10s",
            "-o",
            "ConnectTimeout=120s",
            "ray@0.0.0.{}".format(worker_id),
            "bash",
            "--login",
            "-c",
            "-i",
            cmd,
        ]

    calls = process_runner.calls
    assert len(process_runner.calls) == num_workers

    # Much easier to debug this loop than the function call.
    for i in range(num_workers):
        expected = get_expected(i)
        for x, y in zip(calls[i], expected):
            assert x == y
        process_runner.assert_has_call("1.2.3.4", exact=expected)


if __name__ == "__main__":
    import os
    import sys

    if os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))