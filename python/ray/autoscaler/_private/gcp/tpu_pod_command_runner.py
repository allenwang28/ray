import ray

from concurrent.futures import ThreadPoolExecutor

from functools import partial
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler._private.gcp.node import GCPTPUNode


_MAX_NUM_CONCURRENT_ACTIVE_CONNECTIONS = 32


class TPUVMSSHCommandRunner(SSHCommandRunner):
    """An SSH command runner for TPU VMs that overwrites IP addresses."""

    def __init__(
        self, internal_ip, external_ip, *args, **kwargs):
        self._internal_ip = internal_ip
        self._external_ip = external_ip
        super().__init__(*args, **kwargs)

    def _get_node_ip(self):
        if self.use_internal_ip:
            return self._internal_ip
        else:
            return self._external_ip


class TPUVMDockerCommandRunner(DockerCommandRunner):
    """A Docker command runner for TPU VMs that overwrites IP addresses."""
    def __init__(self, internal_ip, external_ip, docker_config, **common_args):
        super().__init__(docker_config, **common_args)
        self.ssh_command_runner = TPUVMSSHCommandRunner(internal_ip, external_ip, **common_args)


class TPUPodCommandRunner:
    """TODO"""

    def __init__(
        self,
        instance: GCPTPUNode,
        log_prefix: str,
        node_id: str,
        auth_config: Dict[str, Any],
        cluster_name: str,
        process_runner: ModuleType,
        use_internal_ip: bool,
        docker_config: Optional[Dict[str, Any]] = None):
        def create_command_runner(worker_id, internal_ip, external_ip):
            common_args = {
                "internal_ip": internal_ip,
                "external_ip": external_ip,
                "log_prefix": "[worker_{}]".format(worker_id) + log_prefix,
                "node_id": node_id,
                "provider": self,
                "auth_config": auth_config,
                "cluster_name": cluster_name,
                "process_runner": process_runner,
                "use_internal_ip": use_internal_ip,
            }
            if docker_config and docker_config["container_name"] != "":
                return TPUVMDockerCommandRunner(docker_config, **common_args)
            else:
                return TPUVMSSHCommandRunner(**common_args)

        self._command_runners = []
        self._num_workers = instance.get_num_workers()
        for i in range(self._num_workers):
            self._command_runners.append(
                create_command_runner(
                    worker_id=i,
                    internal_ip=instance.get_internal_ip(i),
                    external_ip=instance.get_external_ip(i)))

    def run(self, *args, **kwargs) -> str:
        with ThreadPoolExecutor(
            _MAX_NUM_CONCURRENT_ACTIVE_CONNECTIONS) as executor:
            results = executor.map(
                lambda i: self._command_runners[i].run(*args, **kwargs),
                range(self._num_workers))
        # What's the right way to consolidate these results?
        #return "hi"
        return list(results)[0]

    def run_rsync_up(self, *args, **kwargs) -> None:
        with ThreadPoolExecutor(
            _MAX_NUM_CONCURRENT_ACTIVE_CONNECTIONS) as executor:
            executor.map(
                lambda i: self._command_runners[i].run_rsync_up(*args, **kwargs),
                range(self._num_workers))

    def run_rsync_down(self, *args, **kwargs) -> None:
        """Rsync files down from the cluster node.

        Args:
            source: The (remote) source directory or file.
            target: The (local) destination path.
        """
        with ThreadPoolExecutor(
            _MAX_NUM_CONCURRENT_ACTIVE_CONNECTIONS) as executor:
            executor.map(
                lambda i: self._command_runners[i].run_rsync_down(*args, **kwargs),
                range(self._num_workers))

    def remote_shell_command_str(self) -> str:
        """Return the command the user can use to open a shell."""
        # This may not be what we expect.
        return self._command_runners[0].remote_shell_command_str()

    def run_init(self, *args, **kwargs) -> Optional[bool]:
        """Used to run extra initialization commands.

        Args:
            as_head: Run as head image or worker.
            file_mounts: Files to copy to the head and worker nodes.
            sync_run_yet: Whether sync has been run yet.

        Returns:
            optional: Whether initialization is necessary.
        """
        with ThreadPoolExecutor(
            _MAX_NUM_CONCURRENT_ACTIVE_CONNECTIONS) as executor:
            results = executor.map(
                lambda i: self._command_runners[i].run_init(*args, **kwargs),
                range(self._num_workers))
        # May not work as expected
        return any(results)