import subprocess

import ray
import sys
from ray._private.test_utils import (
    wait_for_condition,
    get_metric_check_condition,
    MetricSamplePattern,
)
from ray.cluster_utils import AutoscalingCluster
from ray.autoscaler.node_launch_exception import NodeLaunchException


def test_ray_status_e2e(shutdown_only):
    cluster = AutoscalingCluster(
        head_resources={"CPU": 0},
        worker_node_types={
            "type-i": {
                "resources": {"CPU": 1, "fun": 1},
                "node_config": {},
                "min_workers": 1,
                "max_workers": 1,
            },
            "type-ii": {
                "resources": {"CPU": 1, "fun": 100},
                "node_config": {},
                "min_workers": 1,
                "max_workers": 1,
            },
        },
    )

    try:
        cluster.start()
        ray.init(address="auto")

        @ray.remote(num_cpus=0, resources={"fun": 2})
        class Actor:
            def ping(self):
                return None

        actor = Actor.remote()
        ray.get(actor.ping.remote())

        assert "Demands" in subprocess.check_output("ray status", shell=True).decode()
        assert (
            "Total Demands"
            not in subprocess.check_output("ray status", shell=True).decode()
        )
        assert (
            "Total Demands"
            in subprocess.check_output("ray status -v", shell=True).decode()
        )
        assert (
            "Total Demands"
            in subprocess.check_output("ray status --verbose", shell=True).decode()
        )
    finally:
        cluster.shutdown()

