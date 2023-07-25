import inspect
import time
import os

import pytest
import shutil
import sys
from unittest.mock import MagicMock

import ray
from ray import tune
from ray.air import CheckpointConfig
from ray.cluster_utils import Cluster

from ray.util import tpu_util


def _start_new_cluster():
    cluster = Cluster(
        initialize_head=True,
        connect=True,
        head_node_args={
            "num_cpus": 1,
            "_system_config": {
                "health_check_initial_delay_ms": 0,
                "health_check_period_ms": 1000,
                "health_check_failure_threshold": 10,
            },
        },
    )
    return cluster


@pytest.fixture
def start_connected_cluster_with_tpu_pods():
    """Starts head with two fake TPU pods attached."""
    cluster = Cluster(
        initialize_head=True,
        connect=True,
        head_node_args={
            "num_cpus": 0,
            "_system_config": {
                "health_check_initial_delay_ms": 0,
                "health_check_period_ms": 1000,
                "health_check_failure_threshold": 10,
            },
        },
    )
    cluster.add_node(num_cpus=1, resources={"TPUPOD-1": 4})
    cluster.add_node(num_cpus=1, resources={"TPUPOD-1": 4})
    cluster.add_node(num_cpus=1, resources={"TPUPOD-2": 4})
    cluster.add_node(num_cpus=1, resources={"TPUPOD-2": 4})
    cluster.wait_for_nodes()
    yield cluster
    # The code after the yield will run as teardown code.

    ray.shutdown()
    cluster.shutdown()


def test_smoke(start_connected_cluster_with_tpu_pods):
  """Test."""

  @ray.remote
  def run_on_tpu():
    return "hi"
  
  print(ray.get_runtime_context().get())
  cluster = start_connected_cluster_with_tpu_pods
  print(ray.available_resources())
  print(tpu_util.get_available_tpu_pod_resources())
  for node in cluster.list_all_nodes():
    print(node.get_resource_spec().resources)

  handles = tpu_util.run_remote(run_on_tpu, requested_tpu_resources=8)
  [print(ray.get(handle)) for handle in handles]
  # Check that it lands on the right nodes...
  for node in cluster.list_all_nodes():
    print(node.get_resource_spec().resources)


if __name__ == "__main__":
    import pytest

    # Run 1 if failure
    # Run 2 if success

    #sys.exit(pytest.main(["-v", "-rx", "--reruns", "1", __file__]))
    sys.exit(pytest.main(["-v", "-rP", "--reruns", "1", __file__]))
