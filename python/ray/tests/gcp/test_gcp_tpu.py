from ray.autoscaler._private.gcp.node_provider import GCPNodeProvider

from ray.autoscaler._private.gcp.node import (
    GCPCompute,
    GCPNode,
    GCPNodeType,
    GCPResource,
    GCPTPU,
)
from googleapiclient import discovery, errors
import subprocess

config = {
  "type": "gcp",
  "region": "us-central2",
  "availability_zone": "us-central2-b",
  "project_id": "mlperf-high-priority-project",
  "_has_tpus": True,
}
cluster_name = "tputest"

GCPNodeProvider.bootstrap_config(cluster_config={})

provider = GCPNodeProvider(
  provider_config=config,
  cluster_name=cluster_name)


# Need a test case for duplicated compute instances
resources = provider.non_terminated_nodes({})

auth_config = {
  "ssh_user": "ubuntu",
  "ssh_private_key": "~/.ssh/id_rsa",
}

# Need a test for GCPTPUNode instance for get external/internal IPs
command_runner = provider.get_command_runner(
    log_prefix="test: ",
    node_id=resources[0],
    auth_config=auth_config,
    cluster_name=cluster_name,
    process_runner=subprocess,
    use_internal_ip=True,
    docker_config=None)


print(command_runner.run(
  cmd="echo hi",
  with_output=True,
))
