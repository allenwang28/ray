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
cluster_name = "tpupodtest"

provider = GCPNodeProvider(
  provider_config=config,
  cluster_name=cluster_name)


# Need a test case for duplicated compute instances
resources = provider.non_terminated_nodes({})
print(resources)
node_id = resources[1]
print(provider._get_node(node_id).get_labels())

node_type = provider.node_tags(node_id).get("ray-user-node-type")
print(node_type)


resource = {'TPU': 1}
if 'TPU' in resource.keys():
  resource["test_abc"] = 1
print(resource)
