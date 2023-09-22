

tpu = discovery.build(
        "tpu",
        "v2alpha",
        credentials=None,
        cache_discovery=False,
        discoveryServiceUrl="https://tpu.googleapis.com/$discovery/rest",
)
parent = "projects/mlperf-high-priority-project/locations/us-central2-b"
"""
config = {
  "acceleratorType": "v4-8",
  "runtimeVersion": "tpu-vm-v4-base",
}
print(dir(tpu.projects().locations().queuedResources()))
"""


gcp_tpu = GCPTPU(resource=tpu, project_id='mlperf-high-priority-project',
                 availability_zone='us-central2-b', cluster_name='tputest')
base_config = {
  'acceleratorType': 'v4-16',
  'runtimeVersion': 'tpu-vm-v4-base',
}
labels = {
  'ray-node-name': 'allencwang-ray-test',
}

#print(gcp_tpu.create_instance(base_config=base_config, labels=labels, wait_for_operation=True))
instances = gcp_tpu.list_instances()
print(instances)
instance = instances[0]
print(instance.get_labels())
print(instance.is_running())
print(instance.get('name'))
print(instance.num_nodes)
print(instance.is_pod)
for node in instance.get_nodes():
  print(node)
  print(node.get_external_ip())
#print(instance.get_external_ip())
#print(len(instance.get_info()))
#print(gcp_tpu.get_instance(node_id=instance.get('name')))
"""