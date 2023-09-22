import ray

_TPU_PREFIX = "TPUPOD"


def get_available_tpu_pod_resources():
  cluster_resources = ray.available_resources()
  uid_to_resource_count_map = dict(filter(
    lambda item: item[0].startswith(_TPU_PREFIX), cluster_resources.items()))
  uids = uid_to_resource_count_map.keys
  uids_to_nodes_map = {}
  """
  for uid in uids:
    ray.cluster_resources
    uids_to_nodes_map[uid] = 
  """

  return uid_to_resource_count_map


def run_remote(remote_function, requested_tpu_resources: int):
  assert requested_tpu_resources % 4 == 0
  available_tpu_resources = get_available_tpu_pod_resources()
  matching_resources = dict(filter(lambda item: item[1] == requested_tpu_resources, available_tpu_resources.items()))

  if not matching_resources:
    print("Couldn't find the exact resource matching. Unexpected behavior for now.")
    raise NotImplementedError("TPU Autoscaling not working yet")
  
  # Take the first matching resource
  # TODO - note there's definitely possibility of a race condition here.
  print(matching_resources.keys())
  tpu_tag = list(matching_resources.keys())[0]
  num_calls_needed = requested_tpu_resources // 4
  return [remote_function.options(resources={tpu_tag: 4}).remote() for _ in range(num_calls_needed)]


def remote(*args, **kwargs):
  pass

