import ray

print("importing tpu_util")
tpu_remote = None


@ray.remote(num_cpus=1)
class TPURemoteJobScheduler:
  """A utility Ray actor for scheduling workloads on TPUs.

  Attributes:
    raylet_schedule_wait_in_s: `int`,
    enable_autoscaling: `bool`, whether or not to enable TPU Pod autoscaling.

  """

  def __init__(self,
               raylet_schedule_wait_in_s: int = 3,
               enable_autoscaling: bool = False):
    self._raylet_schedule_wait_in_s = raylet_schedule_wait_in_s
    self._enable_autoscaling = enable_autoscaling
    self._last_scheduled = None
    self._locked = False

  def run_on_tpu_pod(self, ray_remote, requested_tpu_hosts: int):
    import time 
    while True:
      while self._locked:
        time.sleep(5)

      self._locked = True

      cluster_resources = ray.available_resources()
      available_tpu_resources = dict(filter(
        lambda item: item[0].endswith("-tpu"),
        cluster_resources.items()))
      matching_resources = dict(
          filter(lambda item: item[1] == requested_tpu_hosts,
                  available_tpu_resources.items()))

      if not matching_resources:
        print("No matching resources were found. Waiting..")
        time.sleep(10)
      else:
        print("Matching resources ", matching_resources)
        tpu_id = list(matching_resources.keys())[0]
        retval = [
          ray_remote.options(resources={tpu_id: 1}).remote() for _ in range(requested_tpu_hosts)
        ]
        # Wait for a bit so the Raylets get scheduled...
        time.sleep(self._raylet_schedule_wait_in_s)
        self._locked = False
        return retval


def run_on_tpu_pod(ray_remote, requested_tpu_hosts: int):
  if not tpu_remote:
    print("Creating TPURemoteJobScheduler")
    tpu_remote = TPURemoteJobScheduler.remote()
  return tpu_remote.run_on_tpu_pod.remote(
    ray_remote, requested_tpu_hosts=requested_tpu_hosts)


def get_tpu_pod_results(handle):
  handles = ray.get(handle)
  return ray.get(handles)

