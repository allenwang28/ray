import ray
import util

ray.init("auto", ignore_reinit_error=True)

print("ray initialized")

@ray.remote(resources={"TPU": 1})
def func_a():
  import socket
  import time
  import jax
  print("running func_a")
  print(socket.gethostname(), ": hi func_a")
  return jax.device_count()

@ray.remote(resources={"TPU": 1})
def func_b():
  import socket
  import time
  import jax
  print("running func_b")
  print(socket.gethostname(), ": hi func_b")
  return jax.local_device_count()


a_pod_remote = util.run_on_tpu_pod(func_a, requested_tpu_hosts=2, enable_autoscaling=True)
b_pod_remote = util.run_on_tpu_pod(func_b, requested_tpu_hosts=2, enable_autoscaling=True)

print(util.get_tpu_pod_results(a_pod_remote))
print(util.get_tpu_pod_results(b_pod_remote))

ray.shutdown()
