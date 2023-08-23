import ray
import time
from tpu_util import GlobalTPUManager

ray.init(runtime_env={
  "working_dir": ".",
})

@ray.remote(num_tpus=4)
class SampleActorShard:

  def __init__(self, worker_id: int):
    import socket
    self._worker_id = worker_id
    self._host = socket.gethostname()
    print(f"[{self._host}] Initializing {worker_id}")

  def say_something(self, something: str):
    return f"{self._worker_id} says {something}."


@ray.remote(num_cpus=1)
class ActorHandler:

  def __init__(self, num_hosts: int, scheduling_fn):
    print("DEBUG: ActorHandler instantiated")
    self.num_hosts = num_hosts
    self._shard_pool = scheduling_fn(num_hosts, SampleActorShard)

  def say_something(self, something: str):
    print("DEBUG: ActorHandler saying something")
    somethings = [something] * self.num_hosts
    return list(self._shard_pool.map(lambda a, v: a.say_something.remote(v), somethings))


global_tpu_manager = GlobalTPUManager.remote()

def schedule_actors_on_tpu_pod(
  num_hosts: int, actor_def) -> ray.util.ActorPool:
  print("trying to schedule actors now")
  return ray.get(global_tpu_manager.schedule_actors_on_tpu_pod.remote(
    num_hosts=num_hosts,
    actor_def=actor_def
  ))


num_hosts = 2
print("Creating actor handlers")
handler1 = ActorHandler.remote(num_hosts=num_hosts, scheduling_fn=schedule_actors_on_tpu_pod)
handler2 = ActorHandler.remote(num_hosts=num_hosts, scheduling_fn=schedule_actors_on_tpu_pod)

time.sleep(5)

print(ray.get(handler1.say_something.remote("helllllo there!")))
print(ray.get(handler2.say_something.remote("what?")))

