from io import BytesIO  # pylint:disable=g-importing-member
from typing import List
from fastapi import FastAPI  # pylint:disable=g-importing-member
from fastapi.responses import Response  # pylint:disable=g-importing-member
from ray import serve

import time

from typing import Any, Callable, Iterable, List, Tuple, Union
from tpu_util import GlobalTPUManager

import ray

# The number of hosts within the TPU VM pod slice.
_NUM_HOSTS_PER_POD_SLICE = 2
_MAX_BATCH_SIZE = 4


app = FastAPI()


@serve.deployment(num_replicas=1)
class TPUManager:
  def __init__(self):
    self._tpu_manager = GlobalTPUManager.remote()

  def schedule_actors_on_pod(
    self, num_hosts: int, actor_def) -> List[Any]:
    ref = self._tpu_manager.schedule_actors_on_tpu_pod.remote(
      num_hosts=num_hosts,
      actor_def=actor_def)
    return ray.get(ref)


@serve.deployment(num_replicas=1, route_prefix="/")
@serve.ingress(app)
class APIIngress:
  """`APIIngress`, e.g. the request router/load balancer.

  Attributes:
    handle: The handle to the TPUModelServer deployment.

  """

  def __init__(self, handle) -> None:
    self.handle = handle

  @app.get(
      "/generate",
      responses={200: {"content": {"image/png": {}}}},
      response_class=Response,
  )
  async def generate(self, prompt: str):
    """Requests the generation of an individual prompt.

    Args:
      prompt: An individual prompt.

    Returns:
      A Response.

    """
    handle = await self.handle.generate.remote(prompt)
    return await handle


@ray.remote(num_tpus=4)
class TPUShard:
  """Representation of a single TPU VM worker."""

  def __init__(self, worker_id: int):
    import socket
    self._worker_id = worker_id

  def generate(self, prompts: Iterable[str]) -> List[str]:
    import time
    import socket

    # Simulate latency
    time.sleep(3)
    results = []

    for prompt in prompts:
      results.append(
        f"[{socket.gethostname()}] sample from worker {self._worker_id}"
        f" for prompt {prompt}"
      )
    return results


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "target_num_ongoing_requests_per_replica": _MAX_BATCH_SIZE,
    })
class TPUModelServer():
  """Representation of a TPU model server.

  This handles TPU shards, effectively giving us an SPMD backend.
  """

  def __init__(self, handle,  num_hosts: int = _NUM_HOSTS_PER_POD_SLICE):
    self.handle = handle
    self.num_hosts = num_hosts
    self._shards_ref = handle.schedule_actors_on_pod.remote(num_hosts, TPUShard)

  @serve.batch(batch_wait_timeout_s=10, max_batch_size=_MAX_BATCH_SIZE)
  async def batched_generate_handler(self, prompts: Iterable[str]) -> List[str]:
    original_num_prompts = len(prompts)
    num_to_pad = _MAX_BATCH_SIZE - len(prompts)
    prompts += [""] * num_to_pad

    shards_ref = ray.get(await self._shards_ref)

    results_refs = []
    shard_results = []
    for shard in shards_ref:
      results_refs.append(shard.generate.remote(prompts))

    for results_ref in results_refs:
      result = ray.get(results_ref)
      shard_results.append(ray.get(results_ref))

    results = []
    # shard_results is of shape [num_hosts, num_prompts]
    for prompt_index in range(original_num_prompts):
      result = ""
      for host_index in range(self.num_hosts):
        result += shard_results[host_index][prompt_index] + " "
      results.append(result)

    return results

  async def generate(self, prompt: str) -> str:
    return await self.batched_generate_handler(prompt)

  def exit(self):
    for actor in self._actors:
      actor.exit.remote()

tpu_manager_handle = TPUManager.bind()
tpu_model_server_handle = TPUModelServer.bind(tpu_manager_handle)
deployment = APIIngress.bind(tpu_model_server_handle)
