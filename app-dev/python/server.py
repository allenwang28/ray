from io import BytesIO  # pylint:disable=g-importing-member
from typing import List
from fastapi import FastAPI  # pylint:disable=g-importing-member
from fastapi.responses import Response  # pylint:disable=g-importing-member
from ray import serve

import time

from typing import Any, Callable, Iterable, List, Tuple, Union
from tpu_util import RayActorDefinition, RayActorInstantiation, get_id_and_actor_handles

import ray

ray.init()

app = FastAPI()

# The number of hosts within the TPU VM pod slice.
_NUM_HOSTS_PER_POD_SLICE = 2


@serve.deployment(num_replicas=1, route_prefix="/")
@serve.ingress(app)
class APIIngress:
  """`APIIngress`, e.g. the request router/load balancer.

  Attributes:
    handle: The handle to the TPUModelServer deployment.

  """

  def __init__(self, handle) -> None:
    self.handle = model_server_handle

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
    return await self.handle.generate(prompt)


@ray.remote(num_tpus=4)
class TPUShard:
  """Representation of a single TPU VM worker."""

  def __init__(self, worker_id: int):
    self._worker_id = worker_id

  def generate(self, prompt: str) -> str:
    import time

    # Simulate latency
    time.sleep(3)
    return f"sample from worker {self._worker_id} for prompt {prompt}"


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "target_num_ongoing_requests_per_replica": _MAX_BATCH_SIZE,
    })
class TPUModelServer():
  """Representation of a TPU model server.

  This is the main handler for a 

  """

  def __init__(self, num_hosts: _NUM_HOSTS_PER_POD_SLICE):
    # Blocking operation - we can't continue without this
    self.num_hosts = num_hosts
    self._id, self._actors = global_tpu_manager.get_id_and_actor_handles(
      num_hosts=self.num_hosts,
      actor_def=TPUShard,
    )

  def combine_results(self, results: Iterable[str]):
    pass

  def generate(self):
    pass

  def exit(self):
    for actor in self._actors:
      actor.exit.remote()

tpu_model_server_handle = TPUModelServer.bind()
deployment = APIIngress(tpu_model_server_handle)
