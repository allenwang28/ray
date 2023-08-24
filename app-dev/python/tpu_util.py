import ray
import time

from typing import Any, Dict, List, Optional


@ray.remote(num_cpus=1)
class TPUDict:
  def __init__(self):
    self._tpu_group_availability_map = {}
    self._initialized = False

  def mark_initialized(self):
    self._initialized = True

  def is_initialized(self):
    return self._initialized

  def get_existing_count(self) -> int:
    return len(self._tpu_group_availability_map)

  def get_keys(self) -> List[str]:
    return self._tpu_group_availability_map.keys()

  def has_entry(self, tpu_id: str) -> bool:
   return tpu_id in self._tpu_group_availability_map

  def add_entry(self, tpu_id: str, hosts: int):
    print(
      f"DEBUG: Adding entry. "
      f"tpu_group_availability_map: {self._tpu_group_availability_map}")
    self._tpu_group_availability_map[tpu_id] = dict(
      hosts=hosts,
      available=True
    )
    print(
      f"DEBUG: Done adding entry. "
      f"tpu_group_availability_map: {self._tpu_group_availability_map}")

  def remove_entry(self, tpu_id: str):
    print(
      f"DEBUG: Removing entry. "
      f"tpu_group_availability_map: {self._tpu_group_availability_map}")
    self._tpu_group_availability_map[tpu_id].pop(tpu_id)
    print(
      f"DEBUG: Done removing entry. "
      f"tpu_group_availability_map: {self._tpu_group_availability_map}")

  def claim_resource(self, group_id: str) -> bool:
    """Claims a group id.

    This is a write operation.

    Returns:
      bool, representing whether or not the resource was successfully
      claimed.

    """
    print(f"DEBUG: claiming resource: {group_id}")
    print(f"availability map: {self._tpu_group_availability_map}")
    if group_id not in self._tpu_group_availability_map:
      logging.warning(f"{group_id} is not a valid group_id.")
      return False
    print(f"DEBUG: {group_id} should be valid")
    entry = self._tpu_group_availability_map[group_id]
    print(f"DEBUG: entry: {entry}")
    if not entry["available"]:
      logging.warning(f"{group_id} is not available.")
      return False
    
    entry["available"] = False
    return True

  def get_valid_group_id(self, num_hosts: int) -> Optional[int]:
    """Returns all group ids that are perfect matches.

    This is a read operation.

    Args:
      num_hosts: int, the number of hosts 

    Returns:
      List[str], representing the group ids that are perfect matches

    """
    print(f"Trying to get valid group id from {self._tpu_group_availability_map}")
    valid_entries = list(filter(
      lambda item: item[1]["hosts"] and item[1]["available"],
      self._tpu_group_availability_map.items()))
    candidate_ids = list(map(lambda entry: entry[0], valid_entries))
    for candidate_id in candidate_ids:
      claimed = self.claim_resource(candidate_id)
      if claimed:
        return candidate_id
    return None


@ray.remote(num_cpus=1)
def periodically_update_resources(resource_map: TPUDict, update_rate_in_s: int):
  """Syncs internal representation of resources with the Raylet GCS."""
  print("Starting to periodically update.")
  while True:
    cluster_resources = ray.available_resources()
    tpu_resources = dict(filter(
      lambda item: item[0].endswith("-tpu"),
      cluster_resources.items()))

    # Add any new resources
    for tpu_id in tpu_resources.keys():
      has_entry = ray.get(resource_map.has_entry.remote(tpu_id))
      if not has_entry:
        resource_map.add_entry.remote(tpu_id=tpu_id, hosts=int(tpu_resources[tpu_id]))

    resource_map.mark_initialized.remote()
    time.sleep(update_rate_in_s)


@ray.remote(num_cpus=1)
class GlobalTPUManager:
  """A utility manager for TPU pod interactions.

  TODO: Detail about this

  _tpu_group_availability_map: A dictionary where keys
    are all TPU group IDs in the Ray cluster and values are
    another dictionary denoting the number of resources and whether
    or not it's being utilized. E.g.:
    {
        "tpu_0": {
            "num_hosts": 4, # v4-32
            "available": False,
        }
        ...
    }

  """
  def __init__(self):
    print("DEBUG: Instantiating GlobalTPUManager")
    self._tpu_dict = TPUDict.remote()
    self._tpu_group_availability_map = {}
    self._periodic_update_handle = periodically_update_resources.remote(
      self._tpu_dict, 5)

  def _make_resource_request(self):
    """Requests more resources from Ray."""
    print("Requesting more resources.")
    num_existing_hosts = ray.get(self._tpu_dict.get_existing_count.remote())
    resource_request = [{'TPU': 4}] * (num_existing_hosts + 1)
    ray.autoscaler.sdk.request_resources(bundles=resource_request)

  def schedule_actors_on_tpu_pod(
    self, num_hosts: int, actor_def: Any) -> ray.util.ActorPool:
    """Schedules actors on a TPU pod."""
    requested_resources = False
    while True:
      candidate_id = ray.get(
        self._tpu_dict.get_valid_group_id.remote(num_hosts))
      print(f"Candidate id: {candidate_id}")
      if not candidate_id:
        # make resource request and restart
        if not requested_resources and ray.get(self._tpu_dict.is_initialized.remote()):
          self._make_resource_request()
          requested_resources = True
        time.sleep(5)
      else:
        actors = [actor_def.options(resources={candidate_id: 1}).remote(i) for i in range(num_hosts)] 
        return ray.util.ActorPool(actors)
