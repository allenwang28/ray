# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
from concurrent import futures
import functools

from io import BytesIO  # pylint:disable=g-importing-member
import numpy as np
import requests


_PROMPTS = [
    "hi there",
]


def send_request_and_receive_image(prompt: str, url: str) -> BytesIO:
  """Sends a single prompt request and returns the Image."""
  inputs = "%20".join(prompt.split(" "))
  resp = requests.get(f"{url}?prompt={inputs}")
  print("Got response: ", resp.content)
  return resp.content


def send_requests(num_requests: int, batch_size: int,
                  url: str = "http://localhost:8000/generate"):
  """Sends a list of requests and processes the responses."""
  print("Num requests: ", num_requests)
  print("Batch size: ", batch_size)

  prompts = _PROMPTS
  if num_requests > len(_PROMPTS):
    # Repeat until larger than num_requests
    prompts = _PROMPTS * int(np.ceil(num_requests / len(_PROMPTS)))

  prompts = np.random.choice(
      prompts, num_requests, replace=False)

  with futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
    results = executor.map(
        functools.partial(send_request_and_receive_image, url=url),
        prompts)
    print(list(results))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Sends requests to Diffusion.")
  parser.add_argument(
      "--num_requests", help="Number of requests to send.",
      default=8)
  parser.add_argument(
      "--batch_size", help="The number of requests to send at a time.",
      default=8)

  parser.add_argument(
      "--ip", help="The IP address to send the requests to.")

  args = parser.parse_args()

  address = f"http://{args.ip}:8000/generate"
  send_requests(
      num_requests=int(args.num_requests), batch_size=int(args.batch_size),
      url=address)
