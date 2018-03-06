# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Make a request JSON from local images to send to CloudML serving API.

Example usage:
  $ python images_to_json.py -o request.json img1.jpg img2.jpg
  $ gcloud beta ml predict \
      --model=MY_MODEL_NAME \
      --instances=request.json > response.yaml
"""

import argparse
import base64
from cStringIO import StringIO
import json
import sys

from PIL import Image

desired_width = 299
desired_height = 299


def parse_args():
  """Handle the command line arguments.

  Returns:
    Output of argparse.ArgumentParser.parse_args.
  """

  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--output', default='request.json',
                      help='Output file to write encoded images to.')
  parser.add_argument('-r', '--resize', dest='resize', action='store_true',
                      help='Will resize images locally first.  Not needed, but'
                      ' will reduce network traffic.')
  parser.add_argument('inputs', nargs='+', type=argparse.FileType('r'),
                      help='A list of .emb files to serialize into a '
                      'request json')

  args = parser.parse_args()

  check = lambda filename: filename.lower().endswith(('emb'))
  if not all(check(input_file.name) for input_file in args.inputs):
    sys.stderr.write('All inputs must be .emb')
    sys.exit(1)

  return args


def make_request_json(input_images, output_json, do_resize):
  """Produces a JSON request suitable to send to CloudML Prediction API.

  Args:
    input_images: List of file handles corresponding to images to be encoded.
    output_json: File handle of the output json where request will be written.
    do_resize: Boolean specifying if script should resize images.
  """

  from tensorflow.python.lib.io import file_io
  import numpy as np

  data_path = 'data/emb.csv'
  items = np.genfromtxt(file_io.FileIO(data_path, mode='r'),
      delimiter=',', dtype=None, skip_header=1)
  data_map = {}
  for item in items:
    key = item[0]
    text_embedding = [float(x) for x in item[1].rstrip().split(' ')]
    data_map[key] = {
        'key': str(key),
        'text_embedding': text_embedding,
        'category_id': item[2],
        'price': item[3],
        'images_count': item[4],
        'created_at_ts': item[5],
        'offerable': item[6],
        }

  with open(output_json, 'w') as ff:
    for image_handle in input_images:
      # Uses argparse to check permissions, but ignore pre-opened file handle.
      tokens = image_handle.name.split('/')
      tokens = tokens[-1].split('.')
      id = int(tokens[0])
      data = data_map[id]

      encoded_contents = base64.b64encode(image_handle.read())
      new_data = {'image_embedding_bytes': {'b64': encoded_contents}}
      new_data.update(data)
      row = json.dumps(new_data)

      ff.write(row)
      ff.write('\n')

  print 'Wrote {} images to {}'.format(len(input_images), output_json)


def main():
  args = parse_args()
  make_request_json(args.inputs, args.output, args.resize)


if __name__ == '__main__':
  main()
