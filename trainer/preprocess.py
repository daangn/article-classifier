#-*- coding: utf-8 -*-
# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Example dataflow pipeline for preparing image training data.

The tool requires two main input files:

'input' - URI to csv file, using format:
gs://image_uri1,labela,labelb,labelc
gs://image_uri2,labela,labeld
...

'input_dict' - URI to a text file listing all labels (one label per line):
labela
labelb
labelc

The output data is in format accepted by Cloud ML framework.

This tool produces outputs as follows.
It creates one training example per each line of the created csv file.
When processing CSV file:
- all labels that are not present in input_dict are skipped

To execute this pipeline locally using default options, run this script
with no arguments. To execute on cloud pass single argument --cloud.

To execute this pipeline on the cloud using the Dataflow service and non-default
options:
python -E preprocess.py \
--input_path=PATH_TO_INPUT_CSV_FILE \
--input_dict=PATH_TO_INPUT_DIC_TXT_FILE \
--output_path=YOUR_OUTPUT_PATH \
--cloud

For other flags, see PrepareImagesOptions() bellow.

To run this pipeline locally run the above command without --cloud.

TODO(b/31434218)
"""

# TODO(mikehcheng): Beam convention for stage names is CapitalCase as opposed to
# English sentences (eg ReadAndConvertToJpeg as opposed to
# "Read and convert to JPEG"). Fix all samples that don't conform to the
# convention.

# TODO(mikehcheng): Standardize the casing of the various counters (metrics)
# used within this file. So far we have been using underscore_case for metrics.


import argparse
import csv
import datetime
import errno
import io
import logging
import os
import subprocess
import sys
csv.field_size_limit(sys.maxsize)

import apache_beam as beam
from apache_beam.metrics import Metrics
# pylint: disable=g-import-not-at-top
# TODO(yxshi): Remove after Dataflow 0.4.5 SDK is released.
try:
  try:
    from apache_beam.options.pipeline_options import PipelineOptions
  except ImportError:
    from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
  from apache_beam.utils.options import PipelineOptions
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io

from trainer.model import BOTTLENECK_TENSOR_SIZE, WORD_DIM, MAX_WORDS_LENGTH, \
        TOTAL_CATEGORIES_COUNT, MAX_USERNAME_CHARS_COUNT
from trainer.emb import id_to_path, ID_COL, LABEL_COL, IMAGES_COUNT_COL

error_count = Metrics.counter('main', 'errorCount')
missing_label_count = Metrics.counter('main', 'missingLabelCount')
csv_rows_count = Metrics.counter('main', 'csvRowsCount')
skipped_empty_line = Metrics.counter('main', 'skippedEmptyLine')
unlabeled_image = Metrics.counter('main', 'unlabeled_image')
unknown_label = Metrics.counter('main', 'unknown_label')
empty_imgs_count = Metrics.counter('main', 'empty_imgs_count')
no_texts_count = Metrics.counter('main', 'no_texts_count')
empty_imgs_count = Metrics.counter('main', 'empty_imgs_count')
labels_counters = []
for i in range(11):
    labels_counters.append(Metrics.counter('main', "LabelsCount%d" % i))


class Default(object):
  """Default values of variables."""
  FORMAT = 'jpeg'


class ExtractLabelIdsDoFn(beam.DoFn):
  """Extracts (uri, label_ids) tuples from CSV rows.
  """

  def start_bundle(self, context=None):
    self.label_to_id_map = {}

  # The try except is for compatiblity across multiple versions of the sdk
  def process(self, row, all_labels):
    try:
      row = row.element
    except AttributeError:
      pass
    if not self.label_to_id_map:
      for i, label in enumerate(all_labels):
        label = label.strip()
        self.label_to_id_map[label] = i

    # Row format is: image_uri(,label_ids)*
    if not row:
      skipped_empty_line.inc()
      return

    csv_rows_count.inc()

    # In a real-world system, you may want to provide a default id for labels
    # that were not in the dictionary.  In this sample, we simply skip it.
    # This code already supports multi-label problems if you want to use it.
    label_ids = []
    label = row[LABEL_COL].strip()
    try:
        label_id = self.label_to_id_map[label]
        label_ids.append(label_id)
        labels_counters[label_id].inc()
    except IndexError as e:
        logging.error("total labels count: %d, invalid label_id: %d",
                len(labels_counters), label_id)
        raise e
    except KeyError:
        unknown_label.inc()

    if not label_ids:
      unlabeled_image.inc()

    yield row, label_ids


class ReadImageAndConvertToJpegDoFn(beam.DoFn):
  """Read files from GCS and convert images to JPEG format.

  We do this even for JPEG images to remove variations such as different number
  of channels.
  """
  SHAPE = [1, 1, 1, BOTTLENECK_TENSOR_SIZE]

  def __init__(self, emb_path):
    self._emb_path = emb_path

  def process(self, element):
    try:
      row, label_ids = element.element
    except AttributeError:
      row, label_ids = element

    id = int(row[ID_COL])
    images_count = int(row[IMAGES_COUNT_COL])

    if images_count < 1:
        embedding = None
    else:
        emb_filepath = "%s/%s" % (self._emb_path, id_to_path(id))
        if not file_io.file_exists(emb_filepath):
            embedding = None
            logging.warn('file is not exists: %s', emb_filepath)
            empty_imgs_count.inc()
        else:
            embedding = self._fetch_embedding(emb_filepath)

    yield row, label_ids, embedding

  def _fetch_embedding(self, emb_filepath):
    try:
        embedding = np.frombuffer(
                file_io.read_file_to_string(emb_filepath),
                dtype=np.float32)
        embedding = embedding.reshape(self.SHAPE)
    except ValueError as e:
        logging.warn('Could not load an embedding file from %s: %s', emb_filepath, str(e))
        error_count.inc()
        if e.message.startswith('cannot reshape array of size 0 into'):
            file_io.delete_file(emb_filepath)
            return
        raise e


class ExtractTextDataDoFn(beam.DoFn):
  def process(self, element):
    try:
      item, label_ids, embedding = element.element
    except AttributeError:
      item, label_ids, embedding = element

    text_embedding_inline = item[12]

    try:
        text_embedding, text_length = self.get_embedding_and_length(text_embedding_inline, MAX_WORDS_LENGTH)
        if text_length < 1:
            no_texts_count.inc()
            logging.error('no text: %s', text_embedding_inline)
    except Exception as e:
        error_count.inc()
        logging.error(text_embedding_inline)
        raise e

    yield item, label_ids, embedding, {
          'text_embedding': text_embedding,
          'text_length': text_length,
          }

  def get_embedding_and_length(self, inline, max_length):
      if inline == '':
          embedding = []
      else:
          embedding = [float(x) for x in inline.split()]
      length = len(embedding) / WORD_DIM
      if length > max_length:
          length = max_length
          embedding = embedding[:WORD_DIM * length]
      else:
          embedding += [0.0] * ((max_length - length) * WORD_DIM)
      return embedding, length


class TFExampleFromImageDoFn(beam.DoFn):
  def __init__(self):
    self._empty_embedding = [0.0] * BOTTLENECK_TENSOR_SIZE

  def process(self, element):

    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    try:
      element = element.element
    except AttributeError:
      pass
    row, label_ids, embedding, data = element

    id = row[ID_COL]
    category_id = int(row[2])
    price = int(row[3])
    images_count = int(row[4])
    created_at_ts = long(row[5])
    offerable = int(row[6])
    recent_articles_count = int(row[7])
    blocks_inline = row[8]
    title_length = int(row[9])
    content_length = int(row[10])

    if category_id < 1 or category_id - 1 > TOTAL_CATEGORIES_COUNT:
        error_count.inc()
        raise 'invalid catgory_id: %d' % category_id

    if embedding is None:
        embedding = self._empty_embedding
    else:
        embedding = embedding.ravel().tolist()

    username = row[11].decode('utf-8')
    username_length = len(username)
    username_chars = list(username)

    if username_length < MAX_USERNAME_CHARS_COUNT:
        username_chars += [u''] * (MAX_USERNAME_CHARS_COUNT - username_length)
    elif username_length > MAX_USERNAME_CHARS_COUNT:
        username_chars = username_chars[0:MAX_USERNAME_CHARS_COUNT]

    username_chars = [x.encode('utf-8') for x in username_chars]

    example = tf.train.Example(features=tf.train.Features(feature={
        'id': _bytes_feature([id]),
        'embedding': _float_feature(embedding),
        'text_embedding': _float_feature(data['text_embedding']),
        'text_length': _int_feature([data['text_length']]),
        'category_id': _int_feature([category_id]),
        'price': _int_feature([price]),
        'images_count': _int_feature([images_count]),
        'recent_articles_count': _int_feature([recent_articles_count]),
        'title_length': _int_feature([title_length]),
        'content_length': _int_feature([content_length]),
        'blocks_inline': _bytes_feature([blocks_inline]),
        'username_chars': _bytes_feature(username_chars),
        'username_length': _int_feature([username_length]),
        'created_at_ts': _int_feature([created_at_ts]),
        'offerable': _int_feature([offerable]),
        'label': _int_feature(label_ids),
    }))

    yield example


def configure_pipeline(p, opt):
  """Specify PCollection and transformations in pipeline."""
  read_input_source = beam.io.ReadFromText(
      opt.input_path, strip_trailing_newlines=True)
  read_label_source = beam.io.ReadFromText(
      opt.input_dict, strip_trailing_newlines=True)
  labels = (p | 'Read dictionary' >> read_label_source)

  _ = (p
       | 'Read input' >> read_input_source
       | 'Parse input' >> beam.Map(lambda line: csv.reader([line.encode('utf-8')]).next())
       | 'Extract label ids' >> beam.ParDo(ExtractLabelIdsDoFn(),
                                           beam.pvalue.AsIter(labels))
       | 'Read and convert to JPEG'
       >> beam.ParDo(ReadImageAndConvertToJpegDoFn(opt.emb_path))
       | 'Extract text data' >> beam.ParDo(ExtractTextDataDoFn())
       | 'Embed and make TFExample' >> beam.ParDo(TFExampleFromImageDoFn())
       # TODO(b/35133536): Get rid of this Map and instead use
       # coder=beam.coders.ProtoCoder(tf.train.Example) in WriteToTFRecord
       # below.
       | 'SerializeToString' >> beam.Map(lambda x: x.SerializeToString())
       | 'Save to disk'
       >> beam.io.WriteToTFRecord(opt.output_path,
                                  file_name_suffix='.tfrecord.gz'))


def run(in_args=None):
  """Runs the pre-processing pipeline."""

  pipeline_options = PipelineOptions.from_dictionary(vars(in_args))
  with beam.Pipeline(options=pipeline_options) as p:
    configure_pipeline(p, in_args)


def default_args(argv):
  """Provides default values for Workflow flags."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--input_path',
      required=True,
      help='Input specified as uri to CSV file. Each line of csv file '
      'contains colon-separated GCS uri to an image and labels.')
  parser.add_argument(
      '--emb_path',
      default='data/image_embeddings',
      help='')
  parser.add_argument(
      '--input_dict',
      required=True,
      help='Input dictionary. Specified as text file uri. '
      'Each line of the file stores one label.')
  parser.add_argument(
      '--output_path',
      required=True,
      help='Output directory to write results to.')
  parser.add_argument(
      '--project',
      type=str,
      help='The cloud project name to be used for running this pipeline')

  parser.add_argument(
      '--job_name',
      type=str,
      default='flowers-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
      help='A unique job identifier.')
  parser.add_argument(
      '--num_workers', default=3, type=int, help='The number of workers.')
  parser.add_argument('--cloud', default=False, action='store_true')
  parser.add_argument(
      '--runner',
      help='See Dataflow runners, may be blocking'
      ' or not, on cloud or not, etc.')
#  parser.add_argument(
#      '--extra_package', type=str, help='')
  parser.add_argument(
      '--setup_file', type=str, help='')
  parser.add_argument(
      '--autoscaling_algorithm', default='THROUGHPUT_BASED', type=str, help='')
  parser.add_argument(
      '--max_num_workers', default=30, type=int, help='')

  parsed_args, _ = parser.parse_known_args(argv)

  if parsed_args.cloud:
    # Flags which need to be set for cloud runs.
    default_values = {
        'project':
            get_cloud_project(),
        'temp_location':
            os.path.join(os.path.dirname(parsed_args.output_path), 'temp'),
        'runner':
            'DataflowRunner',
        'save_main_session':
            True,
    }
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  else:
    # Flags which need to be set for local runs.
    default_values = {
        'runner': 'DirectRunner',
    }
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


  for kk, vv in default_values.iteritems():
    if kk not in parsed_args or not vars(parsed_args)[kk]:
      vars(parsed_args)[kk] = vv

  return parsed_args


def get_cloud_project():
  cmd = [
      'gcloud', '-q', 'config', 'list', 'project',
      '--format=value(core.project)'
  ]
  with open(os.devnull, 'w') as dev_null:
    try:
      res = subprocess.check_output(cmd, stderr=dev_null).strip()
      if not res:
        raise Exception('--cloud specified but no Google Cloud Platform '
                        'project found.\n'
                        'Please specify your project name with the --project '
                        'flag or set a default project: '
                        'gcloud config set project YOUR_PROJECT_NAME')
      return res
    except OSError as e:
      if e.errno == errno.ENOENT:
        raise Exception('gcloud is not installed. The Google Cloud SDK is '
                        'necessary to communicate with the Cloud ML service. '
                        'Please install and set up gcloud.')
      raise


def main(argv):
  arg_dict = default_args(argv)
  run(arg_dict)


if __name__ == '__main__':
  main(sys.argv[1:])
