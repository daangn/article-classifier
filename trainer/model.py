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
"""Flowers classification model.
"""

import argparse
import logging

import tensorflow as tf
from tensorflow.contrib import layers

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils
from tensorflow.python.lib.io import file_io

import util
from util import override_if_not_in_args

from rnn import stack_bidirectional_dynamic_rnn, simple_rnn, multi_rnn

slim = tf.contrib.slim

LOGITS_TENSOR_NAME = 'logits_tensor'
LABEL_COLUMN = 'label'
EMBEDDING_COLUMN = 'embedding'

TOTAL_CATEGORIES_COUNT = 62
MAX_PRICE = 100000000.0
MAX_IMAGES_COUNT = 14.0
DAY_TIME = 60.0 * 60 * 24

IMAGE_COUNT_SECTION = [1, 2, 5, 10]
PRICE_SECTION = [0, 1000, 10000, 30000, 50000, 10*10000, 30*10000, 100*10000, 1000*10000, 10000*10000]
RECENT_ARTICLES_COUNT_SECTION = [0, 1, 5, 20, 60, 100]

BOTTLENECK_TENSOR_SIZE = 1536
WORD_DIM = 50
MAX_TEXT_LENGTH = 256
TEXT_EMBEDDING_SIZE = WORD_DIM * MAX_TEXT_LENGTH
FEATURES_COUNT = 3
BLOCKS_COUNT = 66
EXTRA_EMBEDDING_SIZE = FEATURES_COUNT + len(PRICE_SECTION) \
    + len(IMAGE_COUNT_SECTION) + len(RECENT_ARTICLES_COUNT_SECTION)


class GraphMod():
  TRAIN = 1
  EVALUATE = 2
  PREDICT = 3


def build_signature(inputs, outputs):
  """Build the signature.

  Not using predic_signature_def in saved_model because it is replacing the
  tensor name, b/35900497.

  Args:
    inputs: a dictionary of tensor name to tensor
    outputs: a dictionary of tensor name to tensor
  Returns:
    The signature, a SignatureDef proto.
  """
  signature_inputs = {key: saved_model_utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}
  signature_outputs = {key: saved_model_utils.build_tensor_info(tensor)
                       for key, tensor in outputs.items()}

  signature_def = signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.PREDICT_METHOD_NAME)

  return signature_def


def create_model():
  """Factory method that creates model to be used by generic task.py."""
  parser = argparse.ArgumentParser()
  # Label count needs to correspond to nubmer of labels in dictionary used
  # during preprocessing.
  parser.add_argument('--label_count', type=int, default=5)
  parser.add_argument('--dropout', type=float, default=0.5)
  parser.add_argument('--input_dict', type=str)
  args, task_args = parser.parse_known_args()
  override_if_not_in_args('--max_steps', '1000', task_args)
  override_if_not_in_args('--batch_size', '100', task_args)
  override_if_not_in_args('--eval_set_size', '370', task_args)
  override_if_not_in_args('--eval_interval_secs', '2', task_args)
  override_if_not_in_args('--log_interval_secs', '2', task_args)
  override_if_not_in_args('--min_train_eval_rate', '2', task_args)
  return Model(args.label_count, args.dropout, args.input_dict), task_args


class GraphReferences(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.examples = None
    self.train = None
    self.global_step = None
    self.metric_updates = []
    self.metric_values = []
    self.keys = None
    self.predictions = []
    self.input_image = None
    self.input_text = None
    self.input_category_id = None
    self.input_price = None
    self.input_images_count = None
    self.input_created_at_ts = None
    self.input_offerable = None
    self.input_recent_articles_count = None
    self.input_blocks_inline = None
    self.input_text_length = None
    self.ids = None
    self.labels = None

def find_nearest_idx(array, value):
    return tf.argmin(tf.abs(
        tf.expand_dims(array, 0) - tf.expand_dims(value, 1)
    ), 1)

def get_extra_embeddings(tensors):
    images_count_section = tf.constant(IMAGE_COUNT_SECTION, dtype=tf.float32)
    price_section = tf.constant(PRICE_SECTION, dtype=tf.float32)
    recent_articles_count_section = tf.constant(RECENT_ARTICLES_COUNT_SECTION, dtype=tf.float32)

    tensors.input_created_at_ts = tf.placeholder(tf.float64, shape=[None])
    tensors.input_offerable = tf.placeholder(tf.float32, shape=[None])

    created_at_ts = tensors.input_created_at_ts
    offerable = tensors.input_offerable

    #price_section = tf.one_hot(find_nearest_idx(price_section, price), len(PRICE_SECTION))
    #images_count_section = tf.one_hot(find_nearest_idx(images_count_section, images_count), len(IMAGE_COUNT_SECTION))
    #recent_articles_count_section = tf.one_hot(find_nearest_idx(
    #    recent_articles_count_section, recent_articles_count), len(RECENT_ARTICLES_COUNT_SECTION))

    #price_norm = tf.minimum(price / MAX_PRICE, 1.0)
    #is_free = tf.cast(tf.equal(price, 0), tf.float32)
    #images_count_norm = tf.minimum(images_count / MAX_IMAGES_COUNT, 1.0)
    created_hour = created_at_ts % DAY_TIME * 1.0 / DAY_TIME
    created_hour = tf.cast(created_hour, tf.float32)
    day = tf.cast(created_at_ts / DAY_TIME % 7 / 7.0, tf.float32)

    #extra_embeddings = tf.concat([price_norm, is_free, images_count_norm, offerable, created_hour, day], 0)
    extra_embeddings = tf.concat([offerable, created_hour, day], 0)
    extra_embeddings = tf.reshape(extra_embeddings, [-1, FEATURES_COUNT])
    #extra_embeddings = tf.concat([extra_embeddings, price_section, images_count_section,
    #    recent_articles_count_section, blocks], 1)
    return extra_embeddings

def blocks_inline_to_matrix(inline):
    with tf.variable_scope("blocks"):
        splited_items = tf.string_split(inline, ' ')
        splited_values = tf.string_split(splited_items.values, '-')
        values = tf.string_to_number(splited_values.values, tf.int32)
        ids = tf.one_hot(values[::2] - 1, BLOCKS_COUNT, dtype=tf.int32)
        counts = values[1::2]
        counts = tf.expand_dims(counts, -1)
        values = counts * ids
        indices = splited_items.indices[:,0]
        inlines_count = tf.shape(inline)[0]
        one_hot_indices = tf.one_hot(indices, inlines_count, dtype=tf.int32)
        results = tf.matmul(tf.transpose(one_hot_indices), values)
        return tf.cast(results, tf.float32)


class Model(object):
  """TensorFlow model for the flowers problem."""

  def __init__(self, label_count, dropout, labels_path):
    self.label_count = label_count
    self.dropout = dropout
    self.labels = file_io.read_file_to_string(labels_path).strip().split('\n')

  def get_labels(self):
      return self.labels

  def id_to_key(self, id):
      return self.labels[id]

  def add_final_training_ops(self,
                             embeddings,
                             all_labels_count,
                             hidden_layer_size=BOTTLENECK_TENSOR_SIZE / 4,
                             dropout_keep_prob=None):
    """Adds a new softmax and fully-connected layer for training.

     The set up for the softmax and fully-connected layers is based on:
     https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

     This function can be customized to add arbitrary layers for
     application-specific requirements.
    Args:
      embeddings: The embedding (bottleneck) tensor.
      all_labels_count: The number of all labels including the default label.
      hidden_layer_size: The size of the hidden_layer. Roughtly, 1/4 of the
                         bottleneck tensor size.
      dropout_keep_prob: the percentage of activation values that are retained.
    Returns:
      softmax: The softmax or tensor. It stores the final scores.
      logits: The logits tensor.
    """
    with tf.name_scope('input'):
      with tf.name_scope('Wx_plus_b'):
        hidden = layers.fully_connected(embeddings, hidden_layer_size)
        # We need a dropout when the size of the dataset is rather small.
        if dropout_keep_prob:
          hidden = tf.nn.dropout(hidden, dropout_keep_prob)
        hidden = layers.fully_connected(hidden, hidden_layer_size)
        if dropout_keep_prob:
          hidden = tf.nn.dropout(hidden, dropout_keep_prob)
        logits = layers.fully_connected(
            hidden, all_labels_count, activation_fn=None)

    softmax = tf.nn.softmax(logits, name='softmax')
    return softmax, logits

  def build_inception_graph(self):
    image_str_tensor = tf.placeholder(tf.string, shape=[None])

    def decode(image_str_tensor):
        embeddings = tf.decode_raw(image_str_tensor, tf.float32)
        return embeddings

    inception_embeddings = tf.map_fn(
        decode, image_str_tensor, back_prop=False, dtype=tf.float32)
    inception_embeddings = tf.reshape(inception_embeddings, [-1, BOTTLENECK_TENSOR_SIZE])
    return image_str_tensor, inception_embeddings

  def build_graph(self, data_paths, batch_size, graph_mod):
    """Builds generic graph for training or eval."""
    tensors = GraphReferences()
    is_training = graph_mod == GraphMod.TRAIN
    if data_paths:
      tensors.keys, tensors.examples = util.read_examples(
          data_paths,
          batch_size,
          shuffle=is_training,
          num_epochs=None if is_training else 2)
    else:
      tensors.examples = tf.placeholder(tf.string, name='input', shape=(None,))

    if graph_mod == GraphMod.PREDICT:
      inception_input, inception_embeddings = self.build_inception_graph()
      embeddings = inception_embeddings

      text_embeddings = tf.placeholder(tf.float32, shape=[None, TEXT_EMBEDDING_SIZE])
      text_lengths = tf.placeholder(tf.int64, shape=[None, 1])
      category_ids = tf.placeholder(tf.int64, shape=[None, 1])
      price = tf.placeholder(tf.int64, shape=[None])
      images_count = tf.placeholder(tf.int64, shape=[None])
      recent_articles_count = tf.placeholder(tf.int64, shape=[None])
      blocks_inline = tf.placeholder(tf.string, shape=[None])

      tensors.input_image = inception_input
      tensors.input_text = text_embeddings
      tensors.input_text_length = text_lengths
      tensors.input_category_id = category_ids
      tensors.input_price = price
      tensors.input_images_count = images_count
      tensors.input_recent_articles_count = recent_articles_count
      tensors.input_blocks_inline = blocks_inline

      extra_embeddings = get_extra_embeddings(tensors)
    else:
      # For training and evaluation we assume data is preprocessed, so the
      # inputs are tf-examples.
      # Generate placeholders for examples.
      with tf.name_scope('inputs'):
        feature_map = {
            'id':
                tf.FixedLenFeature(
                    shape=[], dtype=tf.string, default_value=['']),
            # Some images may have no labels. For those, we assume a default
            # label. So the number of labels is label_count+1 for the default
            # label.
            'label':
                tf.FixedLenFeature(
                    shape=[1], dtype=tf.int64,
                    default_value=[self.label_count]),
            'embedding':
                tf.FixedLenFeature(
                    shape=[BOTTLENECK_TENSOR_SIZE], dtype=tf.float32),
            'text_embedding':
                tf.FixedLenFeature(
                    shape=[TEXT_EMBEDDING_SIZE], dtype=tf.float32),
            'text_length':
                tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'extra_embedding':
                tf.FixedLenFeature(
                    shape=[EXTRA_EMBEDDING_SIZE], dtype=tf.float32),
            'category_id':
                tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'price':
                tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'images_count':
                tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'recent_articles_count':
                tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'blocks_inline':
                tf.FixedLenFeature(shape=[1], dtype=tf.string),
        }
        parsed = tf.parse_example(tensors.examples, features=feature_map)
        labels = tf.squeeze(parsed['label'])
        tensors.labels = labels
        tensors.ids = tf.squeeze(parsed['id'])
        embeddings = parsed['embedding']
        text_embeddings = parsed['text_embedding']
        text_lengths = parsed['text_length']
        extra_embeddings = parsed['extra_embedding']
        category_ids = parsed['category_id']
        price = parsed['price']
        images_count = parsed['images_count']
        recent_articles_count = parsed['recent_articles_count']
        blocks_inline = parsed['blocks_inline']

    with tf.variable_scope("category_embeddings", reuse=tf.AUTO_REUSE):
        category_embeddings = tf.get_variable('table', [TOTAL_CATEGORIES_COUNT, 5])
        category_ids = tf.minimum(category_ids - 1, TOTAL_CATEGORIES_COUNT - 1)
        category_ids = tf.reshape(category_ids, [-1])
        category_embeddings = tf.nn.embedding_lookup(category_embeddings, category_ids)

    with tf.variable_scope("continuos_features", reuse=tf.AUTO_REUSE):
        blocks = blocks_inline_to_matrix(blocks_inline)
        continuos_features = tf.concat([price, images_count, recent_articles_count, blocks], 1)
        continuos_features = tf.cast(continuos_features, tf.float32)
        extra_embeddings = layers.fully_connected(continuos_features, 10,
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params={'is_training': is_training})

    # We assume a default label, so the total number of labels is equal to
    # label_count+1.
    all_labels_count = self.label_count + 1
    with tf.name_scope('final_ops'):
      dropout_keep_prob = self.dropout if is_training else None
      embeddings = layers.fully_connected(embeddings, BOTTLENECK_TENSOR_SIZE / 8)
      extra_embeddings = layers.fully_connected(extra_embeddings, int(EXTRA_EMBEDDING_SIZE / 2),
              normalizer_fn=tf.contrib.layers.batch_norm,
              normalizer_params={'is_training': is_training})
      if dropout_keep_prob:
          embeddings = tf.nn.dropout(embeddings, dropout_keep_prob)
          extra_embeddings = tf.nn.dropout(extra_embeddings, dropout_keep_prob)
          category_embeddings = tf.nn.dropout(category_embeddings, dropout_keep_prob)

      text_embeddings = tf.reshape(text_embeddings, [-1, MAX_TEXT_LENGTH, WORD_DIM])
      text_lengths = tf.reshape(text_lengths, [-1])
      layer_sizes = [WORD_DIM, WORD_DIM*2]
      initial_state = tf.concat([embeddings, extra_embeddings, category_embeddings], 1, name='initial_state')
      initial_state = layers.fully_connected(initial_state, WORD_DIM * 2)
      initial_state = layers.fully_connected(initial_state, WORD_DIM)

      #text_embeddings = multi_rnn(text_embeddings, layer_sizes, text_lengths,
      #        dropout_keep_prob=dropout_keep_prob, attn_length=0,
      #        initial_state=initial_state, base_cell=tf.contrib.rnn.BasicLSTMCell)
      text_embeddings = stack_bidirectional_dynamic_rnn(text_embeddings, layer_sizes,
              text_lengths, initial_state=initial_state, attn_length=0,
              dropout_keep_prob=dropout_keep_prob, is_training=is_training)

      embeddings = tf.concat([embeddings, text_embeddings, extra_embeddings],
          1, name='article_embeddings')
      softmax, logits = self.add_final_training_ops(
          embeddings,
          all_labels_count,
          hidden_layer_size=BOTTLENECK_TENSOR_SIZE / 8,
          dropout_keep_prob=dropout_keep_prob)

    # Prediction is the index of the label with the highest score. We are
    # interested only in the top score.
    prediction = tf.argmax(softmax, 1)
    tensors.predictions = [prediction, softmax]

    if graph_mod == GraphMod.PREDICT:
      return tensors

    with tf.name_scope('evaluate'):
      loss_value = loss(logits, labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    if is_training:
      tensors.train, tensors.global_step = training(loss_value)
    else:
      tensors.global_step = tf.Variable(0, name='global_step', trainable=False)

    # Add means across all batches.
    loss_updates, loss_op = util.loss(loss_value)
    accuracy_updates, accuracy_op = util.accuracy(logits, labels)

    precisions = []
    recalls = []
    for i in range(self.label_count):
        op, updates = tf.metrics.recall_at_k(labels, logits, 1, class_id=i)
        recalls.append({'op': op, 'updates': updates})
        op, updates = tf.metrics.sparse_precision_at_k(labels, logits, 1, class_id=i)
        precisions.append({'op': op, 'updates': updates})

    if not is_training:
      tf.summary.scalar('accuracy', accuracy_op)
      tf.summary.scalar('loss', loss_op)
      tf.summary.histogram('histogram_loss', loss_op)
      for i in range(self.label_count):
          label_name = self.labels[i]
          tf.summary.scalar('recall_%s' % label_name, recalls[i]['op'])
          tf.summary.scalar('precision_%s' % label_name, precisions[i]['op'])

    precision_updates = [x['updates'] for x in precisions]
    precision_ops = [x['op'] for x in precisions]
    recall_updates = [x['updates'] for x in recalls]
    recall_ops = [x['op'] for x in recalls]
    tensors.metric_updates = loss_updates + accuracy_updates + [recall_updates, precision_updates]
    tensors.metric_values = [loss_op, accuracy_op, recall_ops, precision_ops]
    return tensors

  def build_train_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMod.TRAIN)

  def build_eval_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMod.EVALUATE)

  def restore_from_checkpoint(self, session, trained_checkpoint_file):
    """To restore model variables from the checkpoint file.

       The graph is assumed to consist of an inception model and other
       layers including a softmax and a fully connected layer. The former is
       pre-trained and the latter is trained using the pre-processed data. So
       we restore this from two checkpoint files.
    Args:
      session: The session to be used for restoring from checkpoint.
      trained_checkpoint_file: path to the trained checkpoint for the other
                               layers.
    """
    if not trained_checkpoint_file:
      return

    # Restore the rest of the variables from the trained checkpoint.
    trained_saver = tf.train.Saver()
    trained_saver.restore(session, trained_checkpoint_file)

  def build_prediction_graph(self):
    """Builds prediction graph and registers appropriate endpoints."""

    tensors = self.build_graph(None, 1, GraphMod.PREDICT)

    keys_placeholder = tf.placeholder(tf.string, shape=[None])
    inputs = {
        'key': keys_placeholder,
        'image_embedding_bytes': tensors.input_image,
        'text_embedding': tensors.input_text,
        'text_length': tensors.input_text_length,
        'category_id': tensors.input_category_id,
        'price': tensors.input_price,
        'images_count': tensors.input_images_count,
        'created_at_ts': tensors.input_created_at_ts,
        'offerable': tensors.input_offerable,
        'recent_articles_count': tensors.input_recent_articles_count,
        'blocks_inline': tensors.input_blocks_inline,
    }

    # To extract the id, we need to add the identity function.
    keys = tf.identity(keys_placeholder)
    outputs = {
        'key': keys,
        'prediction': tensors.predictions[0],
        'scores': tensors.predictions[1]
    }

    return inputs, outputs

  def export(self, last_checkpoint, output_dir):
    """Builds a prediction graph and xports the model.

    Args:
      last_checkpoint: Path to the latest checkpoint file from training.
      output_dir: Path to the folder to be used to output the model.
    """
    logging.info('Exporting prediction graph to %s', output_dir)
    with tf.Session(graph=tf.Graph()) as sess:
      # Build and save prediction meta graph and trained variable values.
      inputs, outputs = self.build_prediction_graph()
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      self.restore_from_checkpoint(sess, last_checkpoint)
      signature_def = build_signature(inputs=inputs, outputs=outputs)
      signature_def_map = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
      }
      builder = saved_model_builder.SavedModelBuilder(output_dir)
      builder.add_meta_graph_and_variables(
          sess,
          tags=[tag_constants.SERVING],
          signature_def_map=signature_def_map)
      builder.save()

  def format_metric_values(self, metric_values):
    """Formats metric values - used for logging purpose."""

    # Early in training, metric_values may actually be None.
    loss_str = 'N/A'
    accuracy_str = 'N/A'
    try:
      loss_str = '%.3f' % metric_values[0]
      accuracy_str = '%.3f' % metric_values[1]
    except (TypeError, IndexError):
      pass

    return '%s, %s' % (loss_str, accuracy_str)


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss_op):
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(epsilon=0.001)

    # for batch norm http://ruishu.io/2016/12/27/batchnorm/
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss_op, global_step)
      return train_op, global_step
