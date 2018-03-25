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
from tensorflow.python.saved_model import tag_constants, main_op
from tensorflow.python.saved_model import utils as saved_model_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.layers import Embedding, Conv1D

import util
from util import override_if_not_in_args

from rnn import stack_bidirectional_dynamic_rnn, simple_rnn, multi_rnn

LOGITS_TENSOR_NAME = 'logits_tensor'
LABEL_COLUMN = 'label'
EMBEDDING_COLUMN = 'embedding'

BLOCKS_COUNT = 72
TOTAL_CATEGORIES_COUNT = 62

DAY_TIME = 60 * 60 * 24

BOTTLENECK_TENSOR_SIZE = 1536
CHAR_DIM = 10
WORD_DIM = 50
MAX_WORDS_LENGTH = 256
MAX_USERNAME_CHARS_COUNT = 12
TEXT_EMBEDDING_SIZE = WORD_DIM * MAX_WORDS_LENGTH


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
  parser.add_argument('--label_count', type=int)
  parser.add_argument('--dropout', type=float, default=0.5)
  parser.add_argument('--input_dict', type=str)
  parser.add_argument('--char_dict', type=str)
  parser.add_argument('--attention', type=str, default='no_use')
  parser.add_argument('--username_type', type=str, default='rnn')
  parser.add_argument('--activation', type=str, default='relu')
  parser.add_argument('--rnn_cell_wrapper', type=str, default='residual')
  parser.add_argument('--variational_dropout', type=str, default='use')
  parser.add_argument('--rnn_type', type=str, default='LSTM')
  parser.add_argument('--rnn_layers_count', type=int, default=2)
  parser.add_argument('--final_layers_count', type=int, default=1)
  args, task_args = parser.parse_known_args()
  override_if_not_in_args('--max_steps', '1000', task_args)
  override_if_not_in_args('--batch_size', '100', task_args)
  override_if_not_in_args('--eval_set_size', '370', task_args)
  override_if_not_in_args('--eval_interval_secs', '2', task_args)
  override_if_not_in_args('--log_interval_secs', '2', task_args)
  override_if_not_in_args('--min_train_eval_rate', '2', task_args)
  return Model(args.label_count, args.dropout, args.input_dict,
          use_attention=args.attention=='use', rnn_type=args.rnn_type,
          rnn_layers_count=args.rnn_layers_count,
          final_layers_count=args.final_layers_count,
          char_dict_path=args.char_dict, username_type=args.username_type,
          rnn_cell_wrapper=args.rnn_cell_wrapper,
          variational_dropout=args.variational_dropout,
          activation=args.activation), task_args


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
    self.input_title_length = None
    self.input_content_length = None
    self.input_blocks_inline = None
    self.input_username_chars = None
    self.input_username_length = None
    self.input_text_length = None
    self.ids = None
    self.labels = None

def find_nearest_idx(array, value):
    return tf.argmin(tf.abs(
        tf.expand_dims(array, 0) - tf.expand_dims(value, 1)
    ), 1)

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
        return tf.matmul(tf.transpose(one_hot_indices), values)


class Model(object):
  """TensorFlow model for the flowers problem."""

  def __init__(self, label_count, dropout, labels_path, use_attention=False,
          rnn_type='LSTM', rnn_layers_count=2, final_layers_count=2, char_dict_path=None,
          rnn_cell_wrapper=None, variational_dropout=None,
          username_type=None, activation=None):
    self.label_count = label_count
    self.dropout = dropout
    self.labels = file_io.read_file_to_string(labels_path).strip().split('\n')
    self.use_attention = use_attention
    self.rnn_type = rnn_type
    self.rnn_layers_count = rnn_layers_count
    self.final_layers_count = final_layers_count
    self.username_type = username_type
    self.activation = activation
    self.rnn_cell_wrapper = rnn_cell_wrapper
    self.variational_recurrent = variational_dropout == 'use'

    username_chars = file_io.read_file_to_string(char_dict_path).decode('utf-8').strip().split('\n')
    self.username_chars = username_chars

  def get_labels(self):
      return self.labels

  def id_to_key(self, id):
      return self.labels[id]

  def add_final_training_ops(self, hidden, all_labels_count):
    """Adds a new softmax and fully-connected layer for training.

     The set up for the softmax and fully-connected layers is based on:
     https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

     This function can be customized to add arbitrary layers for
     application-specific requirements.
    Args:
      embeddings: The embedding (bottleneck) tensor.
      all_labels_count: The number of all labels including the default label.
    Returns:
      softmax: The softmax or tensor. It stores the final scores.
      logits: The logits tensor.
    """
    logits = layers.fully_connected(hidden, all_labels_count, activation_fn=None)
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
      image_embeddings = inception_embeddings

      text_embeddings = tf.placeholder(tf.float32, shape=[None, TEXT_EMBEDDING_SIZE])
      text_lengths = tf.placeholder(tf.int64, shape=[None])
      category_ids = tf.placeholder(tf.int64, shape=[None])
      price = tf.placeholder(tf.int64, shape=[None])
      images_count = tf.placeholder(tf.int64, shape=[None])
      recent_articles_count = tf.placeholder(tf.int64, shape=[None])
      title_length = tf.placeholder(tf.int64, shape=[None])
      content_length = tf.placeholder(tf.int64, shape=[None])
      blocks_inline = tf.placeholder(tf.string, shape=[None])
      username_chars = tf.placeholder(tf.string, shape=[None, MAX_USERNAME_CHARS_COUNT])
      username_length = tf.placeholder(tf.int64, shape=[None])
      created_at_ts = tf.placeholder(tf.int64, shape=[None])
      offerable = tf.placeholder(tf.int64, shape=[None])

      tensors.input_image = inception_input
      tensors.input_text = text_embeddings
      tensors.input_text_length = text_lengths
      tensors.input_category_id = category_ids
      tensors.input_price = price
      tensors.input_images_count = images_count
      tensors.input_recent_articles_count = recent_articles_count
      tensors.input_title_length = title_length
      tensors.input_content_length = content_length
      tensors.input_blocks_inline = blocks_inline
      tensors.input_username_chars = username_chars
      tensors.input_username_length = username_length
      tensors.input_created_at_ts = created_at_ts
      tensors.input_offerable = offerable

      username_chars = tf.reshape(username_chars, [-1, MAX_USERNAME_CHARS_COUNT])
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
                tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'category_id':
                tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'price':
                tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'images_count':
                tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'recent_articles_count':
                tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'title_length':
                tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'content_length':
                tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'blocks_inline':
                tf.FixedLenFeature(shape=[], dtype=tf.string),
            'username_chars':
                tf.FixedLenFeature(shape=[MAX_USERNAME_CHARS_COUNT], dtype=tf.string),
            'username_length':
                tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'created_at_ts':
                tf.FixedLenFeature(shape=[], dtype=tf.int64),
            'offerable':
                tf.FixedLenFeature(shape=[], dtype=tf.int64),
        }
        parsed = tf.parse_example(tensors.examples, features=feature_map)
        labels = tf.squeeze(parsed['label'])
        tensors.labels = labels
        tensors.ids = tf.squeeze(parsed['id'])
        image_embeddings = parsed['embedding']
        text_embeddings = parsed['text_embedding']
        text_lengths = parsed['text_length']
        category_ids = parsed['category_id']
        price = parsed['price']
        images_count = parsed['images_count']
        recent_articles_count = parsed['recent_articles_count']
        title_length = parsed['title_length']
        content_length = parsed['content_length']
        blocks_inline = parsed['blocks_inline']
        username_chars = parsed['username_chars']
        username_length = parsed['username_length']
        created_at_ts = parsed['created_at_ts']
        offerable = parsed['offerable']

    dropout_keep_prob = self.dropout if is_training else None

    def dropout(x, keep_prob):
        if keep_prob:
            return tf.nn.dropout(x, keep_prob)
        return x

    def dense(x, units):
        for unit in units:
            if self.activation == 'maxout':
                x = layers.fully_connected(x, unit, activation_fn=None)
                x = tf.contrib.layers.maxout(x, unit)
                x = tf.reshape(x, [-1, unit])
            elif self.activation == 'none':
                x = layers.fully_connected(x, unit,
                        normalizer_fn=tf.contrib.layers.batch_norm,
                        normalizer_params={'is_training': is_training})
            else:
                x = layers.fully_connected(x, unit)
            x = dropout(x, dropout_keep_prob)
        return x

    with tf.variable_scope("username"):
        table = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(self.username_chars),
                default_value=len(self.username_chars))
        char_ids = table.lookup(username_chars)
        char_dict_size = len(self.username_chars) + 1 # add unknown char
        x = Embedding(char_dict_size, CHAR_DIM)(char_ids)
        mask = tf.sequence_mask(username_length, MAX_USERNAME_CHARS_COUNT, dtype=tf.float32)
        x = x * tf.expand_dims(mask, 2)

        if self.username_type == 'dense':
            username = tf.reshape(x, [-1, MAX_USERNAME_CHARS_COUNT * CHAR_DIM])
            username = dense(username, [30, 30])
        elif self.username_type == 'cnn':
            filters = 5
            k3 = tf.layers.conv1d(x, filters, 3)
            k3 = tf.layers.max_pooling1d(k3, 3, 3)
            k2 = tf.layers.conv1d(x, filters, 2)
            k2 = tf.layers.max_pooling1d(k2, 2, 2)
            k1 = tf.layers.conv1d(x, filters, 1)
            k1 = tf.layers.max_pooling1d(k1, 2, 2)
            x = tf.concat([k1, k2, k3], 1)
            x = tf.reshape(x, [-1, filters * 14])
            username = layers.fully_connected(x, 30,
                    normalizer_fn=tf.contrib.layers.batch_norm,
                    normalizer_params={'is_training': is_training})
        elif self.username_type == 'rnn':
            outputs, last_states = stack_bidirectional_dynamic_rnn(x, [CHAR_DIM],
                    username_length, dropout_keep_prob=dropout_keep_prob,
                    cell_wrapper=self.rnn_cell_wrapper,
                    variational_recurrent=self.variational_recurrent,
                    is_training=is_training)
            username = last_states
        elif self.username_type == 'none':
            username = None
        else:
            raise Exception('Invaild username_type: %s' % self.username_type)

    with tf.variable_scope("user"):
        recent_articles_count = tf.minimum(recent_articles_count, 300)
        recent_articles_count = tf.expand_dims(recent_articles_count, 1)
        recent_articles_count = tf.to_int32(recent_articles_count)
        blocks = blocks_inline_to_matrix(blocks_inline)
        blocks = tf.minimum(blocks, 50)

        user = tf.concat([recent_articles_count#, blocks
            ], 1)
        user = tf.cast(user, tf.float32)
        user = tf.layers.batch_normalization(user, training=is_training)
        if self.username_type != 'none':
            user = tf.concat([user, username], 1)
            user = dense(user, [30])
        else:
            user = dropout(user, dropout_keep_prob)

    with tf.variable_scope("category"):
        category_ids = tf.minimum(category_ids - 1, TOTAL_CATEGORIES_COUNT - 1)
        category = Embedding(TOTAL_CATEGORIES_COUNT, 5)(category_ids)
        category = dropout(category, dropout_keep_prob)

    with tf.variable_scope("continuous"):
        price = tf.minimum(price, 1000000000)
        title_length = tf.minimum(title_length, 100)
        content_length = tf.minimum(content_length, 3000)
        created_time = tf.mod(created_at_ts, DAY_TIME)
        day = tf.mod(created_at_ts / DAY_TIME, 7)

        continuous = tf.stack([price, images_count, title_length,
            content_length#, offerable, created_time, day
            ], 1)
        continuous = tf.cast(continuous, tf.float32)
        continuous = tf.concat([continuous, tf.square(continuous)], 1)
        continuous = tf.layers.batch_normalization(continuous, training=is_training)
        continuous = dropout(continuous, dropout_keep_prob)

    with tf.variable_scope("image"):
        image_embeddings = dense(image_embeddings, [384, 192])

    with tf.variable_scope('bunch'):
      bunch = tf.concat([image_embeddings, category, continuous, user], 1)
      bunch = dense(bunch, [192])

    with tf.variable_scope('text'):
      initial_state = dense(bunch, [WORD_DIM * 2, WORD_DIM])

      layer_sizes = [WORD_DIM * (2**i) for i in range(self.rnn_layers_count)]
      text_embeddings = tf.reshape(text_embeddings, [-1, MAX_WORDS_LENGTH, WORD_DIM])
      base_cell = tf.contrib.rnn.BasicLSTMCell if self.rnn_type == 'LSTM' else tf.contrib.rnn.GRUCell
      text_outputs, text_last_states = stack_bidirectional_dynamic_rnn(text_embeddings, layer_sizes,
              text_lengths, initial_state=(None if self.use_attention else initial_state),
              cell_wrapper=self.rnn_cell_wrapper, variational_recurrent=self.variational_recurrent,
              base_cell=base_cell, dropout_keep_prob=dropout_keep_prob, is_training=is_training)

      if self.use_attention:
          with tf.name_scope('attention'):
              attention_mechanism = tf.contrib.seq2seq.LuongAttention(WORD_DIM, text_outputs, text_lengths)
              alignments = attention_mechanism(initial_state, None)
              expanded_alignments = array_ops.expand_dims(alignments, 1)
              context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
              context = array_ops.squeeze(context, [1])
          hidden = tf.concat([bunch, context], 1)
      else:
          hidden = text_last_states
          #hidden = tf.concat([bunch, text_last_states], 1)

    with tf.variable_scope('final_ops'):
      hidden = dense(hidden, [WORD_DIM*2] * self.final_layers_count + [WORD_DIM])
      softmax, logits = self.add_final_training_ops(hidden, self.label_count)

    # Prediction is the index of the label with the highest score. We are
    # interested only in the top score.
    prediction = tf.argmax(softmax, 1)
    tensors.predictions = [prediction, softmax]

    if graph_mod == GraphMod.PREDICT:
      return tensors

    def is_l2_var_name(name):
        for token in ['bias', 'table', 'BatchNorm']:
            if token in name:
                return False
        return True

    with tf.name_scope('evaluate'):
      loss_value = loss(logits, labels)
      #l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() if is_l2_var_name(v.name) ])
      #loss_value += l2_loss * 0.001

    # Add to the Graph the Ops that calculate and apply gradients.
    if is_training:
      tensors.train, tensors.global_step = training(loss_value)
    else:
      tensors.global_step = tf.Variable(0, name='global_step', trainable=False)

    # Add means across all batches.
    loss_updates, loss_op = util.loss(loss_value)
    accuracy_updates, accuracy_op = util.accuracy(logits, labels)

    precision = {'ops': [], 'updates': []}
    recall = {'ops': [], 'updates': []}

    with tf.name_scope('metrics'):
        for i in range(self.label_count):
            op, update = tf.metrics.recall_at_k(labels, logits, 1, class_id=i)
            recall['ops'].append(op)
            recall['updates'].append(update)
            op, update = tf.metrics.sparse_precision_at_k(labels, logits, 1, class_id=i)
            precision['ops'].append(op)
            precision['updates'].append(update)

    if not is_training:
      tf.summary.scalar('accuracy', accuracy_op)
      tf.summary.scalar('loss', loss_op)
      for i in range(self.label_count):
          label_name = self.labels[i]
          tf.summary.scalar('recall_%s' % label_name, recall['ops'][i])
          tf.summary.scalar('precision_%s' % label_name, precision['ops'][i])

    tensors.metric_updates = loss_updates + accuracy_updates + \
            recall['updates'] + precision['updates']
    tensors.metric_values = [loss_op, accuracy_op, recall['ops'], precision['ops']]
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
        'title_length': tensors.input_title_length,
        'content_length': tensors.input_content_length,
        'blocks_inline': tensors.input_blocks_inline,
        'username_chars': tensors.input_username_chars,
        'username_length': tensors.input_username_length,
    }

    # To extract the id, we need to add the identity function.
    keys = tf.identity(keys_placeholder)
    outputs = {
        'key': keys,
        'prediction': tensors.predictions[0],
        'scores': tensors.predictions[1],
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
          signature_def_map=signature_def_map,
          main_op=tf.tables_initializer())
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
