import tensorflow as tf

from tensor2tensor.models import transformer
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_prepare_encoder
from tensor2tensor.models.transformer import transformer_encoder
from tensor2tensor.models.transformer import features_to_nonpadding

def transfomer_encoder(embeddings, max_length):
    if max_length > 50:
        hparams = transformer.transformer_relative_tiny()
    else:
        hparams = transformer.transformer_tiny()
    hparams.hidden_size = 50
    hparams.filter_size = 256
    hparams.num_heads = 5
    hparams.max_length = max_length

    inputs = embeddings
    target_space = tf.constant(1, dtype=tf.int32)
    features = {'inputs': inputs}

    (encoder_input, encoder_self_attention_bias, _) = (
        transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    encoder_output = transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams,
        nonpadding=features_to_nonpadding(features, "inputs"))

    return encoder_output

    #Modes = tf.estimator.ModeKeys
    #transformer_encoder_model = transformer.TransformerEncoder(hparams, Modes.TRAIN)

    #embeddings = tf.expand_dims(embeddings, 2)
    #output = transformer_encoder_model.body(
    #  {'inputs': embeddings, 'target_space_id': tf.constant(1, dtype=tf.int32)})
    #output = tf.squeeze(output, 2)
    #return output
