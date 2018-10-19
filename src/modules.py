# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    ''' Does question have mask here? '''
    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class BidirectionalAttn(object):
    def __init__(self, keep_prob, hidden_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.hidden_vec_size = hidden_vec_size

    def build_graph(self, question_embeding_vecs, question_masks, context_embedding_vecs, context_masks):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          question_embeding_vecs: Tensor shape (batch_size, num_questions, value_vec_size).
          question_masks: Tensor shape (batch_size, num_questions).
            1s where there's real input, 0s where there's padding
          context_embedding_vecs: Tensor shape (batch_size, num_contexts, value_vec_size)
          context_masks: Tensor shape (batch_size, num_contexts).

        Intermediate variables:
          similarity_matrix: Tensor shape (batch_size, num_contexts, num_questions)
          element_dot_product_matrix: The matrix of CQ = c_i \dot q_j. Tensor shape (batch_size, num_contexts, num_questions, value_vec_size)
          {w_sim}^T (c_i; q_j; c_i \dot q_j)
          So we need to expand C and Q to the same size of CQ.

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BidirectionalAttn"):
            # Calculate attention distribution
            tiled_c = tf.tile(tf.expand_dims(context_embedding_vecs, 2), [1, 1, tf.shape(question_embeding_vecs)[1], 1])
            tiled_q = tf.tile(tf.expand_dims(question_embeding_vecs, 1), [1, tf.shape(context_embedding_vecs)[1], 1, 1])
            concated_matrix = tf.concat([tiled_c, tiled_q, tiled_c * tiled_q], axis=3)

            logits = tf.contrib.layers.fully_connected(concated_matrix, num_outputs=1, activation_fn=None)
            # weight_sim = tf.get_variable("weight_sim", shape=[3 * self.hidden_vec_size, 1])
            # logits = tf.matmul(tf.reshape(concated_matrix, [-1, 3 * self.hidden_vec_size]), weight_sim)
            # similarity_matrix = tf.reshape(logits, tf.shape(concated_matrix)[:-1])
            similarity_matrix = tf.squeeze(logits, axis=[3])  # Tensor shape (batch_size, num_contexts, num_questions)

            # Calculate c2q attention
            attn_logits_mask = tf.expand_dims(question_masks, 1)  # shape (batch_size, 1, num_questions)
            # shape (batch_size, num_contexts, num_questions). take softmax over values
            _, c2q_attn_dist = masked_softmax(similarity_matrix, attn_logits_mask, 2)
            # Tensor shape (batch_size, num_contexts, value_vec_size)
            c2q_attn_output = tf.matmul(c2q_attn_dist, question_embeding_vecs)

            # Calculate q2c attention
            sim_matrix_max = tf.reduce_max(similarity_matrix, axis=2)  # Tensor shape (batch_size, num_contexts)
            # Tensor shape (batch_size, num_contexts)
            _, q2c_attn_dist = masked_softmax(sim_matrix_max, context_masks, 1)
            # Tensor shape (batch_size, 1, value_vec_size)
            q2c_attn_output = tf.matmul(tf.expand_dims(q2c_attn_dist, 1), context_embedding_vecs)

            output = tf.concat([context_embedding_vecs, c2q_attn_output, context_embedding_vecs * c2q_attn_output,
                                context_embedding_vecs * q2c_attn_output], axis=2)
            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return output


class CoAttention(object):
    def __init__(self, keep_prob, hidden_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.hidden_vec_size = hidden_vec_size

    def build_graph(self, question_embeding_vecs, question_masks, context_embedding_vecs, context_masks):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          question_embeding_vecs: Tensor shape (batch_size, num_questions, value_vec_size).
          question_masks: Tensor shape (batch_size, num_questions).
            1s where there's real input, 0s where there's padding
          context_embedding_vecs: Tensor shape (batch_size, num_contexts, value_vec_size)
          context_masks: Tensor shape (batch_size, num_contexts).

        Intermediate variables:
          similarity_matrix: Tensor shape (batch_size, num_contexts, num_questions)
          element_dot_product_matrix: The matrix of CQ = c_i \dot q_j. Tensor shape (batch_size, num_contexts, num_questions, value_vec_size)
          {w_sim}^T (c_i; q_j; c_i \dot q_j)
          So we need to expand C and Q to the same size of CQ.

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("CoAttention"):
            transformed_question_vecs = tf.contrib.layers.\
                fully_connected(question_embeding_vecs, num_outputs=self.hidden_vec_size, activation_fn=tf.tanh)
            sentinel_c = tf.get_variable("sentinel_c", shape=[1, self.hidden_vec_size])
            sentinel_q = tf.get_variable("sentinel_q", shape=[1, self.hidden_vec_size])
            # tensor shape: [ batch_size, N+1, l ]
            context_entended_vec = tf.concat([context_embedding_vecs, tf.tile(tf.expand_dims(sentinel_c, 0),
                                              [tf.shape(question_embeding_vecs)[0], 1, 1])], axis=1)
            # tensor shape: [ batch_size, M+1, l ]
            question_entended_vec = tf.concat([transformed_question_vecs, tf.tile(tf.expand_dims(sentinel_q, 0),
                                              [tf.shape(question_embeding_vecs)[0], 1, 1])], axis=1)
            # tensor shape: [ batch_size, N+1, M+1]
            affinity_mat = tf.matmul(context_entended_vec, tf.transpose(question_entended_vec, perm=[0, 2, 1]))

            # Create a (batch_size, 1) vector
            #mask_extend_placeholder = tf.placeholder(tf.int32, shape=(tf.shape(question_embeding_vecs)[0], 1))
            t = tf.fill([tf.shape(question_embeding_vecs)[0], 1], 1)

            # shape (batch_size, 1, M+1)
            attn_question_masks = tf.expand_dims(
                tf.concat([question_masks, t], axis=1), 1)
            # shape (batch_size, N+1, M+1). take softmax over column axis
            _, alpha_dist = masked_softmax(affinity_mat, attn_question_masks, 2)
            # tensor shape: (batch_size, N+1, l)
            c2q_attn_outputs = tf.matmul(alpha_dist, question_entended_vec)

            # shape (batch_size, N+1, 1)
            attn_context_masks = tf.expand_dims(
                tf.concat([context_masks, t], axis=1), 2)
            # shape (batch_size, N+1, M+1). take softmax over row axis
            _, beta_dist = masked_softmax(affinity_mat, attn_context_masks, 1)
            # tensor shape: (batch_size, M+1, l)
            q2c_attn_outputs = tf.matmul(tf.transpose(beta_dist, perm=[0, 2, 1]), context_entended_vec)
            attn_2nd_lv = tf.matmul(alpha_dist, q2c_attn_outputs)  # tensor shape: (batch_size, N+1, l)

            concated_vec = tf.concat([attn_2nd_lv, c2q_attn_outputs], axis=2)
            trancated_concated_vec = concated_vec[:, :-1, :]
            encoder = RNNEncoder(2 * self.hidden_vec_size, self.keep_prob)
            # (batch_size, context_len, hidden_size*4)
            bi_gru_output = encoder.build_graph(trancated_concated_vec, context_masks)
            return bi_gru_output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
