# -*- coding: utf-8 -*-
import tensorflow as tf


class LMModel(object):

    def __init__(self, is_training, config, inputs):

        self.is_training = is_training
        self.inputs = inputs
        self.slice_size = self.inputs.slice_size
        self.batch_size = self.inputs.batch_size

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.keep_prob = config.keep_prob
        self.rnn_model = config.rnn_model
        self.max_grad_norm = config.max_grad_norm
        self.init_scale = config.init_scale

        self.cost = None
        self.lr = None
        self.new_lr = None
        self.initial_state = None
        self.final_state = None

        self.optim = None
        self.lr_updater = None

        eb_inputs = self._input_embedding()
        dp_inputs = self._input_dropout(eb_inputs)
        outputs = self._cudnn_layers(dp_inputs)
        logits = self._predict_layer(outputs)
        self._cost(logits)
        
        if self.is_training:
            self._optimize()

    def _input_embedding(self):
        embedding = tf.get_variable("embedding", [self.vocab_size, self.hidden_size], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, self.inputs.input_data)
        return inputs

    def _input_dropout(self, embedding_inputs):
        d_inputs = embedding_inputs
        if self.is_training and self.keep_prob < 1:
            d_inputs = tf.nn.dropout(embedding_inputs, self.keep_prob)
        return d_inputs

    def _cudnn_layers(self, drop_inputs):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(drop_inputs, [1, 0, 2])
        # build layer-wise network structures with different cells.
        if self.rnn_model == "gru":
            cell = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=self.num_layers, num_units=self.hidden_size, input_size=self.hidden_size, dropout=1 - self.keep_prob if self.is_training else 0)
        else:
            cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=self.num_layers, num_units=self.hidden_size, input_size=self.hidden_size, dropout=1 - self.keep_prob if self.is_training else 0)
        # parameter size of the structure built
        params_size_t = cell.params_size()
        rnn_params = tf.get_variable("rnn_params", initializer=tf.random_uniform( [params_size_t], -self.init_scale, self.init_scale), validate_shape=False)
        # memory c and/or history h cell operations in rnn
        c = tf.zeros([self.num_layers, self.batch_size, self.hidden_size], tf.float32)
        h = tf.zeros([self.num_layers, self.batch_size, self.hidden_size], tf.float32)
        if self.rnn_model == "gru":
            self.initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=h),)
            outputs, h = cell(inputs, h, rnn_params, self.is_training)
        else: # "lstm"
            self.initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
            outputs, h, c = cell(inputs, h, c, rnn_params, self.is_training)
        # outputs
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, self.hidden_size])
        if self.rnn_model == "gru":
            self.final_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=h),)
        else:
            self.final_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        return outputs

    def _predict_layer(self, pre_layer_outputs):
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(pre_layer_outputs, softmax_w, softmax_b)
        logits = tf.reshape(logits, [self.batch_size, self.slice_size, self.vocab_size])
        return logits

    def _cost(self, logits):
        loss = tf.contrib.seq2seq.sequence_loss(logits, self.inputs.targets, tf.ones([self.batch_size, self.slice_size], dtype=tf.float32), average_across_timesteps=False, average_across_batch=True)
        self.cost = tf.reduce_sum(loss)

    def _optimize(self):
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.optim = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step()
        )
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self.lr_updater = tf.assign(self.lr, self.new_lr)

    def update_lr(self, session, lr_value):
        session.run(self.lr_updater, feed_dict={self.new_lr: lr_value})
