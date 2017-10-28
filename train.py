# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import input as ip
import lm
import config as cf
import numpy as np


def run_epoch(session, model, eval_op=None, verbose=False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {"cost": model.cost, "final_state": model.final_state}
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.inputs.num_slice):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.inputs.slice_size

        if verbose and step % (model.inputs.num_slice // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.inputs.num_slice, np.exp(costs / iters),
                   iters * model.inputs.batch_size /
                   (time.time() - start_time)))
    return np.exp(costs / iters)


def main(_):

    config = cf.Config()
    eval_config = cf.Config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    train_data, valid_data, test_data, _ = ip.get_raw_data(config.data_path)
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input = ip.Input(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = lm.LMModel(is_training=True, config=config, inputs=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Training Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = ip.Input(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = lm.LMModel(is_training=False, config=config, inputs=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = ip.Input(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = lm.LMModel(is_training=False, config=eval_config, inputs=test_input)

        sv = tf.train.Supervisor(logdir=config.save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=False)

        with sv.managed_session(config=config_proto) as session:
            for i in range(config.max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.lr_const_epoch, 0)
                m.update_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.optim, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)

            print("Saving model to %s." % config.save_path)
            sv.saver.save(session, config.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
