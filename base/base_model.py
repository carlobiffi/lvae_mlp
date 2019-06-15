import tensorflow as tf
import os
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import time

class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load_latest(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # load specified checkpoint from the experiment path defined in the config file
    def load_from_checkpoint(self, sess):
        checkpoint = self.config.load_checkpoint_path
        if checkpoint:
            print("Loading model checkpoint {} ...\n".format(checkpoint))
            self.saver.restore(sess, checkpoint)
            print("Model loaded")

    # load from specified checkpoint some variables
    def part_load_from_checkpoint(self, sess):
        var_list = tf.global_variables()
        # present model variables
        check_var_list = checkpoint_utils.list_variables(self.config.load_checkpoint_path)
        # chekpoint variables
        check_var_list = [x[0] for x in check_var_list]
        check_var_set = set(check_var_list)
        vars_in_checkpoint = [x for x in var_list if x.name[:x.name.index(":")] in check_var_set]
        #vars in teh present model that are also present in the checkpoint
        saverPart = tf.train.Saver(var_list=vars_in_checkpoint)
        saverPart.restore(sess, self.config.load_checkpoint_path)
        print("Restored variables from the parsed checkpoint")

    # initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    # an abstract function to initialize the saver used for saving and loading the checkpoint.
    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    # an abstract function to define the model.
    def build_model(self):
        raise NotImplementedError