import tensorflow as tf

class BaseTrain:
    def __init__(self, sess, model, data_train, data_eval, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data_train = data_train
        self.data_eval = data_eval
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        lambda_z_wu = self.config.lambda_z_wu_range[0]
        # for self.cur_epoch in range(self.model.global_step_tensor.eval(self.sess), self.config.num_iter + 1, 1):
        for self.cur_epoch in range(self.config.start_iter, self.config.num_iter + 1, 1):
            print("Epoch: " + str(self.cur_epoch) + " Lambda:" + str(lambda_z_wu))
            self.train_epoch(lambda_z_wu)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            if(lambda_z_wu < self.config.lambda_z_wu_range[1]):
                if(lambda_z_wu <  self.config.lambda_z_wu_range[3]):
                    lambda_z_wu += self.config.lambda_z_wu_range[2]
                else:
                    lambda_z_wu += self.config.lambda_z_wu_range[2] * 5
            else:
                lambda_z_wu = self.config.lambda_z_wu_range[1]

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
