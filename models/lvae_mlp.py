from base.base_model import BaseModel
import tensorflow as tf
from utils.myfuncs import *


class lvae_mlp(BaseModel):
    def __init__(self, config, input_batch=None):
        self.input_batch=input_batch
        super(lvae, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.e_mus, self.e_sigmas = [0] * self.config.layers_ladder, [0] * self.config.layers_ladder
        self.p_mus, self.p_sigmas = [0] * (self.config.layers_ladder - 1), [0] * (self.config.layers_ladder - 1)
        self.d_mus, self.d_sigmas = [0] * self.config.layers_ladder, [0] * self.config.layers_ladder
        self.zs = [0] * self.config.layers_ladder

        self.is_training = tf.placeholder(tf.bool)
        self.lambda_z_wu = tf.placeholder(tf.float32, shape=(), name="lambda_z_wu")

        if(self.input_batch is not None):
            self.x = self.input_batch[0]
            print("x:", self.x.get_shape().as_list())
            self.y = self.input_batch[1]
            print("y:", self.y.get_shape().as_list())

        else:
            self.x = tf.placeholder(tf.float32, shape=[None] + self.config.input_shape)
            self.y = tf.placeholder(tf.int64, shape=[None])

        eps = 1e-8  # epsilon for numerical stability

        ########################
        #### TENSORFLOW GRAPH ##
        ########################

        self.flow = self.x

        with tf.variable_scope('PREDICTION'):
            with tf.variable_scope('encoder'):
                print("\n## 3D CONVOLUTIONAL ENCODER ##")
                print("INPUT:", self.flow.get_shape().as_list())
                ####
                #### encoder
                ####
                with tf.variable_scope('encoder'):
                    for l in range(self.config.layers_encoder):
                        self.flow = tf_conv3d(self.flow, self.config.activation_maps_encoder[l],
                                              k_d=self.config.ks_encoder[l][0], k_h=self.config.ks_encoder[l][1],
                                              k_w=self.config.ks_encoder[l][2],
                                              d_d=self.config.s_encoder[l][0], d_h=self.config.s_encoder[l][1],
                                              d_w=self.config.s_encoder[l][2],
                                              name="conv3d_e" + str(l), bn=True,
                                              is_training=self.is_training, activation=tf.nn.relu)
                        print("ENCODER LAYER " + str(l) + ": ", self.flow.get_shape().as_list())

                    self.flow = tf.contrib.layers.flatten(self.flow)
                    print("OUTPUT:", self.flow.get_shape().as_list())
                    ####
                    #### end of encoder
                    ####

            ####
            #### hierarchical latent space
            ####
            print("\n## HIERARCHICAL LATENT SPACE ##")
            print("# ENCODER #")
            with tf.variable_scope('ladder_encoder'):

                for l in range(self.config.layers_ladder):
                    self.flow = tf_dense(self.flow, self.config.H_SIZES[l],
                                         name="h" + str(l), bn=True,
                                         is_training=self.is_training,
                                         activation=tf.nn.elu)
                    print("H_" + str(l) + ": ", self.flow.get_shape().as_list())
                    #
                    # self.flow = tf_dense(self.flow, self.config.H_SIZES[l],
                    #                      name="h_bis" + str(l), bn=True,
                    #                      is_training=self.is_training,
                    #                      activation=tf.nn.elu)
                    # print("H_bis" + str(l) + ": ", self.flow.get_shape().as_list())

                    if (l == self.config.layers_ladder - 1):
                        self.flow = tf_dense(self.flow, self.config.H_SIZES[l]/2,
                                             name="h_tris_" + str(l), bn=True,
                                             is_training=self.is_training,
                                             activation=tf.nn.tanh)
                        print("H_tris_" + str(l) + ": ", self.flow.get_shape().as_list())

                        _, self.e_mus[l], self.e_sigmas[l] = vae_sampler2(scope="l" + str(l), x=self.flow,
                                                                         size=self.config.Z_SIZES[l],
                                                                         activation=None,
                                                                         is_training=self.is_training)
                        print("MU AND SIGMA LAYER " + str(l) + ": ", self.e_mus[l].get_shape().as_list())

                    else:
                        if (self.config.dense_in_ladder_enc == 0):
                            self.dbms = tf_dense(self.flow, self.config.Z_SIZES[l],
                                                 name="h" + str(l), bn=True,
                                                 is_training=self.is_training,
                                                 activation=tf.nn.elu)
                            print("DENSE BEFORE MU AND SIGMA L_" + str(l) + ": ", self.flow.get_shape().as_list())
                            _, self.e_mus[l], self.e_sigmas[l] = vae_sampler2(scope="l" + str(l), x=self.dbms,
                                                                             size=self.config.Z_SIZES[l],
                                                                             activation=None,
                                                                             is_training=self.is_training)
                            print("MU AND SIGMA LAYER " + str(l) + ": ", self.e_mus[l].get_shape().as_list())
                        if (self.config.dense_in_ladder_enc == 1):
                            _, self.e_mus[l], self.e_sigmas[l] = vae_sampler2(scope="l" + str(l), x=self.flow,
                                                                             size=self.config.Z_SIZES[l],
                                                                             activation=None,
                                                                             is_training=self.is_training)
                            print("MU AND SIGMA LAYER " + str(l) + ": ", self.e_mus[l].get_shape().as_list())
                        if (self.config.dense_in_ladder_enc == 2):
                            _, self.e_mus[l], self.e_sigmas[l] = vae_sampler2(scope="l" + str(l), x=self.flow,
                                                                              size=self.config.Z_SIZES[l],
                                                                              size_pre=self.config.Z_SIZES_PRE[l],
                                                                              activation=None,
                                                                              is_training=self.is_training)
                            print("MU AND SIGMA LAYER " + str(l) + ": ", self.e_mus[l].get_shape().as_list())

                mu, sigma = self.e_mus[self.config.layers_ladder - 1], self.e_sigmas[self.config.layers_ladder - 1]

                self.d_mus[self.config.layers_ladder - 1], self.d_sigmas[self.config.layers_ladder - 1] = mu, sigma

                self.zs[self.config.layers_ladder - 1] = sampler(self.d_mus[self.config.layers_ladder - 1], tf.exp(
                    self.d_sigmas[self.config.layers_ladder - 1]))

                print("INNER Z", self.zs[self.config.layers_ladder - 1].get_shape().as_list(), "\n")

            with tf.variable_scope('predictor'):
                ####
                #### MLP branch
                ####
                ####
                if self.config.layer_predict == 0:
                    self.outMLP = tf_dense(self.zs[self.config.layers_ladder - 1],
                                           self.config.num_classes,
                                           bn=False,
                                           activation=None,
                                           is_training=self.is_training,
                                           name="dense_2_mu")
                    print("OUTPUT PREDICTION:", self.outMLP.get_shape().as_list(),"\n")

                if self.config.layer_predict == 1:
                    d1 = tf_dense(self.zs[self.config.layers_ladder - 1],
                                  self.config.layer_predict_dim,
                                  bn=True,
                                  activation=tf.nn.relu,
                                  is_training=self.is_training,
                                  name="dense_1_mu")
                    print("PREDICTION LAYER:", d1.get_shape().as_list())
                    self.outMLP = tf_dense(d1,
                                           self.config.num_classes,
                                           bn=False,
                                           activation=None,
                                           is_training=self.is_training,
                                           name="dense_2_mu")
                    print("OUTPUT PREDICTION:", self.outMLP.get_shape().as_list(), "\n")

                if self.config.layer_predict == 2:
                    d1 = tf_dense(self.zs[self.config.layers_ladder - 1],
                                  self.config.layer_predict_dim,
                                  bn=True,
                                  activation=tf.nn.relu,
                                  is_training=self.is_training,
                                  name="dense_1_mu")
                    print("PREDICTION LAYER:", d1.get_shape().as_list())
                    d2 = tf_dense(d1,
                                  self.config.layer_predict_dim,
                                  bn=True,
                                  activation=tf.nn.relu,
                                  is_training=self.is_training,
                                  name="dense_2_mu")
                    print("PREDICTION LAYER:", d2.get_shape().as_list())
                    self.outMLP = tf_dense(d2,
                                           self.config.num_classes,
                                           bn=False,
                                           activation=None,
                                           is_training=self.is_training,
                                           name="dense_3_mu")
                    print("OUTPUT PREDICTION:", self.outMLP.get_shape().as_list(), "\n")

                self.prob = tf.nn.softmax(self.outMLP)
                # gradientsl
                self.grad_node1 = tf.gradients(self.outMLP[0, 1], self.zs[self.config.layers_ladder - 1])[0]
                self.grad_node0 = tf.gradients(self.outMLP[0, 0], self.zs[self.config.layers_ladder - 1])[0]
                self.grad_node = tf.gradients(self.outMLP, self.zs[self.config.layers_ladder - 1])[0]
                ####
                #### end of MLP branch
                ####

        print("# DECODER #")
        with tf.variable_scope('RECONSTRUCTION'):
            with tf.variable_scope('ladder_decoder'):
                for l in range(self.config.layers_ladder - 2, -1, -1):
                    if (self.config.dense_in_ladder_dec == 1):
                        self.flow = tf_dense(self.zs[l + 1],
                                             self.config.H_SIZES[l],
                                             name="h" + str(l),
                                             bn=True,
                                             is_training=self.is_training,
                                             activation=tf.nn.relu)
                        print("DENSE_" + str(l), self.flow.get_shape().as_list())
                        self.flow = tf_dense(self.zs[l + 1],
                                             self.config.H_SIZES[l],
                                             name="h_bis" + str(l),
                                             bn=True,
                                             is_training=self.is_training,
                                             activation=tf.nn.relu)
                        print("DENSE_bis" + str(l), self.flow.get_shape().as_list())
                        _, self.p_mus[l], self.p_sigmas[l] = vae_sampler2(scope="g" + str(l),
                                                                         x=self.flow,
                                                                         size=self.config.Z_SIZES[l],
                                                                         activation=None,
                                                                         is_training=self.is_training)

                    if (self.config.dense_in_ladder_dec == 0):
                        _, self.p_mus[l], self.p_sigmas[l] = vae_sampler2(scope="g" + str(l),
                                                                         x=self.zs[l + 1],
                                                                         size=self.config.Z_SIZES[l],
                                                                         activation=None,
                                                                         is_training=self.is_training)

                    self.zs[l], self.d_mus[l], self.d_sigmas[l] = precision_weighted_sampler(scope="g2" + str(l),
                                                                                             musigma1=(self.e_mus[l],
                                                                                                       self.e_sigmas[
                                                                                                           l]),
                                                                                             musigma2=(self.p_mus[l],
                                                                                                       self.p_sigmas[
                                                                                                           l]),
                                                                                             is_training=self.is_training)
                    print("Z, MU AND SIGMA " + str(l), self.zs[l].get_shape().as_list())
                    ####
                    #### end of the hierarchical latent space
                    ####

            ####
            #### decoder
            ####
            print("\n## 3D CONVOLUTIONAL DECODER ##")
            self.flow = tf_dense(self.zs[0], self.config.dim_start_up_decoder[0] * self.config.dim_start_up_decoder[1] *
                                 self.config.dim_start_up_decoder[2],
                                 bn=True, activation=tf.nn.relu, is_training=self.is_training, name="dense_upg")
            print("DENSE:", self.flow.get_shape().as_list())
            # if(self.config.dropout>0):
            #     self.flow = tf.nn.dropout(self.flow, self.config.dropout)
            self.flow = tf.reshape(self.flow, [tf.shape(self.x)[0]] + self.config.dim_start_up_decoder + [1])
            print("INPUT:", self.flow.get_shape().as_list())

            conv_count = 0
            deconv_count = 0

            with tf.variable_scope('decoder'):
                for l in range(len(self.config.decoder_structure) - 1):
                    if self.config.decoder_structure[l] == "conv":
                        self.flow = tf_conv3d(self.flow, self.config.maps_up_conv[conv_count],
                                              k_d=self.config.ks_up_conv[conv_count][0],
                                              k_h=self.config.ks_up_conv[conv_count][1],
                                              k_w=self.config.ks_up_conv[conv_count][2],
                                              d_d=self.config.s_up_conv[conv_count][0],
                                              d_h=self.config.s_up_conv[conv_count][1],
                                              d_w=self.config.s_up_conv[conv_count][2],
                                              name="up_conv" + str(conv_count), bn=True, is_training=self.is_training,
                                              activation=tf.nn.relu)
                        print("LAYER " + str(l) + "(CONVOLUTION):", self.flow.get_shape().as_list())
                        conv_count += 1

                    if self.config.decoder_structure[l] == "deconv":
                        self.flow = tf_deconv3d(self.flow, self.config.dim_up_decoder[deconv_count] + [
                            self.config.maps_up_deconv[deconv_count]],
                                                k_d=self.config.ks_up_deconv[deconv_count][0],
                                                k_h=self.config.ks_up_deconv[deconv_count][1],
                                                k_w=self.config.ks_up_deconv[deconv_count][2],
                                                d_d=self.config.s_up_deconv[deconv_count][0],
                                                d_h=self.config.s_up_deconv[deconv_count][1],
                                                d_w=self.config.s_up_deconv[deconv_count][2],
                                                name="up_dec" + str(deconv_count), bn=True,
                                                is_training=self.is_training, activation=tf.nn.relu)
                        print("LAYER " + str(l) + "(DECONVOLUTION):", self.flow.get_shape().as_list())
                        deconv_count += 1

            self.x_ = tf_conv3d(self.flow, self.config.maps_up_conv[conv_count],
                                k_d=self.config.ks_up_conv[conv_count][0], k_h=self.config.ks_up_conv[conv_count][1],
                                k_w=self.config.ks_up_conv[conv_count][2],
                                d_d=self.config.s_up_conv[conv_count][0], d_h=self.config.s_up_conv[conv_count][1],
                                d_w=self.config.s_up_conv[conv_count][2],
                                name="up_conv" + str(conv_count), bn=False, is_training=self.is_training,
                                activation=tf.nn.sigmoid)
            print("OUTPUT:", self.x_.get_shape().as_list())
            ####
            #### end of decoder
            ####

            ####
            #### END TENSORFLOW GRAPH
            ####

        with tf.name_scope("loss"):
            #####
            #####  Reconstruction losses
            #####
            self.dice_loss1 = 1 - dice_coe_mean(self.x_[:, :, :, :, 0], self.x[:, :, :, :, 0], axis=(1, 2, 3),
                                                loss_type='sorensen')
            self.dice_loss2 = 1 - dice_coe_mean(self.x_[:, :, :, :, 1], self.x[:, :, :, :, 1], axis=(1, 2, 3),
                                                loss_type='sorensen')
            print("\nDSC 1:", self.dice_loss1.get_shape().as_list())
            print("DSC 2:", self.dice_loss2.get_shape().as_list())

            #####
            #####  KL divergence
            #####
            self.kl_terms = [0] * self.config.layers_ladder
            self.kl_terms_ba = [0] * self.config.layers_ladder
            self.test_ratio = [0] * self.config.layers_ladder


            for l in range(self.config.layers_ladder):
                d_mu, d_sigma = self.d_mus[l], self.d_sigmas[l]  # d distribution
                d_sigma2 = tf.square(d_sigma)
                if (l == self.config.layers_ladder - 1):
                    self.kl_terms[l] = self.lambda_z_wu * self.config.alphas[l] * (0.5 * tf.reduce_sum(tf.square(d_mu) + d_sigma2 - tf.log(eps + d_sigma2) - 1.0 ,1))
                    self.kl_terms_ba[l] = 0.5 * tf.reduce_sum(tf.square(d_mu) + d_sigma2 - tf.log(eps + d_sigma2) - 1.0 ,1)
                else:
                    p_mu, p_sigma = self.p_mus[l], self.p_sigmas[l]  # prior distribution +
                    p_sigma2 = tf.square(p_sigma)
                    self.kl_terms[l] = self.lambda_z_wu * self.config.alphas[l] * (0.5 * tf.reduce_sum(((tf.square(d_mu - p_mu) + d_sigma2) / p_sigma2) - 1.0 + 2 * tf.log((p_sigma / d_sigma) + eps), 1))
                    self.kl_terms_ba[l] = 0.5 * tf.reduce_sum(((tf.square(d_mu - p_mu) + d_sigma2) / p_sigma2) - 1.0 + 2 * tf.log((p_sigma / d_sigma) + eps), 1)
                    self.test_ratio = 0.5 * tf.reduce_sum(2 * tf.log((p_sigma / d_sigma) + eps), 1)
                    # self.kl_terms[l] =  tf.reduce_sum(p_sigma / d_sigma,1)
                    # self.kl_terms_ba[l] = tf.reduce_sum(p_sigma / d_sigma,1)


            self.kl_divergence = tf.add_n(self.kl_terms)
            print("KL_divergence:", self.kl_divergence.get_shape().as_list())
            #####
            #####  MLP loss
            #####
            one_hot_labels = tf.reshape(tf.one_hot(self.y, depth=self.config.num_classes),
                                        [-1, self.config.num_classes])
            # self.ceMLP = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=self.outMLP), 1)
            print("one_hot_labels:", one_hot_labels.get_shape().as_list())
            print("self.outMLP:", self.outMLP.get_shape().as_list())

            self.ceMLP = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=self.outMLP)
            print("CE MLP: ", self.ceMLP.get_shape().as_list())

            #####
            #####  TOTAL loss
            #####
            self.tot_loss = tf.reduce_mean(self.dice_loss1 + self.dice_loss2 + self.lambda_z_wu * self.config.beta * self.ceMLP +  self.kl_divergence)
            print("TOTAL LOSS:", self.tot_loss.get_shape().as_list())

            # losses average over the batch
            self.dice_loss1_ba = tf.reduce_mean(self.dice_loss1)
            self.dice_loss2_ba = tf.reduce_mean(self.dice_loss2)
            # for l in range(self.config.layers_ladder):
            #     self.kl_terms_ba[l] = tf.reduce_mean(self.kl_terms_ba[l])
            self.ceMLP_ba = tf.reduce_mean(self.ceMLP)

            # OPTIMIZERS
            # boundaries
            boundaries = self.config.boundaries
            values = self.config.boundaries_values
            # learning rate
            learning_rate = tf.train.piecewise_constant(self.global_step_tensor, boundaries, values)
            #definition
            optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = optimiser.minimize(self.tot_loss, global_step=self.global_step_tensor)


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=None)
        # it would be great to add A OPTION to save at fixed EPOCHS
