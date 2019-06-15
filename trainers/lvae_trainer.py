from base.base_train import BaseTrain
import tensorflow as tf
import numpy as np
import os
from utils.dirs import create_dirs
from utils.myfuncs import *

import pandas as pd
import time

class train_lvae(BaseTrain):
    def __init__(self, sess, model, data_train, data_eval, config,logger):
        super(train_lvae, self).__init__(sess, model, data_train, data_eval, config,logger)

    def train_epoch(self, lambda_z_wu):
        if(self.model.global_step_tensor.eval(self.sess)>self.config.loop_dwu*self.config.steps_dwu):
            loop = range(self.config.num_iter_per_epoch)
        else:
            loop = range(self.config.loop_dwu)

        loop_eval = range(self.config.num_iter_per_eval)

        #### TRAINING
        tot_loss_list = []
        dice_loss1_list = []
        dice_loss2_list = []
        for i in range(self.config.layers_ladder):
            exec("kls_list_%s = []" % i)
        ceMLP_list = []

        # start_time = time.time()
        for _ in loop:
            # print('size queue =', self.data_train.queue_size.eval(session=self.sess))
            tot_loss, dice_loss1, dice_loss2, kls, ceMLP = self.train_step(lambda_z_wu)
            tot_loss_list.append(tot_loss)
            dice_loss1_list.append(dice_loss1)
            dice_loss2_list.append(dice_loss2)
            for i in range(self.config.layers_ladder):
                exec("kls_list_%s.append(kls[%s])" % (i,i))
            ceMLP_list.append(ceMLP)
        # end_time = time.time()
        # print("Total time taken this loop: ", end_time - start_time)

        #mean losses of the training
        tot_loss_tr = np.mean(tot_loss_list)
        dice_loss1_tr = np.mean(dice_loss1_list)
        dice_loss2_tr = np.mean(dice_loss2_list)
        kls_list_tr = [0] * self.config.layers_ladder
        for i in range(self.config.layers_ladder):
            exec ("kls_list_tr[%s] = np.mean(kls_list_%s)" % (i,i))
        ceMLP_tr = np.mean(ceMLP_list)

        #PRINT THE VALUES OF THE LOSS FUNCTION TERMS
        batch_x = self.data_train.input_x[0:self.config.batch_size, :, :, :, :]
        batch_y = self.data_train.input_y[0:self.config.batch_size]
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.lambda_z_wu: lambda_z_wu, self.model.is_training: True}
        dice_loss1, dice_loss2, kl_divergence, ceMLP, outMLP, prob, kl_terms = self.sess.run([self.model.dice_loss1,
                                                                                      self.model.dice_loss2,
                                                                                      self.model.kl_divergence,
                                                                                      self.model.ceMLP,
                                                                                      self.model.outMLP,
                                                                                      self.model.prob,
                                                                                      self.model.kl_terms],
                                                                                      feed_dict=feed_dict)
        print("dice_loss1: ", dice_loss1, "\n dice_loss2: ", dice_loss2,"\n kl_divergence: ", kl_divergence,
              "\n ceMLP: ", ceMLP, "\n outMLP: ", outMLP, "\n y: ", self.data_train.input_y[0:10], "\n prob:", prob
              , "\n Kl terms:", kl_terms)
        ################################################

        #### EVALUATION

        tot_loss_list = []
        dice_loss1_list = []
        dice_loss2_list = []
        for i in range(self.config.layers_ladder):
            exec("kls_list_%s = []" % i)
        ceMLP_list = []

        for _ in loop_eval:
            tot_loss, dice_loss1, dice_loss2, kls, ceMLP = self.eval_step(lambda_z_wu)
            tot_loss_list.append(tot_loss)
            dice_loss1_list.append(dice_loss1)
            dice_loss2_list.append(dice_loss2)
            for i in range(self.config.layers_ladder):
                exec("kls_list_%s.append(kls[%s])" % (i,i))
            ceMLP_list.append(ceMLP)

        #mean losses of the evaluation
        tot_loss_ev = np.mean(tot_loss_list)
        dice_loss1_ev = np.mean(dice_loss1_list)
        dice_loss2_ev = np.mean(dice_loss2_list)
        kls_list_ev = [0] * self.config.layers_ladder
        for i in range(self.config.layers_ladder):
            exec ("kls_list_ev[%s] = np.mean(kls_list_%s)" % (i,i))
        ceMLP_ev = np.mean(ceMLP_list)

        ################################################

        #### LOGGER

        cur_it = self.model.global_step_tensor.eval(self.sess)
        print('Global Step = {}; '
              'Training loss = {:.5f}; '
              'Evaluation loss = {:.5f};'.format(cur_it, tot_loss_tr, tot_loss_ev))

        xeval, x_eval = self.eval_recon()
        xtrain, x_train = self.train_recon()

        expected_output_size = self.config.expected_output_size

        summaries_dict_train = {
            'tot_loss': tot_loss_tr,
            'dice_loss1': dice_loss1_tr,
            'dice_loss2': dice_loss2_tr,
            'ceMLP': ceMLP_tr,
            '19_x_train_input': xtrain[0, int(self.config.input_shape[0]/2), :, :, 0].reshape(expected_output_size),
            '19_x_train_rec': x_train[0, int(self.config.input_shape[0]/2), :, :, 0].reshape(expected_output_size),
            '20_x_train_input': xtrain[0, int(self.config.input_shape[0] / 2), :, :, 1].reshape(expected_output_size),
            '20_x_train_rec': x_train[0, int(self.config.input_shape[0] / 2), :, :, 1].reshape(expected_output_size),
            '19_y_train_input': xtrain[0, :, int(self.config.input_shape[1]/2), :, 0].reshape(expected_output_size),
            '19_y_train_rec': x_train[0, :, int(self.config.input_shape[1]/2), :, 0].reshape(expected_output_size),
            '20_y_train_input': xtrain[0, :, int(self.config.input_shape[1]/2), :, 1].reshape(expected_output_size),
            '20_y_train_rec': x_train[0, :, int(self.config.input_shape[1]/2), :, 1].reshape(expected_output_size),
            '19_z_train_input': xtrain[0, :, :, int(self.config.input_shape[2]/2), 0].reshape(expected_output_size),
            '19_z_train_rec': x_train[0, :, :, int(self.config.input_shape[2]/2), 0].reshape(expected_output_size),
            '20_z_train_input': xtrain[0, :, :, int(self.config.input_shape[2]/2), 1].reshape(expected_output_size),
            '20_z_train_rec': x_train[0, :, :, int(self.config.input_shape[2]/2), 1].reshape(expected_output_size),
        }

        for i in range(self.config.layers_ladder):
            exec("temp={'kl_%s': kls_list_tr[%s]}" % (i,i))
            exec("summaries_dict_train.update(temp)")

        summaries_dict_eval = {
            'tot_loss': tot_loss_ev,
            'dice_loss1': dice_loss1_ev,
            'dice_loss2': dice_loss2_ev,
            'ceMLP': ceMLP_ev,
            '19_x_eval_input': xeval[0, int(self.config.input_shape[0]/2), :, :, 0].reshape(expected_output_size),
            '19_x_eval_rec': x_eval[0, int(self.config.input_shape[0]/2), :, :, 0].reshape(expected_output_size),
            '20_x_eval_input': xeval[0, int(self.config.input_shape[0] / 2), :, :, 1].reshape(expected_output_size),
            '20_x_eval_rec': x_eval[0, int(self.config.input_shape[0] / 2), :, :, 1].reshape(expected_output_size),
            '19_y_eval_input': xeval[0, :, int(self.config.input_shape[1]/2), :, 0].reshape(expected_output_size),
            '19_y_eval_rec': x_eval[0, :, int(self.config.input_shape[1]/2), :, 0].reshape(expected_output_size),
            '20_y_eval_input': xeval[0, :, int(self.config.input_shape[1]/2), :, 1].reshape(expected_output_size),
            '20_y_eval_rec': x_eval[0, :, int(self.config.input_shape[1]/2), :, 1].reshape(expected_output_size),
            '19_z_eval_input': xeval[0, :, :, int(self.config.input_shape[2]/2), 0].reshape(expected_output_size),
            '19_z_eval_rec': x_eval[0, :, :, int(self.config.input_shape[2]/2), 0].reshape(expected_output_size),
            '20_z_eval_input': xeval[0, :, :, int(self.config.input_shape[2]/2), 1].reshape(expected_output_size),
            '20_z_eval_rec': x_eval[0, :, :, int(self.config.input_shape[2]/2), 1].reshape(expected_output_size),
        }

        for i in range(self.config.layers_ladder):
            exec("temp={'kl_%s': kls_list_ev[%s]}" % (i,i))
            exec("summaries_dict_eval.update(temp)")

        self.logger.summarize(cur_it, summaries_dict=summaries_dict_train, summarizer="train")
        self.logger.summarize(cur_it, summaries_dict=summaries_dict_eval, summarizer="eval")

        if(self.cur_epoch>10 and (self.cur_epoch%self.config.saveEvery) == 0):
            latent_distributions = self.get_mean_sigma()
            counter = 0
            for l in range(len(self.config.Z_SIZES)):
                df = pd.DataFrame(latent_distributions[counter])
                df.to_csv(self.config.results_path + "logs/" +str(self.cur_epoch) + "_d_mu"+str(l)+".csv", header=None)
                df = pd.DataFrame(latent_distributions[counter + 1])
                df.to_csv(self.config.results_path + "logs/" +str(self.cur_epoch) + "_d_sigmas" + str(l) + ".csv", header=None)
                if(l!=len(self.config.Z_SIZES)-1):
                    df = pd.DataFrame(latent_distributions[counter + 2])
                    df.to_csv(self.config.results_path + "logs/" +str(self.cur_epoch) + "_p_mu" + str(l) + ".csv", header=None)
                    df = pd.DataFrame(latent_distributions[counter + 3])
                    df.to_csv(self.config.results_path + "logs/" + str(self.cur_epoch) + "_p_sigmas" + str(l) + ".csv", header=None)
                    df = pd.DataFrame(latent_distributions[counter + 4])
                    df.to_csv(self.config.results_path + "logs/" +str(self.cur_epoch) + "_e_mu" + str(l) + ".csv", header=None)
                    df = pd.DataFrame(latent_distributions[counter + 5])
                    df.to_csv(self.config.results_path + "logs/" +str(self.cur_epoch) + "_e_sigmas" + str(l) + ".csv", header=None)
                counter+=6

        if((self.cur_epoch%self.config.saveEvery) == 0):
            create_dirs([os.path.join(self.config.checkpoint_dir, str(self.cur_epoch))])
            self.model.saver.save(self.sess, os.path.join(self.config.checkpoint_dir, str(self.cur_epoch), str(self.cur_epoch)),
                                  global_step=self.model.global_step_tensor)

    def eval_step(self, lambda_z_wu = 1):
        batch_x, batch_y = next(self.data_eval.next_batch(self.config.batch_size))

        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.lambda_z_wu: lambda_z_wu, self.model.is_training: False}
        z = self.sess.run(self.model.e_mus[self.config.layers_ladder - 1], feed_dict=feed_dict)

        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.lambda_z_wu: lambda_z_wu,
                     self.model.zs[self.config.layers_ladder - 1]: z, self.model.is_training: False}
        ceMLP = self.sess.run(self.model.ceMLP_ba, feed_dict=feed_dict)

        for l in range(self.config.layers_ladder - 2, -1, -1):
            feed_dict = {self.model.x: batch_x,
                         self.model.y: batch_y,
                         self.model.lambda_z_wu: lambda_z_wu,
                         self.model.zs[l+1]: z,
                         self.model.is_training: False}
            z = self.sess.run(self.model.d_mus[l], feed_dict=feed_dict)

        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.lambda_z_wu: lambda_z_wu, self.model.zs[0]: z, self.model.is_training: False}
        tot_loss, dice_loss1, dice_loss2, kls = self.sess.run([self.model.tot_loss,
                                                               self.model.dice_loss1_ba,
                                                               self.model.dice_loss2_ba,
                                                               self.model.kl_terms_ba],
                                                               feed_dict=feed_dict)

        return tot_loss, dice_loss1, dice_loss2, kls, ceMLP

    def eval_recon(self, lambda_z_wu = 1):
        batch_x, batch_y = next(self.data_eval.next_batch(self.config.batch_size))

        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.lambda_z_wu: lambda_z_wu, self.model.is_training: False}
        z = self.sess.run(self.model.e_mus[self.config.layers_ladder - 1], feed_dict=feed_dict)

        for l in range(self.config.layers_ladder - 2, -1, -1):
            feed_dict = {self.model.x: batch_x,
                         self.model.y: batch_y,
                         self.model.lambda_z_wu: lambda_z_wu,
                         self.model.zs[l+1]: z,
                         self.model.is_training: False}
            z = self.sess.run(self.model.d_mus[l], feed_dict=feed_dict)
            # print(l)

        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.lambda_z_wu: lambda_z_wu, self.model.zs[0]: z, self.model.is_training: False}
        x, x_ = self.sess.run([self.model.x, self.model.x_], feed_dict=feed_dict)

        return x, x_


    def train_step(self, lambda_z_wu = 1):
        feed_dict = {self.model.lambda_z_wu: lambda_z_wu, self.model.is_training: True}
        _, tot_loss, dice_loss1, dice_loss2, kls, ceMLP = self.sess.run([self.model.train_step,
                                                                              self.model.tot_loss,
                                                                              self.model.dice_loss1_ba,
                                                                              self.model.dice_loss2_ba,
                                                                              self.model.kl_terms_ba,
                                                                              self.model.ceMLP_ba],
                                                                              feed_dict=feed_dict)

        return tot_loss, dice_loss1, dice_loss2, kls, ceMLP

    def train_recon(self, lambda_z_wu = 1):
        feed_dict = {self.model.lambda_z_wu: lambda_z_wu, self.model.is_training: True}
        x, x_ = self.sess.run([self.model.x, self.model.x_], feed_dict=feed_dict)
        return x, x_

    def get_mean_sigma(self, lambda_z_wu = 1):
    #return distribution of the training set for mu and sigma at all the steps of the ladder
        for l in range(len(self.config.Z_SIZES)):
            exec ("d_mu%s_a =  np.empty((self.config.toPrint, self.config.Z_SIZES[%s]))" % (l,l), globals(), locals())
            exec ("d_sigmas%s_a =  np.empty((self.config.toPrint, self.config.Z_SIZES[%s]))" % (l,l), globals(), locals())
            if(l!=len(self.config.Z_SIZES)-1):
                exec ("p_mu%s_a =  np.empty((self.config.toPrint, self.config.Z_SIZES[%s]))" % (l, l), globals(), locals())
                exec ("p_sigmas%s_a = np.empty((self.config.toPrint, self.config.Z_SIZES[%s]))" % (l, l), globals(), locals())
                exec ("e_mu%s_a =  np.empty((self.config.toPrint, self.config.Z_SIZES[%s]))" % (l, l), globals(), locals())
                exec ("e_sigmas%s_a =  np.empty((self.config.toPrint, self.config.Z_SIZES[%s]))" % (l, l), globals(), locals())

        for i in range(0, self.config.toPrint, self.config.batch_size):
            batch_x = self.data_train.input_x[i:i+self.config.batch_size, :, :, :, :]
            batch_y = self.data_train.input_y[i:i+self.config.batch_size]

            feed_dict = {self.model.x: batch_x,
                         self.model.y: batch_y,
                         self.model.lambda_z_wu: lambda_z_wu,
                         self.model.is_training: False}
            la = self.config.layers_ladder - 1
            command = "locals()['z'], locals()['d_sigmas%s_a'][i:i+self.config.batch_size, :] = " \
                      "self.sess.run([self.model.e_mus[%s], self.model.e_sigmas[%s]], feed_dict=feed_dict)" % (la, la, la)
            exec (command, globals(), locals())
            # print(command)
            command = "locals()['d_mu%s_a'][i:i+self.config.batch_size, :] = locals()['z']" % la
            exec (command, globals(), locals())
            # print(command)

            for l in range(self.config.layers_ladder - 2, -1, -1):
                feed_dict = {self.model.x: batch_x,
                             self.model.y: batch_y,
                             self.model.lambda_z_wu: lambda_z_wu,
                             self.model.zs[l + 1]: locals()['z'] ,
                             self.model.is_training: False}
                command = "locals()['z'], " \
                          "locals()['p_sigmas%s_a'][i:i+self.config.batch_size, :], " \
                          "locals()['e_mu%s_a'][i:i+self.config.batch_size, :], locals()['e_sigmas%s_a'][i:i+self.config.batch_size, :]," \
                          "locals()['d_mu%s_a'][i:i+self.config.batch_size, :], locals()['d_sigmas%s_a'][i:i+self.config.batch_size, :] = " \
                          "self.sess.run([self.model.p_mus[%s], self.model.p_sigmas[%s], self.model.e_mus[%s], self.model.e_sigmas[%s]," \
                          "self.model.d_mus[%s], self.model.d_sigmas[%s]], " \
                          "feed_dict=feed_dict)" % (l, l, l, l, l, l, l, l, l, l, l)
                # print(command)
                exec (command, globals(), locals())
                command = "locals()['p_mu%s_a'][i:i+self.config.batch_size, :] = locals()['z']" % l
                # print(command)
                exec (command, globals(), locals())

        return_expression = []
        for l in range(len(self.config.Z_SIZES)):
            return_expression.append(locals()["d_mu"+str(l)+"_a"])
            return_expression.append(locals()["d_sigmas"+str(l)+"_a"])
            if(l!=len(self.config.Z_SIZES)-1):
                return_expression.append(locals()["p_mu"+str(l)+"_a"])
                return_expression.append(locals()["p_sigmas"+str(l)+"_a"])
                return_expression.append(locals()["e_mu"+str(l)+"_a"])
                return_expression.append(locals()["e_sigmas"+str(l)+"_a"])

        return return_expression
