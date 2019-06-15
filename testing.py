import tensorflow as tf
import numpy as np
import nibabel as nib
import os
from utils.myfuncs import *
from utils.hausdorff_util import *

# Compute Accuracy
from sklearn.metrics import accuracy_score
from data_loader.data_loader import DataGenerator
from models.lvae_mlp import lvae_mlp
from utils.config import process_config
from utils.utils import get_args

def testing(config):
    # create tensorflow session
    sess = tf.Session()
    # create an instance of the model you want
    model = lvae_mlp(config)
    # load model from specified checkpoint
    model.load_from_checkpoint(sess)

    # data object containing test data
    dataName = "testing"
    data_test = DataGenerator(config, dataName)

    ########### ########### ########### ########### ########### ###########
    ########### COMPUTE ACCURACY AND SAVE Z_MEAN TO CSV FILE ###########
    ########### PRINT INPUT IMAGE AND ITS RECONSTRUCTION ###########
    ########### ########### ########### ########### ########### ###########

    img_example = nib.load(config.path_example_seg)

    if not os.path.exists(config.results_path + 'recon'):
        os.makedirs(config.results_path + 'recon')

    result_array = np.empty([data_test.get_N_sub(), config.num_classes])
    ind = 0
    lambda_z_wu = 1

    print("Saving mus and sigmas matrices...")

    with open(config.results_path + '/%s_pred.csv' % dataName, 'wb') as outCSVpred:
        with open(config.results_path + '/%s_mean.csv' % dataName, 'wb') as outCSVmean:
            with open(config.results_path + '/%s_sigma.csv' % dataName, 'wb') as outCSVsigma:
                for i in range(0, data_test.get_N_sub()):
                    batch_x = data_test.get_X()[i:i + 1, :, :, :, :].reshape(1, data_test.get_X().shape[1],
                                                                       data_test.get_X().shape[2],
                                                                       data_test.get_X().shape[3],
                                                                       data_test.get_X().shape[4])
                    batch_y = data_test.get_Y()[i:i + 1]

                    feed_dict = {model.x: batch_x,
                                 model.y: batch_y,
                                 model.lambda_z_wu: lambda_z_wu,
                                 model.is_training: False}

                    z, sigma = sess.run([model.e_mus[config.layers_ladder - 1],
                                         model.e_sigmas[config.layers_ladder - 1]], feed_dict=feed_dict)

                    print(z)

                    feed_dict = {model.x: batch_x,
                                 model.y: batch_y,
                                 model.zs[config.layers_ladder - 1]: z,
                                 model.lambda_z_wu: lambda_z_wu,
                                 model.is_training: False}

                    pred = sess.run(model.prob, feed_dict=feed_dict)

                    result_array[ind, :] = pred

                    np.savetxt(outCSVpred, pred, delimiter=",")

                    np.savetxt(outCSVmean, z, delimiter=",")
                    np.savetxt(outCSVsigma, sigma, delimiter=",")

                    print(batch_y)

                    for l in range(config.layers_ladder - 2, -1, -1):
                        feed_dict = {model.x: batch_x,
                                     model.y: batch_y,
                                     model.lambda_z_wu: lambda_z_wu,
                                     model.zs[l + 1]: z,
                                     model.is_training: False}
                        z = sess.run(model.d_mus[l], feed_dict=feed_dict)

                    feed_dict = {model.x: batch_x,
                                 model.y: batch_y,
                                 model.lambda_z_wu: lambda_z_wu,
                                 model.zs[0]: z,
                                 model.is_training: False}
                    rec = sess.run(model.x_, feed_dict=feed_dict)

                    toSave = nib.Nifti1Image(data_test.get_X()[i, :, :, :, 0].reshape(
                        [config.input_shape[0], config.input_shape[1], config.input_shape[2], 1]),
                        img_example.affine)
                    nib.save(toSave, config.results_path + 'ACDC/input_' + str(i) + '_' + dataName + '_c0.nii.gz')
                    toSave = nib.Nifti1Image(data_test.get_X()[i, :, :, :, 1].reshape(
                        [config.input_shape[0], config.input_shape[1], config.input_shape[2], 1]),
                        img_example.affine)
                    nib.save(toSave, config.results_path + 'ACDC/input_' + str(i) + '_' + dataName + '_c1.nii.gz')

                    toSave = nib.Nifti1Image(rec[:, :, :, :, 0].reshape(
                        [config.input_shape[0], config.input_shape[1], config.input_shape[2], 1]),
                        img_example.affine)
                    nib.save(toSave, config.results_path + 'ACDC/d_rec_' + str(i) + '_' + dataName + '_c0.nii.gz')
                    toSave = nib.Nifti1Image(rec[:, :, :, :, 1].reshape(
                        [config.input_shape[0], config.input_shape[1], config.input_shape[2], 1]),
                        img_example.affine)
                    nib.save(toSave, config.results_path + 'ACDC/d_rec_' + str(i) + '_' + dataName + '_c1.nii.gz')

                    ind = ind + 1

    print("Done.")

    print(np.argmax(result_array, axis=1))
    print(data_test.get_Y())

    acc_score = accuracy_score(data_test.get_Y(), np.argmax(result_array, axis=1), normalize=True, sample_weight=None)
    print("Number of testing subjects: %d. Accuracy: %f." % (data_test.get_N_sub(), acc_score))

    ########### ########### ########### ########### ########### ###########
    ########### ########### ########### ########### ########### ###########
    ########### ########### ########### ########### ########### ###########


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    testing(config)


if __name__ == '__main__':
    main()
