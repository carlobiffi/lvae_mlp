import tensorflow as tf
import nibabel as nib
import os
from utils.myfuncs import *
from utils.hausdorff_util import *

# Compute Accuracy
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
    dataName = "training"
    data_test = DataGenerator(config, dataName)

    #  highest latent space coordinates to sampled the desidered shapes
    x1 = [-2.5, 0, 2]
    x2 = [-0.015, 0, 0.015]

    ########### ########### ########### ########### ########### ###########
    ########### COMPUTE ACCURACY AND SAVE Z_MEAN TO CSV FILE ###########
    ########### PRINT INPUT IMAGE AND ITS RECONSTRUCTION ###########
    ########### ########### ########### ########### ########### ###########

    img_example = nib.load(config.path_example_seg)

    if not os.path.exists(config.results_path + 'recon'):
        os.makedirs(config.results_path + 'recon')

    lambda_z_wu = 1
    i=1

    #load test dataset
    batch_x = data_test.get_X()[i:i + 1, :, :, :, :].reshape(1, data_test.get_X().shape[1],
                                                       data_test.get_X().shape[2],
                                                       data_test.get_X().shape[3],
                                                       data_test.get_X().shape[4])
    batch_y = data_test.get_Y()[i:i + 1]

    for i in range(len(x1)):
        for j in range(len(x2)):
            z = [[x1[i], x2[j]]]
            for l in range(config.layers_ladder - 2, -1, -1):
                feed_dict = {model.x: batch_x,
                             model.y: batch_y,
                             model.lambda_z_wu: lambda_z_wu,
                             model.zs[l + 1]: z,
                             model.is_training: False}
                z = sess.run(model.p_mus[l], feed_dict=feed_dict)

            feed_dict = {model.x: batch_x,
                         model.y: batch_y,
                         model.lambda_z_wu: lambda_z_wu,
                         model.zs[0]: z,
                         model.is_training: False}
            rec = sess.run(model.x_, feed_dict=feed_dict)

            toSave = nib.Nifti1Image(rec[:, :, :, :, 0].reshape(
                [config.input_shape[0], config.input_shape[1], config.input_shape[2], 1]),
                                     img_example.affine)
            nib.save(toSave, config.results_path + 'visual/rec_' + str(x1[i]) + '_' + str(x2[j]) + '_c0.nii.gz')
            toSave = nib.Nifti1Image(rec[:, :, :, :, 1].reshape(
                [config.input_shape[0], config.input_shape[1], config.input_shape[2], 1]),
                                     img_example.affine)
            nib.save(toSave, config.results_path + 'visual/rec_' + str(x1[i]) + '_' + str(x2[j]) + '_c1.nii.gz')

    print("Done.")


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
