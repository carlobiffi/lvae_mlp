import tensorflow as tf

from data_loader.data_loader import *
from models.lvae_mlp import lvae_mlp
from trainers.lvae_trainer import train_lvae
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.results_path, config.summary_dir, config.checkpoint_dir, config.results_path+"logs/"])

    coord = tf.train.Coordinator()
    with tf.name_scope('create_inputs'):
        # create your data generator
        data_train = DataGenerator(config, "training", coord)
        input_batch = data_train.dequeue(config.batch_size)
        data_eval = DataGenerator(config, "evaluation")

    # create tensorflow session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    #create threads
    threads = data_train.start_threads(sess, n_threads=config.num_threads)

    # create an instance of the model you want
    model = lvae_mlp(config, input_batch)
    # #load model if it exists
    # model.load_latest(sess)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = train_lvae(sess, model, data_train, data_eval, config, logger)

    # here you train your model
    trainer.train()

    exit()

    #stop threads
    coord.request_stop()
    print("stop requested.")
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()
