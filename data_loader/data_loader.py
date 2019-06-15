import numpy as np
import h5py
import time
from numpy import genfromtxt
from utils.myfuncs import *
import threading
import tensorflow as tf
import sys

class DataGenerator:
    def __init__(self, config, mode, coord=None, max_queue_size=80):
        self.config = config
        # informations in the config.json file

        # define the queue of input data
        self.max_queue_size = max_queue_size
        # FIFO queue
        self.queue = tf.PaddingFIFOQueue(max_queue_size, dtypes=[tf.float32, tf.int32], shapes=[[self.config.input_shape[0],self.config.input_shape[1],self.config.input_shape[2],self.config.input_shape[3]], []])
        self.queue_size = self.queue.size()

        self.config = config
        # informations in the config.json file

        self.threads = []
        self.coord = coord

        self.images_placeholder = tf.placeholder(tf.float32, [None, self.config.input_shape[0], self.config.input_shape[1], self.config.input_shape[2], self.config.input_shape[3]])
        self.labels_placeholder = tf.placeholder(tf.int32, [None])

        # Enqueues zero or more elements to the queue
        self.enqueue = self.queue.enqueue_many([self.images_placeholder, self.labels_placeholder])

        # load train, test or eval data depending on the specified mode
        if(mode == "training"):
            t0 = time.time()
            train_h5_file = h5py.File(self.config.training_segmentations_path, 'r')
            self.input_x = train_h5_file[self.config.training_segmentations_name][:]
            train_h5_file.close()
            self.input_y = genfromtxt(self.config.training_labels_path, delimiter=',')
            print('Read training data in {} sec.'.format(time.time() - t0))

        if(mode == "testing"):
            t0 = time.time()
            train_h5_file = h5py.File(self.config.testing_segmentations_path, 'r')
            self.input_x = train_h5_file[self.config.testing_segmentations_name][:]
            train_h5_file.close()
            self.input_y = genfromtxt(self.config.testing_labels_path, delimiter=',')
            print('Read testing data in {} sec.'.format(time.time() - t0))

        if(mode == "evaluation"):
            t0 = time.time()
            train_h5_file = h5py.File(self.config.eval_segmentations_path, 'r')
            self.input_x = train_h5_file[self.config.eval_segmentations_name][:]
            train_h5_file.close()
            self.input_y = genfromtxt(self.config.eval_labels_path, delimiter=',')
            print('Read evaluation data in {} sec.'.format(time.time() - t0))

        print("Data shape: ", self.input_x.shape)

        for l in range(self.config.num_classes):
            temp = [i for i in range(0,len(self.input_y)) if self.input_y[i] == l]
            exec ("self.class%s = temp" % l, globals(), locals())
            exec ("print(self.class%d)" % l)

    #get the number of subjects
    def get_N_sub(self):
        return self.input_x.shape[0]

    #get X
    def get_X(self):
        return self.input_x

    #get Y
    def get_Y(self):
        return self.input_y

    # dequeues and concatenates num_elements elements from this queue.
    def dequeue(self, num_elements):
        output= self.queue.dequeue_many(num_elements)
        return output

    # thread to load data into the queue
    def thread_main(self, sess):
        stop = False
        while not stop:
            lista = np.random.choice(self.input_x.shape[0], self.input_x.shape[0], replace=False)
            for i in range(0,self.input_x.shape[0],self.config.feed_size):
                # keep looping until the queue has less elements than the maximum size
                while self.queue_size.eval(session=sess) == self.max_queue_size:
                    if self.coord.should_stop():
                        break
                # enqueue
                # DATA AUGMENTATION
                # sess.run(self.enqueue, feed_dict={self.images_placeholder: data_augmenter(self.input_x[lista[i:i+self.config.feed_size], :, :, :, :]),
                #                                   self.labels_placeholder: self.input_y[lista[i:i + self.config.feed_size]]})
                # NO DATA AUGMENTATION
                sess.run(self.enqueue, feed_dict={self.images_placeholder: self.input_x[lista[i:i+self.config.feed_size], :, :, :, :],
                                                  self.labels_placeholder: self.input_y[lista[i:i+self.config.feed_size]]})

                # stop queuing process if you receive a stop request
                if self.coord.should_stop():
                    stop = True
                    print("Enqueue thread receives stop request.")
                    break

    # function called before training to start threads
    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

    def next_batch_da(self, batch_size):
        b = int(batch_size/self.config.num_classes)
        command = "idx = np.concatenate(("
        for l in range(self.config.num_classes):
            exec ("idx%s = np.random.choice(self.class%s, b)" % (l,l), globals(), locals())
            if l==0:
                command += "locals()['idx" + str(l) + "']"
            else:
                command += ",locals()['idx" + str(l) + "']"
        command += "), axis=0)"
        exec (command, globals(), locals())
        sample = locals()['idx']

        input = self.input_x[sample,:,:,:,:]

        yield input, self.input_y[sample]

    def next_batch(self, batch_size):
        b = int(batch_size/self.config.num_classes)
        command = "idx = np.concatenate(("
        for l in range(self.config.num_classes):
            exec ("idx%s = np.random.choice(self.class%s, b)" % (l,l), globals(), locals())
            if l==0:
                command += "locals()['idx" + str(l) + "']"
            else:
                command += ",locals()['idx" + str(l) + "']"
        command += "), axis=0)"
        exec (command, globals(), locals())
        sample = locals()['idx']

        input = self.input_x[sample,:,:,:,:]

        yield input, self.input_y[sample]
