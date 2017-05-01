import keras
from keras import backend

import os

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.utils import cnn_model
from Gnet import G_10, G_11, G_12, G_13

FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_string('train_dir', '/home/boyuan/Documents/research/Advesarial_examples/Project/test_store','training director')
flags.DEFINE_string('filename', 'cnnmodel',
                    'model name')
flags.DEFINE_string('train_dir_G10', '/home/boyuan/Documents/research/Advesarial_examples/Project/store/G10','training director')
flags.DEFINE_string('filename_G10', 'G10model',
                    'model name')
flags.DEFINE_string('train_dir_G11', '/home/boyuan/Documents/research/Advesarial_examples/Project/store/G11','training director')
flags.DEFINE_string('filename_G11', 'G11model',
                    'model name')
flags.DEFINE_string('train_dir_G12', '/home/boyuan/Documents/research/Advesarial_examples/Project/store/G12','training director')
flags.DEFINE_string('filename_G12', 'G12model',
                    'model name')
flags.DEFINE_string('train_dir_G13', '/home/boyuan/Documents/research/Advesarial_examples/Project/store/G13','training director')
flags.DEFINE_string('filename_G13', 'G13model',
                    'model name')
def main(argv=None):
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()

    assert Y_train.shape[1] == 10.
    label_smooth = .1

    # clip the label between(0.01 to 0.9)
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    print ("Starting G_10 session...")
    with tf.Session() as sess:
        def evaluate0():
            # Evaluate the accuracy of the MNIST model on legitimate test examples
            eval_params = {'batch_size': FLAGS.batch_size}
            accuracy = model_eval(sess, x, y, predictions_0, X_test, Y_test,
                                  args=eval_params)
            assert X_test.shape[0] == 10000, X_test.shape
            print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Define TF model graph
        model_0 = G_10()
        predictions_0 = model_0(x)
        print("Defined G10 model graph.")

        train_params0 = {
        'nb_epochs': 6,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'train_dir': FLAGS.train_dir_G10,
        'filename': FLAGS.filename_G10
        }
        model_train(sess, x, y, predictions_0, X_train, Y_train, save = True,
                    evaluate=evaluate0, args=train_params0)

    print ("Starting G_11 session...")
    with tf.Session() as sess:
        def evaluate1():
            # Evaluate the accuracy of the MNIST model on legitimate test examples
            eval_params = {'batch_size': FLAGS.batch_size}
            accuracy = model_eval(sess, x, y, predictions_1, X_test, Y_test,
                                  args=eval_params)
            assert X_test.shape[0] == 10000, X_test.shape
            print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Define TF model graph
        model_1 = G_11()
        predictions_1 = model_1(x)
        print("Defined G11 model graph.")

        train_params1 = {
        'nb_epochs': 6,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'train_dir': FLAGS.train_dir_G11,
        'filename': FLAGS.filename_G11
        }
        model_train(sess, x, y, predictions_1, X_train, Y_train, save = True,
                    evaluate=evaluate1, args=train_params1)
    

    print ("Starting G_12 session...")
    with tf.Session() as sess:
        def evaluate2():
            # Evaluate the accuracy of the MNIST model on legitimate test examples
            eval_params = {'batch_size': FLAGS.batch_size}
            accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test,
                                  args=eval_params)
            assert X_test.shape[0] == 10000, X_test.shape
            print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Define TF model graph
        model_2 = G_12()
        predictions_2 = model_2(x)
        print("Defined G12 model graph.")

        train_params2 = {
        'nb_epochs': 6,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'train_dir': FLAGS.train_dir_G12,
        'filename': FLAGS.filename_G12
        }
        model_train(sess, x, y, predictions_2, X_train, Y_train, save = True,
                    evaluate=evaluate2, args=train_params2)

    print ("Starting G_13 session...")
    with tf.Session() as sess:
        def evaluate3():
            # Evaluate the accuracy of the MNIST model on legitimate test examples
            eval_params = {'batch_size': FLAGS.batch_size}
            accuracy = model_eval(sess, x, y, predictions_3, X_test, Y_test,
                                  args=eval_params)
            assert X_test.shape[0] == 10000, X_test.shape
            print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Define TF model graph
        model_3 = G_13()
        predictions_3 = model_3(x)
        print("Defined G13 model graph.")

        train_params3 = {
        'nb_epochs': 6,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'train_dir': FLAGS.train_dir_G13,
        'filename': FLAGS.filename_G13
        }
        model_train(sess, x, y, predictions_3, X_train, Y_train, save = True,
                    evaluate=evaluate3, args=train_params3)

    def model_eval_single(sess, x, y, model, X_test, args=None):
        '''
        compute single image output
        '''
        acc_value = model


        with sess.as_default():
            cur_acc = acc_value.eval(
                feed_dict={x: X_test,
                           keras.backend.learning_phase(): 0})


            # Divide by number of examples to get final value
            accuracy = cur_acc

    return accuracy

if __name__ == '__main__':
    app.run()