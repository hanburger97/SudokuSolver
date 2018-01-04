"""
SE-101 Project Group 47
A python Client for interaction with MNIST handwritten digits CNN models
For Education Purpose Only
General Public Licence

---Authors---
Han Y. Xiao
Bsc. Software Eng. 2022
University of Waterloo

Ayush Kapur
Bsc. Software Eng. 2022
University of Waterloo

Jordan Dearsley
Bsc. Software Eng. 2022
University of Waterloo
"""


import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
from tensorflow.examples.tutorials.mnist import input_data
from helpers import Helpers
import cv2


#from cells import Cells
#from digit import Digit

class Brain:
    """
    Client class for interaction with the Convolutional Neural Network and the model
    for handwritten digit recognitions
    Wrapper to Tensorflow models
    Methods:
        - train_model_mnist() : Train the default MNIST Data set and save the model
        - predict and predictMultiple(): Returns the integer between 0 and 9 it predicts.
    """

    def __init__(self,
                 learning_rate=0.001,
                 nb_steps=500,
                 batch_size=128,
                 display_step=10,
                 nb_input=784,          # MNIST data input (img shape: 28*28)
                 nb_classes=10,         # MNIST total classes (0-9 digits)
                 dropout_rate=0.75,      # Dropout, probability to keep units
                 model_path='./model/mnist'
                 ):

        self.learning_rate = learning_rate
        self.num_steps = nb_steps
        self.batch_size = batch_size
        self.display_step = display_step
        self.num_input = nb_input
        self.num_classes = nb_classes

        self.dropout = dropout_rate

        self.X = tf.placeholder(tf.float32, [None, self.num_input], name='X')
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name='Y')
        self.keep_prob = tf.placeholder(tf.float32)

        self.model_path = model_path
        #JORDAN STUFF
        self.helpers = Helpers() 

    def initialize_weights_biases(self, weights=None, biases=None):
        """
        Only input weights and biases if you know what you are doing.
        Ayush, back off!
        :param weights: For custom weights dictionaries only
        :param biases: For custom biases dictionaries only
        :return:
        """

        if weights : self.weights = weights
        else:
            self.weights = {
                # 5x5 conv, 1 input, 32 outputs
                'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
                # 5x5 conv, 32 inputs, 64 outputs
                'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                # fully connected, 7*7*64 inputs, 1024 outputs
                'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
                # 1024 inputs, 10 outputs (class prediction)
                'out': tf.Variable(tf.random_normal([1024, self.num_classes]))
            }

        if biases : self.biases = biases
        else:
            self.biases = {
                'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([64])),
                'bd1': tf.Variable(tf.random_normal([1024])),
                'out': tf.Variable(tf.random_normal([self.num_classes]))
            }

    def new_conv_layer(self, X, bias, weights, strides=1, padding='SAME'):
        """
        Make new convolutioned layer a.k.a. Filtering layer,
        With the weight matrix over window side and do Dot Product ('Shoutout to Prof. Faustin MATH115')
        then add the scalar with the bias scalar and repeat for next window
        :param X: previous layers, z feature maps of m x n matrix
        :param bias: scalar
        :param weights: matrix same size as filter
        :param strides: scalar
        :param padding: string either 'SAME' or 'VALID'
        :return:
        """
        X = tf.nn.conv2d(X, filter=weights, strides=[1, strides, strides, 1], padding=padding)
        X = tf.nn.bias_add(X, bias)
        X = tf.nn.relu(X)
        return X

    def new_pool_layer(self, X, size=2, strides=2, padding='SAME'):
        """
        Computes a pooling layer, basically a matrix size reduction by keeping the max value within the window and
        discarding the rest
        :param X: previous layer, z feature maps of m x n matrix
        :param size: window size for pooling matrix, scalar
        :param strides:scalar; windows size movement
        :param padding:string either 'SAME' or 'VALID'
        :return: Pooling Layer
        """
        X = tf.nn.max_pool(X, ksize=[1, size, size, 1], strides=[1, strides, strides, 1], padding=padding)
        return X

    def new_fully_connected_layer(self, X, bias, weights):
        """
        Generates a Dense Layer, Basically Matrix Multiplication with Weights then Element-wise addition with Biases
        :param X: previous layers, z feature maps of m x n matrix
        :param bias: scalar
        :param weights: matrix same size as filter
        :return: Dense Layer
        """
        X = tf.reshape(X, [-1, weights.get_shape().as_list()[0]])
        X = tf.add(tf.matmul(X, weights), bias)
        X = tf.nn.relu(X)
        return X

    def new_weight(self, size_list):
        """
        Randomly seed the weights matrices with floats
        :param size_list: array to describe the shape of the weight matrix, ref. to Tensorflow Variable
        :return: Tensorflow Variable of seeded matrix
        """
        try:
            assert size_list
            return tf.Variable(tf.random_normal(size_list))
        except AssertionError:
            print('You must give the shape of the weights in an array format, ref. Tensorflow')
        except Exception as e:
            print(e)

    def new_biasess(self, size):
        """
        Randomly seed the bias vector with floats
        :param size: Scalar, size of the bias vector (1 x size)
        :return: Tensorflow Variable of bias vector
        """
        try:
            assert size
            return tf.Variable(tf.random_normal([size]))
        except AssertionError:
            print('You must define the size of the bias vector')
        except Exception as e:
            print(e)

    def create_network(self):
        """
        Generating a Neural Network following optimal sizes according to Tensorflow Documentation for
        MNIST data set
        :return: Output layer which is 1x10
        """

        self.initialize_weights_biases()
        self.x = tf.reshape(self.x, shape=[-1, 28,28,1])
        self.c1 = self.new_conv_layer(self.x, weights=self.weights['wc1'], bias=self.biases['bc1'])
        self.c1 = self.new_pool_layer(self.c1, size=2)

        self.c2 = self.new_conv_layer(self.c1, weights=self.weights['wc2'], bias=self.biases['bc2'])
        self.c2 = self.new_pool_layer(self.c2, size=2)

        self.d1 = tf.reshape(self.c2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        self.d1 = tf.add(tf.matmul(self.d1, self.weights['wd1']), self.biases['bd1'])
        self.d1 = tf.nn.relu(self.d1)

        self.d1 = tf.nn.dropout(self.d1, self.dropout)

        self.out = tf.add(tf.matmul(self.d1, self.weights['out']), self.biases['out'])
        return self.out

    def construct_model(self):
        """
        Constructing our model for trainning and predicting
        :return: None
        """
        # copy of the placeholder else it will mutate self.X which will give fucked up errors
        self.x = self.X

        self.logits = self.create_network()
        self.prediction = self.logits
        self.prediction = tf.nn.softmax(logits=self.logits)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.prediction, axis=1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

    def openCVProcessing(self, array):
        """

        :param array: 784 x 1
        :return: 28 by 28??? Need flattening??
        """
        pass
        #c = Cells()

        #cell = c.clean(array)
        #digit= Digit(cell).digit

        #digit = c.centerDigit(digit)

        #cell.append(digit // 255)
        # ??

        #return cell
    def imgProcess(self, img):
        """
        Converts it to graysclae, Size it to 28x28, Flattens it to 1x784 then normalize value from 0.0 to 1.0
        :param img: Pillow Image Class
        :return: np.ndarray of 1 x 784 as required for our Neural Network's input
        """
        try:
            #assert type(img) == Image
            width = float(img.size[0])
            height = float(img.size[1])
            newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

            if width > height:  # check which dimension is bigger
                # Width is bigger. Width becomes 20 pixels.
                nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width

                if (nheight == 0):  # rare case but minimum is 1 pixel
                    nheigth = 1
                    # resize and sharpen
                img = img.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
                newImage.paste(img, (4, wtop))  # paste resized image on white canvas
            else:
                # Height is bigger. Heigth becomes 20 pixels.
                nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
                if (nwidth == 0):  # rare case but minimum is 1 pixel
                    nwidth = 1
                    # resize and sharpen
                img = img.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
                newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

                # newImage.save("sample.png")

            tv = list(newImage.getdata())  # get pixel values

            # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
            tva = [(255 - x) * 1.0 / 255.0 for x in tv]

            # This makes it into a np.ndarray structure
            res = np.array([tva], np.float32)

            return res
        except AssertionError:
            print('Error Format Must be Image')
        except Exception as e:
            print(e)
            
    def jordanImageProcess(self,digitMatrix):
        convertedImg = self.convertFromHan(digitMatrix)
        preProcessed = self.preprocess(convertedImg)
        cleanedImage = self.clean(preProcessed)
        npFormat = self.convertToHan(cleanedImage)
        return npFormat
        
    def loadImage(self, path):
        color_img = cv2.imread(path)
        if color_img is None:
            raise IOError('Image not loaded')
        print('Image loaded.')
        return color_img
    
    def preprocess(self,image):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.helpers.thresholdify(255-image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image
    
    def clean(self, cell):
        contour = self.helpers.largestContour(cell.copy())
        x, y, w, h = cv2.boundingRect(contour)
        cell = self.helpers.make_it_square(cell[y:y + h, x:x + w], 28)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
        cell = 255 * (cell // 130)
        return cell
    
    def convertFromHan(self,diget):
        converted = np.array(diget).tolist()
        newDiget = []
        row = []
        index = 0
        for i in range(0,len(converted)):
            if(i%28==0 and not(i==0)):
                newDiget.append(list(row))
                row = []
            row.append(converted[i])
        newDiget.append(list(row))
        npa = np.asarray(newDiget, dtype=np.int)

        return npa

    def convertToHan(self,diget):
        newDiget = []
        for row in diget:
            for pixel in row:
                if pixel == 0:
                    newDiget.append(0)
                elif pixel==255:
                    newDiget.append(1)
        npa = np.asarray(newDiget, dtype=np.int)
        return npa
    ####################################################################################################################
    #       Client Methods
    ####################################################################################################################

    def train_model_mnist(self, restore_continue=False):
        """
        This function will take the default MNIST Dataset from Tensorflow and train a model and
        save it at the model path defined in the Class init
        """
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


        self.construct_model()
        with tf.Session() as sess:
            sess.run(self.init)



            for step in range(1, self.num_steps + 1):
                saver = tf.train.Saver()

                if restore_continue:
                    saver.restore(sess, self.model_path)

                batch_x, batch_y = mnist.train.next_batch(self.batch_size)

                new_batch_x = []

                for el in batch_x:
                    new_batch_x.append(self.jordanImageProcess(el * 255))

                # Run optimization op (backprop)
                sess.run(self.train_op, feed_dict={self.X: new_batch_x, self.Y: batch_y, self.keep_prob: self.dropout})
                if step % self.display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.X: new_batch_x,
                                                                         self.Y: batch_y,
                                                                         self.keep_prob: self.dropout})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

            print("Optimization Finished!")

            # Calculate accuracy for 256 MNIST test images
            print("Testing Accuracy:", \
                  sess.run(self.accuracy, feed_dict={self.X: mnist.test.images[:256],
                                                self.Y: mnist.test.labels[:256],
                                                self.keep_prob: self.dropout}))
            path = self.model_path if self.model_path else './model/mnist'
            saver.save(sess, path)


    def predict(self,
                modelPath=None,
                input_data=None     #The input data must have a format of a 28X28 np.array
                ):

        """
        :param input_data: <np.array> of flattened 28 x 28 => 1x 784 grayscale
        :return: integer of the estimation
        """
        # Load Model and Graph and Variables from file
        path = modelPath if modelPath else self.model_path
        saver = tf.train.Saver()

        try:
            assert type(input_data) == np.ndarray
            self.construct_model()
            with tf.Session() as sess:
                sess.run(self.init)
                saver.restore(sess, path)
                res= self.prediction.eval(feed_dict={self.X: [input_data], self.keep_prob: self.dropout}, session=sess)
                return np.argmax(res[0])
        except AssertionError:
            #   Returns -1 for error
            print('Error Please enter valid Input data for prediction')
            return -1
        except Exception as e:
            print(e)
            return -1


    def predictMultiple(self,
                        modelPath=None,
                        inputs=None
                        ):
        """
        Similar to predict function, this takes multiple np.ndarray in a bigger array
        and predicts them all within the same session and model load, instead of reloading and
        creating a session each time
        :param modelPath: Relative path to saved model
        :param inputs: Array of np.ndarray (1 X 784)
        :return: array of predictions
        """
        path = modelPath if modelPath else self.model_path
        preds = []

        try:
            self.construct_model()
            sess =tf.Session()
            saver = tf.train.Saver(max_to_keep=1)
            sess.run(self.init)
            saver.restore(sess, path)
            for el in inputs:
                assert type(el) == np.ndarray
                if self.isEmpty(el):
                    r= 0
                else:
                    r = np.argmax(self.prediction.eval(feed_dict={self.X:[el], self.keep_prob: 1.0}, session=sess)[0])
                preds.append(r)
            #sess.close()
            saver.save(sess, path)

            return preds
        except AssertionError:
            print('Not right format')
            return -1
        except Exception as e:
            print(e)
            return -1

    def isEmpty(self, ndarray):
        try:
            count = 0;
            for el in np.nditer(ndarray, op_flags=['readwrite']):
                if el[...] > 0.0:
                    count += 1
            return True if count<=10 else False
        except Exception as e:
            return e
    def imgProcessFromFile(self, file=None):
        """
            :param file: path to RGB png file
            :return: np.ndarray 1x784 as required for prediction function
        """
        try:
            assert file
            img = Image.open(file).convert('L')
            return self.imgProcess(img)
        except AssertionError:
            print('Must provide path to png file as argument')

    def imgProcessFromArray(self, array):
        """
        :param array: Numpy Array with shape (width, height, 3) must be 8 bit color channel => max value 255
        :return: np.ndarray 1x784 as required for prediction function
        """
        try:
            assert type(array) == np.array
            img = Image.fromarray(array.astype('unint8'))
            return self.imgProcess(img)
        except AssertionError:
            print('Must provide valid Numpy Array')



if __name__=='__main__':


    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    batch_x, batch_y = mnist.train.next_batch(5)

    b = Brain()

    #res = b.openCVProcessing(batch_x[0])
    #print(res)

    """
    trial = b.imgProcessFromFile('./test.png')
    r = b.predict(input_data=trial[0])
    print(r)
    """
    b.train_model_mnist(restore_continue=False)

    #print(type(batch_x[0]))
    #res = b.predict(input_data=batch_x[0])
    """
    res = b.predictMultiple(inputs=batch_x)
    batch_x, batch_y = mnist.train.next_batch(5)

    re2 = b.predictMultiple(inputs=batch_x)

    print(res)

    print(re2)
    """
