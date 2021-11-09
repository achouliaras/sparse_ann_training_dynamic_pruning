# Keras Using Tensorflow Backend

from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
import keras
import numpy as np
from keras import backend as K
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical

def read_data():
    #read Fashion MNIST data
    num_classes=10

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #normalize data
    xTrainMean = np.mean(x_train, axis=0)
    xTtrainStd = np.std(x_train, axis=0)
    x_train = (x_train - xTrainMean) / xTtrainStd
    x_test = (x_test - xTrainMean) / xTtrainStd

    x_train= np.expand_dims(x_train, axis=3)
    x_test= np.expand_dims(x_test, axis=3)

    return [x_train, x_test, y_train, y_test]

class Constraint(object):
    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class MaskWeights(Constraint):

    def __init__(self, mask):
        self.mask = mask
        self.mask = K.cast(self.mask, K.floatx())

    def __call__(self, w):
        #w.assign(w * self.mask)
        w *= self.mask
        return w

    def get_config(self):
        return {'mask': self.mask}

def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx

def createWeightsMask(epsilon,noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)  # normal tp have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    print ("Create Sparse Matrix: No parameters, NoRows, NoCols ",noParameters,noRows,noCols)
    return [noParameters,mask_weights]

def rewireMask(zeta, weights, noWeights):
        # rewire weight matrix

        # remove zeta largest negative and smallest positive weights
        values = np.sort(weights.ravel())
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)
        largestNegative = values[int((1-zeta) * firstZeroPos)]
        smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + zeta * (values.shape[0] - lastZeroPos)))]
        rewiredWeights = weights.copy();
        rewiredWeights[rewiredWeights > smallestPositive] = 1;
        rewiredWeights[rewiredWeights < largestNegative] = 1;
        rewiredWeights[rewiredWeights != 1] = 0;
        weightMaskCore = rewiredWeights.copy()

        # add zeta random weights
        nrAdd = 0
        noRewires = noWeights - np.sum(rewiredWeights)
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

        return [rewiredWeights, weightMaskCore]

def weightsEvolution():   
    # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
    w1 = model.get_layer("sparse_1").get_weights()
    w2 = model.get_layer("sparse_2").get_weights()
    w3 = model.get_layer("sparse_3").get_weights()
    w4 = model.get_layer("dense_4").get_weights()

    [wm1, wm1Core] = rewireMask(zeta, w1[0], noPar1)
    [wm2, wm2Core] = rewireMask(zeta, w2[0], noPar2)
    [wm3, wm3Core] = rewireMask(zeta, w3[0], noPar3)

    w1[0] = w1[0] * wm1Core
    w2[0] = w2[0] * wm2Core
    w3[0] = w3[0] * wm3Core

def linear_decreasing_variance(max_iter, curr_iter):
        return min_zeta+(max_zeta-min_zeta)*(max_iter-curr_iter)/(max_iter)

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        zeta = linear_decreasing_variance(maxepoches,epoch)
        print("Zeta=",zeta)
        if (epoch < maxepoches-1):
            weightsEvolution()

# set model parameters
epsilon = 20 # control the sparsity level as discussed in the paper
zeta = 0.3 # the fraction of the weights removed
batch_size = 100 # batch size
maxepoches = 500 # number of epochs
learning_rate = 0.01 # SGD learning rate
num_classes = 10 # number of classes
momentum=0.9 # SGD momentum

max_zeta=0.3
min_zeta=0.01

#neurons per layer
hidden_layer1=1000
hidden_layer2=1000
hidden_layer3=1000

# generate an Erdos Renyi sparse weights mask for each layer
[noPar1, wm1] = createWeightsMask(epsilon,28 * 28 *1, hidden_layer1)
[noPar2, wm2] = createWeightsMask(epsilon,hidden_layer1, hidden_layer2)
[noPar3, wm3] = createWeightsMask(epsilon,hidden_layer2, hidden_layer3)

# initialize layers weights
w1 = None
w2 = None
w3 = None
w4 = None

# create a SET-MLP model for Fashion_MNIST with 3 hidden layers
model = keras.Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(hidden_layer1, name="sparse_1",kernel_constraint=MaskWeights(wm1),weights=w1, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(hidden_layer2, name="sparse_2",kernel_constraint=MaskWeights(wm2),weights=w2, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(hidden_layer3, name="sparse_3",kernel_constraint=MaskWeights(wm3),weights=w3, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, name="dense_4",weights=w4, activation='softmax'))

# read Fashion MNIST data
[x_train,x_test,y_train,y_test]=read_data()

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.005,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.005,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

model.summary()

# training process in a for loop

sgd = optimizers.SGD(lr=learning_rate, momentum=momentum)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

historytemp = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=x_train.shape[0]//batch_size,
                        epochs=maxepoches, validation_data=(x_test, y_test), callbacks=[CustomCallback()])

accuracies_per_epoch=[]
accuracies_per_epoch= historytemp.history['val_accuracy']
accuracies_per_epoch=np.asarray(accuracies_per_epoch)

loss, acc = model.evaluate(x_test, y_test,verbose=0)
print("Accuracy on test data:", acc)