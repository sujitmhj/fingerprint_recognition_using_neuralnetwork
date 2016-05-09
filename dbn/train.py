

# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cv2
import scipy.io as sio
# grab the MNIST dataset (if this is the first time you are running
# this script, this make take a minute -- the 55mb MNIST digit dataset
# will be downloaded)
print "[X] downloading data..."
dataset = datasets.fetch_mldata("MNIST Original")

# scale the data to the range [0, 1] and then construct the training
# and testing splits

# (trainX, testX, trainY, testY) = train_test_split(
# 	dataset.data / 255.0, dataset.target.astype("U7"), test_size = 0.33)


# dataset.data = dataset.data/255.0
dataset = sio.loadmat("./mnist-original.mat")
# dataset = sio.loadmat("/home/sujit/scikit_learn_data/mldata/mnist-original.mat")
trainX = dataset['data'].T[0:500,0:]/255.0
trainY = dataset['label'][0][0:20000]


testX = dataset['data'].T[0:,0:]/255.0
testY = dataset['label'][0][0:]
# train the Deep Belief Network with 784 input units (the flattened,
# 28x28 grayscale image), 300 hidden units, 10 output units (one for
# each possible output classification, which are the digits 1-10)
print trainX.shape, trainY.shape
# trainY = np.array(range(2000))
# print type(trainY), trainY
# exit()
print "shape[0]",trainX.shape[1]
dbn = DBN(
	[trainX.shape[1], 200000, 3],
	learn_rates = 0.3,
	learn_rate_decays = 0.9,
	epochs = 10 ,
	verbose = 1)
dbn.fit(trainX, trainY)


# # compute the predictions for the test data and show a classification
# # report
# preds = dbn.predict(testX)
# print classification_report(testY, preds)

# randomly select a few of the test instances
for i in np.random.choice(np.arange(0, len(testY)), size = (500,)):
	# classify the digit
	pred = dbn.predict(np.atleast_2d(testX[i]))
 
	# reshape the feature vector to be a 28x28 pixel image, then change
	# the data type to be an unsigned 8-bit integer
	image = (testX[i] * 255).reshape((5, 5)).astype("uint8")

	# show the image and prediction
	print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
	cv2.imshow("Digit", image)
	cv2.waitKey(0)