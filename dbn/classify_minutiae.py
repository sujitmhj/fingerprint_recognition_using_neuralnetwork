# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cv2
import scipy.io as sio
import pickle

# grab the MNIST dataset (if this is the first time you are running
# this script, this make take a minute -- the 55mb MNIST digit dataset
# will be downloaded)
# print "[X] downloading data..."
dataset = datasets.fetch_mldata("MNIST Original")

# scale the data to the range [0, 1] and then construct the training
# and testing splits

(trainX, testX, trainY, testY) = train_test_split(
	dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33)
# print trainX.shape, trainY.shape
# print type(trainY), trainY


count = 0

# train the Deep Belief Network with 784 input units (the flattened,
# 28x28 grayscale image), 300 hidden units, 10 output units (one for
# each possible output classification, which are the digits 1-10)

try:
	with open('data.pkl', 'rb') as input:
	    dbn = pickle.load(input)

except:
    print("Neural Network is not trained")

# # randomly select a few of the test instances
# for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
# 	# classify the digit
# 	pred = dbn.predict(np.atleast_2d(testX[i]))
 
# 	# reshape the feature vector to be a 28x28 pixel image, then change
# 	# the data type to be an unsigned 8-bit integer
# 	image = (testX[i] * 255).reshape((28, 28)).astype("uint8")
 
# 	# show the image and prediction
# 	print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
# 	cv2.imshow("Digit", image)
# 	cv2.waitKey(0) 

def get_window_image(img,row,column):
    # print row, row
    half = 10
    try:
        img_croped = img[column-half:column+half, row-half:row+half]
        created = True
    except:
        created = False
        img_croped = None
    # print img_croped.shape, "lll"
    return created, img_croped



img = cv2.imread("./input.png")
inv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = inv

x_top = 0
y_top = 0
x_bottom = 0
y_bottom = 0

for x,row in enumerate(inv):
    for y,pix in enumerate(row):
        if pix>100:
            if x<x_top:
                x_top = x
            if x>x_bottom:
                x_bottom = x
            if y<y_top:
                y_top = y
            if y>y_bottom:
                y_bottom = y

img_croped = inv[x_top:x_bottom, y_top:y_bottom]
if img_croped.shape[0] > img_croped.shape[1]:
    size_max = img_croped.shape[0]
else:
    size_max = img_croped.shape[1]
padding = 3
size_max = size_max + 2*padding
blank_image = np.zeros((size_max,size_max), np.uint8)
height_offset = (size_max - img_croped.shape[0])/2
width_offset = (size_max - img_croped.shape[1])/2
blank_image[height_offset:height_offset + img_croped.shape[0],width_offset:width_offset + img_croped.shape[1]] = img_croped
final = cv2.resize(blank_image, (200, 200))
# cv2.imshow('img',blank_image)
# cv2.waitKey(0)

print "Searching Minutiae in image"


for i in range(10,gray.shape[0]-11):
    for j in range(10, gray.shape[1]-11):
        if gray[i,j] == 255:
            continue
        # print i,j
        created, window_image = get_window_image(gray,j,i)
        print window_image.shape




        # final_image = cv2.resize(window_image, (5, 5))


        # print final.shape
        # print "gone",window_image.shape[1], created
        final_image = np.ravel(window_image)/255
        print final_image.shape

        pred = dbn.predict(np.atleast_2d(final_image))

        # print "The input image is ", nepali[int(pred[0])]
        if int(pred[0]) == 2:
            cv2.circle(img, (j,i),3,255,1)
            # cv2.imwrite('circle.png', img2)

            count = count+1
            # cv2.imshow('img',window_image)
            # cv2.waitKey(0)
# cv2.imshow('img',img)
cv2.imwrite("output.jpg", img)
# cv2.waitKey(0)

print count