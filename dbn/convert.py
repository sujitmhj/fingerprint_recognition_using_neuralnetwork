import cv2
import numpy as np
import os
import scipy.io as sio
def get_formatted_img(file_name):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255-gray
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
    padding = 0
    size_max = size_max + 2*padding
    blank_image = np.zeros((size_max,size_max), np.uint8)
    height_offset = (size_max - img_croped.shape[0])/2
    width_offset = (size_max - img_croped.shape[1])/2
    blank_image[height_offset:height_offset + img_croped.shape[0],width_offset:width_offset + img_croped.shape[1]] = img_croped
    final = cv2.resize(blank_image, (5, 5))
    # print final_image.shape
    # cv2.imshow('img',final)
    # cv2.waitKey(0)
    return np.ravel(final)
print "completed"

rootdir = '../image_database/'
nepali = [
    ]



output = {}
flag = False
label_array = []
mapper = {"endings":1,"bifurcations":2, "lines":3,"multiple":3}

for subdir, dirs, files in os.walk(rootdir):
    # print files
    for file in files:
        # try:

        folder_no = float(mapper[subdir.split("/")[-1].split("-")[0]])
        if ".tif" in file:
            image_matrix = get_formatted_img(os.path.join(subdir, file))

            try:
                output['data'] = np.column_stack((output['data'],image_matrix))
                # print output["data"]
            except:
                output['data'] = image_matrix
            label_array.append(folder_no)
            # flag = True
        # except:
        #     pass
        # if flag:
        #     break
output['label'] = np.array(label_array)
                
output['mldata_descr_ordering'] = np.array([[np.array("label"),np.array("data")]],dtype="object")
sio.savemat("./mnist-original.mat", output)
