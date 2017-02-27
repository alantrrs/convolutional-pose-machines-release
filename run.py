import sys
import numpy as np
import math
import cv2 as cv
import scipy
import time
import copy

sys.path.insert(0, '/caffe/python')
sys.path.insert(0, '/conv-pose/testing/python')

import util
import caffe

# Model
person_deployfile = 'model/_trained_person_MPI/pose_deploy_copy_4sg_resize.prototxt'
person_model = 'model/_trained_person_MPI/pose_iter_70000.caffemodel'
pose_deployfile = 'model/_trained_MPI/pose_deploy_resize.prototxt'
pose_model = 'model/_trained_MPI/pose_iter_320000.caffemodel'
boxsize = 368
sigma = 21
npart = 14

# Read image
oriImg = cv.imread('./testing/sample_image/frame0181.png')

# Scale image
scale = boxsize/(oriImg.shape[0]*1.0)
testImg = cv.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

# Pad image so width and height are multiples of 8
testImg_padded, pad = util.padRightDownCorner(testImg)

# Person network
caffe.set_mode_cpu() # Change to GPU
# caffe.set_device(0)
person_net = caffe.Net(person_deployfile, person_model, caffe.TEST)
person_net.blobs['image'].reshape(*(1, 3, testImg_padded.shape[0], testImg_padded.shape[1]))
person_net.reshape()
person_net.forward() # dry run to avoid GPU sync later in caffe

# Detect persons
person_net.blobs['image'].data[...] = np.transpose(np.float32(testImg_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5
start_time = time.time()
output_blobs = person_net.forward()
print('Person net took %.2f ms' % (1000 * (time.time() - start_time)))
print(output_blobs.keys())
print(output_blobs[output_blobs.keys()[0]].shape)

person_map = np.squeeze(person_net.blobs[output_blobs.keys()[0]].data)

person_map_resized = cv.resize(person_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
data_max = scipy.ndimage.filters.maximum_filter(person_map_resized, 3)
maxima = (person_map_resized == data_max)
diff = (data_max > 0.5)
maxima[diff == 0] = 0
x = np.nonzero(maxima)[1]
y = np.nonzero(maxima)[0]

# Crop persons images
num_people = x.size
person_image = np.ones((boxsize, boxsize, 3, num_people)) * 128
for p in range(num_people):
    for x_p in range(boxsize):
        for y_p in range(boxsize):
            x_i = x_p - boxsize/2 + x[p]
            y_i = y_p - boxsize/2 + y[p]
            if x_i >= 0 and x_i < testImg.shape[1] and y_i >= 0 and y_i < testImg.shape[0]:
                person_image[y_p, x_p, :, p] = testImg[y_i, x_i, :]

# Create gaussian map
gaussian_map = np.zeros((boxsize, boxsize))
for x_p in range(boxsize):
    for y_p in range(boxsize):
        dist_sq = (x_p - boxsize/2) * (x_p - boxsize/2) + \
                  (y_p - boxsize/2) * (y_p - boxsize/2)
        exponent = dist_sq / 2.0 /sigma / sigma
        gaussian_map[y_p, x_p] = math.exp(-exponent)

# Pose network
pose_net = caffe.Net(pose_deployfile, pose_model, caffe.TEST)
pose_net.forward()
output_blobs_array = [dict() for dummy in range(num_people)]
for p in range(num_people):
    input_4ch = np.ones((boxsize, boxsize, 4))
    input_4ch[:,:,0:3] = person_image[:,:,:,p]/256.0 - 0.5 # normalize to [-0.5, 0.5]
    input_4ch[:,:,3] = gaussian_map
    pose_net.blobs['data'].data[...] = np.transpose(np.float32(input_4ch[:,:,:,np.newaxis]), (3,2,0,1))
    start_time = time.time()
    output_blobs_array[p] = copy.deepcopy(pose_net.forward()['Mconv7_stage6'])
    print('For person %d, pose net took %.2f ms.' % (p, 1000 * (time.time() - start_time)))

# Get predictions
prediction = np.zeros((14, 2, num_people))
for p in range(num_people):
    for part in range(14):
        part_map = output_blobs_array[p][0, part, :, :]
        part_map_resized = cv.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
        prediction[part,:,p] = np.unravel_index(part_map_resized.argmax(), part_map_resized.shape)
    # mapped back on full image
    prediction[:,0,p] = prediction[:,0,p] - (boxsize/2) + y[p]
    prediction[:,1,p] = prediction[:,1,p] - (boxsize/2) + x[p]

# Visualize
limbs = 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14
num_limb = len(limbs)/2
limbs = np.array(limbs).reshape((num_limb, 2))
limbs = limbs.astype(np.int)
stickwidth = 6
colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
[255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]] # note BGR ...
canvas = testImg.copy()
for p in range(num_people):
    for part in range(npart):
        cv.circle(canvas, (int(prediction[part, 1, p]), int(prediction[part, 0, p])), 3, (0, 0, 0), -1)
    for l in range(limbs.shape[0]):
        cur_canvas = canvas.copy()
        X = prediction[limbs[l,:]-1, 0, p]
        Y = prediction[limbs[l,:]-1, 1, p]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv.fillConvexPoly(cur_canvas, polygon, colors[l])
        canvas = canvas * 0.4 + cur_canvas * 0.6 # for transparency
cv.imwrite('poses.png', canvas)
