import sys

import numpy as np
from scipy.misc import imread, imresize
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

CLASS_LIST = ['confuser', 'EM']

im_path = "./EM_example.jpg" if len(sys.argv) <= 1 else sys.argv[1]
model_fold = 1
model_path = "./weights_LYME_C_EM_ALL_FOLD_{}.hdf5".format(model_fold)

model = load_model(model_path)

im = imresize(imread(im_path, mode='RGB'), model.input_shape[1:])

y_score = np.squeeze(model.predict(preprocess_input(np.expand_dims(im, axis=0))))

y_pred = np.argmax(y_score)

print("Predicted {0} with {1} confidence.").format(CLASS_LIST[y_pred], y_score[y_pred])
