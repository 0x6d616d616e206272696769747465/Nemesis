import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob
import matplotlib
matplotlib.use('Agg')

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

# This is needed to display the images.
#%matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = '/../../../inference_grah'                           # Fast
#MODEL_NAME = '../../../faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'      # Slow Best results
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 12

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Find all the .jpg images in the video_output folder
PATH_TO_TEST_IMAGES_DIR = '/../../../../video_output'
PATH_TO_OUTPUT_IMAGES_DIR = PATH_TO_TEST_IMAGES_DIR + "/output"
file_list = sorted(glob.glob(PATH_TO_TEST_IMAGES_DIR + os.sep + '*.jpg'))  # Get all the pngs in the current directory
TEST_IMAGE_PATHS = file_list
print("Test Images")
print(TEST_IMAGE_PATHS)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    img_idx = 0

    for image_path in TEST_IMAGE_PATHS:
      # Open the image file
      image = Image.open(image_path)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)

      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      print("Show Image")
      #plt.imshow(image_np)
      #plt.show()
      im = Image.fromarray(image_np)
      im.save(PATH_TO_OUTPUT_IMAGES_DIR + "/" + str(img_idx).zfill(7) + ".jpg")
      img_idx += 1
