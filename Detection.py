# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 17:54:37 2021

@author: touQer_abaS
"""

from utils import visualization_utils as vis_util
from utils import label_map_util
import os
import cv2
import tensorflow as tf
import sys
from PIL import ImageTk, Image 

from tkinter import *
from tkinter import filedialog
#import pkg_resources.py2_warn
#import tkinter.messagebox
#import pandas as pd
#import matplotlib.pyplot as plt
#from time import strftime
#import datetime
import numpy as np

window=Tk()
window.title('Flower Identification')
#window.geometry("700x500")


class Detect():
    def __init__(self,window):
        self.window=window
        self.canvas = Canvas(self.window)
        self.canvas.pack()
    
    def browseFiles(self): 
        self.filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File",filetypes = (("all files", "*.txt*"), ("all files",  "*.*")))
        self.abc(self.filename)
    def back_button(self):
        button_explore = Button(window,text = "Browse Files", command =p.browseFiles)
        button_explore.place(x=20,y=10)
    
    def abc(self,source_image_path):
        MODEL_NAME = 'inference_graph/'
        CWD_PATH = os.getcwd()
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')
        
        # Path to image
        PATH_TO_IMAGE = source_image_path
        
        # Number of classes the object detector can identify
        NUM_CLASSES = 10
        
    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
    
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
    
            sess = tf.Session(graph=detection_graph)
    
        # Define input and output tensors (i.e. data) for the object detection classifier
    
        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name(
            'detection_classes:0')
    
        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(source_image_path)
        image2=image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
    
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
    
        # Draw the results of the detection (aka 'visulaize the results')
        #print(boxes)
    
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=7,
            min_score_thresh=0.50)

        
        scale_percent = 60 # percent of original size
        width = int(image2.shape[1] * scale_percent / 100)
        height = int(image2.shape[0] * scale_percent / 100)
        dim = (width, height)
        print(dim) 
        # resize image
        before = cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)
        
        scale_percent = 60 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        print(dim)
# resize image
        after = cv2.resize(image, (1200,600), interpolation = cv2.INTER_AREA)
        #concateImage = np.concatenate((before, after), axis=1)
        # All the results have been drawn on image. Now display the image.
        cv2.imshow('Object detector', after)
        # fname = target_image_path
        #cv2.imwrite(target_image_path, image)
        # # Press any key to close the image
        cv2.waitKey(0)
    
        # # Clean up
        cv2.destroyAllWindows()
if __name__ == "__main__":
    p=Detect(window)
    p.back_button()
    window.mainloop()

