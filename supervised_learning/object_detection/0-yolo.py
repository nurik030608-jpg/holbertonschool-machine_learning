#!/usr/bin/env python3
"""Module that uses the YOLO v3 algorithm to perform object detection."""
import tensorflow.keras as keras


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo instance.

        Args:
            model_path (str): Path to the Darknet Keras model.
            classes_path (str): Path to the list of class names.
            class_t (float): Box score threshold for initial filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Anchor boxes of shape (outputs, anchor_boxes, 2).
        """
        self.model = keras.models.load_model(model_path)
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
