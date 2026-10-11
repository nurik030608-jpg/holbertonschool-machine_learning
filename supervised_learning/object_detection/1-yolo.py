#!/usr/bin/env python3
"""Module that uses the YOLO v3 algorithm to perform object detection."""
import numpy as np
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
            anchors (numpy.ndarray): Anchor boxes of shape
                (outputs, anchor_boxes, 2).
        """
        self.model = keras.models.load_model(model_path)
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process predictions from the Darknet model for a single image.

        Args:
            outputs (list of numpy.ndarray): Predictions from the model.
            image_size (numpy.ndarray): Original image size [image_height,
                image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_shape = self.model.input.shape[1:3]
        img_h, img_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_conf = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_p = 1 / (1 + np.exp(-output[..., 5:]))

            cy = np.arange(grid_h).reshape(grid_h, 1, 1, 1)
            cy = np.tile(cy, (1, grid_w, anchor_boxes, 1))
            cx = np.arange(grid_w).reshape(1, grid_w, 1, 1)
            cx = np.tile(cx, (grid_h, 1, anchor_boxes, 1))

            bx_by = 1 / (1 + np.exp(-t_xy))
            bx_by_net = bx_by + np.concatenate((cx, cy), axis=-1)
            bx_by_net = bx_by_net / np.array([grid_w, grid_h], dtype=float)

            anchor_w = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes, 1)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes, 1)

            bw_bh = np.exp(t_wh) * np.concatenate(
                (anchor_w, anchor_h), axis=-1
            )
            bw_bh = bw_bh / np.array(
                [input_shape[1], input_shape[0]], dtype=float
            )

            x1 = bx_by_net[..., 0:1] - (bw_bh[..., 0:1] / 2)
            y1 = bx_by_net[..., 1:2] - (bw_bh[..., 1:2] / 2)
            x2 = bx_by_net[..., 0:1] + (bw_bh[..., 0:1] / 2)
            y2 = bx_by_net[..., 1:2] + (bw_bh[..., 1:2] / 2)

            x1 *= img_w
            y1 *= img_h
            x2 *= img_w
            y2 *= img_h

            rescaled_boxes = np.concatenate((x1, y1, x2, y2), axis=-1)

            boxes.append(rescaled_boxes)
            box_confidences.append(box_conf)
            box_class_probs.append(box_class_p)

        return boxes, box_confidences, box_class_probs
