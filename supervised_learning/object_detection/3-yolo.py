#!/usr/bin/env python3
def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Perform non-max suppression on filtered bounding boxes.

        Args:
            filtered_boxes (numpy.ndarray): Filtered bounding boxes of shape
                (?, 4).
            box_classes (numpy.ndarray): Class numbers for each box of shape
                (?,).
            box_scores (numpy.ndarray): Box scores for each box of shape (?,).

        Returns:
            tuple: (box_predictions, predicted_box_classes,
                predicted_box_scores)
        """
        unique_classes = np.unique(box_classes)

        box_preds = []
        pred_classes = []
        pred_scores = []

        for c in unique_classes:
            class_mask = box_classes == c
            c_boxes = filtered_boxes[class_mask]
            c_scores = box_scores[class_mask]
            c_classes = box_classes[class_mask]

            nms_indices = tf.image.non_max_suppression(
                c_boxes,
                c_scores,
                max_output_size=100,
                iou_threshold=self.nms_t
            ).numpy()

            box_preds.append(c_boxes[nms_indices])
            pred_classes.append(c_classes[nms_indices])
            pred_scores.append(c_scores[nms_indices])

        if box_preds:
            box_predictions = np.concatenate(box_preds, axis=0)
            predicted_box_classes = np.concatenate(pred_classes, axis=0)
            predicted_box_scores = np.concatenate(pred_scores, axis=0)
        else:
            box_predictions = np.array([])
            predicted_box_classes = np.array([])
            predicted_box_scores = np.array([])

        # Sort all aggregated predictions globally by box score descending
        sort_indices = np.argsort(predicted_box_scores)[::-1]
        box_predictions = box_predictions[sort_indices]
        predicted_box_classes = predicted_box_classes[sort_indices]
        predicted_box_scores = predicted_box_scores[sort_indices]

        return box_predictions, predicted_box_classes, predicted_box_scores
