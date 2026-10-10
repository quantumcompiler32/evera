"""Distinguish the green Pi board from the blue Arduino in this camera view."""

import cv2


def refine_board_classes(frame, result):
    """Correct board labels in detector results using color inside each box.

    The threshold separates the annotated boards in this workshop scene. The
    detector still supplies each box; this only chooses between its two board
    classes after detection.
    """
    if result.names.get(0) != "raspberry_pi" or result.names.get(1) != "arduino":
        return result
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for box in result.boxes:
        if int(box.cls[0]) not in (0, 1):
            continue
        x1, y1, x2, y2 = (round(float(value)) for value in box.xyxy[0])
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            continue
        crop = hsv[y1:y2, x1:x2]
        green = cv2.inRange(crop, (30, 45, 30), (89, 255, 255))
        green_fraction = cv2.countNonZero(green) / (crop.shape[0] * crop.shape[1])
        box.cls[0] = 0 if green_fraction >= 0.025 else 1
    return result
