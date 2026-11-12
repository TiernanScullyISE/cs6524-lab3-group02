"""Keep the plug's silhouette instead of a rectangle full of background pixels."""
import cv2
import numpy as np


def object_mask(frame, config):
    height, width = frame.shape[:2]
    if 'object_roi' not in config:
        return np.full((height, width), 255, np.uint8)
    x1, y1, x2, y2 = config['object_roi']
    x1, x2 = int(x1 * width), int(x2 * width)
    y1, y2 = int(y1 * height), int(y2 * height)
    # A small border supplies definite background samples for GrabCut.
    left, top = max(0, x1 - 15), max(0, y1 - 15)
    right, bottom = min(width, x2 + 15), min(height, y2 + 15)
    crop = frame[top:bottom, left:right].copy()
    # Locate the smooth white plastic first, to keep the initial rectangle tight.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([179, 55, 255]))
    white = cv2.medianBlur(white, 9)
    _, _, areas, _ = cv2.connectedComponentsWithStats(white)
    if len(areas) > 1:
        body = 1 + np.argmax(areas[1:, cv2.CC_STAT_AREA])
        bx, by, bw, bh, area = areas[body]
        if area > 300:
            # The pins point left in the normalised landscape videos.
            x1, y1 = max(x1, left + bx - 40), max(y1, top + by - 8)
            x2, y2 = min(x2, left + bx + bw + 8), min(y2, top + by + bh + 8)
    crop = cv2.resize(crop, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    labels = np.zeros(crop.shape[:2], np.uint8)
    rectangle = ((x1 - left) // 2, (y1 - top) // 2, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.grabCut(crop, labels, rectangle, np.zeros((1, 65)), np.zeros((1, 65)),
                2, cv2.GC_INIT_WITH_RECT)
    foreground = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(foreground)
    mask = np.zeros((height, width), np.uint8)
    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        selected = (components == largest).astype(np.uint8) * 255
        mask[top:bottom, left:right] = cv2.resize(selected, (right - left, bottom - top),
                                                  interpolation=cv2.INTER_NEAREST)
    return mask
