import cv2 as cv
import numpy as np


def draw_trajectories(img, trajectories):
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for t in trajectories:
        coords = [t.coords[0]] + [x for x in t.coords for _ in range(2)] + [t.coords[-1]]
        line = np.array(coords).reshape((-1, 2, 2))
        line = np.int32(line + 0.5)
        cv.polylines(vis, line, 1, (100, 210, 100))
        cv.circle(vis, (int(coords[-1][0]), int(coords[-1][1])), 2, (40, 40, 250), -1)

    return vis


def draw_flow(img, flow, step): # QUI DISEGNA IN BLU LE RIGHE
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h:step, 0:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (255, 0, 0))

    return vis

def draw_flow_hue(img, flow, step): # QUI DISEGNA IN BLU LE RIGHE
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h:step, 0:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # Calculate flow magnitude
    magnitude = np.sqrt(fx**2 + fy**2)

    # Check for division by zero and NaN values
    min_magnitude = np.min(magnitude)
    max_magnitude = np.max(magnitude)
    if min_magnitude == max_magnitude or np.isnan(min_magnitude) or np.isnan(max_magnitude):
        return vis

    # Normalize the magnitude to range [0, 1]
    magnitude = (magnitude - min_magnitude) / (max_magnitude - min_magnitude)

    # Convert the magnitude to hue values
    max_hue = 179  # Maximum hue value
    hue_values = (magnitude * max_hue).astype(np.uint8)

    # Convert HSV to BGR
    hsv = np.zeros((1, len(hue_values), 3), dtype=np.uint8)
    hsv[..., 0] = hue_values
    hsv[..., 1] = 255
    hsv[..., 2] = 255
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)[0]

    # Draw polylines with changing colors
    for i, color in enumerate(bgr):
        cv.polylines(vis, [lines[i]], 0, (int(color[0]), int(color[1]), int(color[2])), 1)

    return vis