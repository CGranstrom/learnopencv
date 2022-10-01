import cv2
import numpy as np
import glob
import os
from utils.file_io import save_flow_image, yield_image_generator
from utils.data_viz import display_flow

def dense_optical_flow(method, images_dir, params=[], to_gray=False):
    image_paths = glob.glob(os.path.join(images_dir, '*.png')) + \
                 glob.glob(os.path.join(images_dir, '*.jpg'))
    image_paths = sorted(image_paths)
    first_frame = cv2.imread(image_paths[0])
    
    # crate HSV & make Value a constant
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255

    for idx, (first_frame, second_frame) in enumerate(yield_image_generator(image_paths)):

        frame_copy = second_frame

        # Preprocessing for exact method
        if to_gray:
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            second_frame = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Optical Flow
        flow = method(first_frame, second_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV image into BGR for demo
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        display_flow(flow, rgb)
        save_flow_image(flow, idx, "sintel/opencv")
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        