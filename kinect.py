import os
import cv2
import numpy as np
from pykinect2024 import PyKinect2024 as PyKinectV2
from pykinect2024 import PyKinectRuntime
from collections import deque
from datetime import datetime
import light
from cnn import cnn

# Initialize Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Infrared)

# Global variables for tracing frame saving
trace_image_count = 0
save_tracing_frames = False
os.chdir('tracing_frames')

# Function to process the infrared frame
def process_infrared_frame(infrared_frame):
    # Reshape the frame to match Kinect's resolution
    frame = np.reshape(infrared_frame, (kinect.infrared_frame_desc.Height, kinect.infrared_frame_desc.Width))
    # Normalize the frame values to 8-bit for display
    frame = np.uint8(frame / 256)  # Convert from 16-bit to 8-bit
    return frame

# Function to process the color frame
def process_color_frame(color_frame):
    # Reshape the color frame to (1080, 1920, 4) since it includes an alpha channel
    frame = np.reshape(color_frame, (kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))
    # Convert from BGRA (default format) to BGR for OpenCV
    frame = frame[:, :, :3]  # Remove alpha channel
    return frame

def calculate_keypoint_center(keypoint):
    """Calculate the center of a contour."""
    moments = cv2.moments(keypoint)
    if moments['m00'] == 0:
        moments['m00'] = 1                
    # Calculate the centroid of the contour
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx, cy

class WandProcessor:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tracing_frame = np.zeros((frame_height, frame_width), dtype=np.uint8)
        self.camera_frame = np.zeros((frame_height, frame_width), dtype=np.uint8)
        self.fg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100)
        self.keypoints = deque(maxlen=1000000)
        self.current_time = None
        self.last_keypoint_time = None
        self.max_trace_distance = 20
        self.minimum_area = 5
        self.time_till_clear = 5
        # self.detector = cv2.SimpleBlobDetector_create(self._blob_detector_params())

    # def _blob_detector_params(self):
    #     params = cv2.SimpleBlobDetector_Params()
    #     params.minThreshold = 100
    #     params.maxThreshold = 255
    #     params.filterByColor = True
    #     params.blobColor = 255
    #     params.filterByArea = True
    #     params.minArea = 5
    #     params.maxArea = 100000
    #     params.filterByCircularity = False
    #     params.minCircularity = 0.5
    #     return params

    def process_frame(self, frame):
        """Process a single frame for wand detection and trace updating."""
        # Ensure the tracing frame matches the size of the input frame
        if self.tracing_frame.shape != frame.shape:
            self.tracing_frame = np.zeros_like(frame)
        
        # Perform background subtraction
        fg_mask = self.fg_subtractor.apply(frame)
        bg_subtracted = cv2.bitwise_and(frame, frame, mask=fg_mask)

        # cv2.imshow("bg_subtracted", bg_subtracted)

        # Perform blob detection
        # keypoints = self.detector.detect(bg_subtracted)
        
        # Threshold the frame to only detect the wand
        bg_subtracted_threshold = cv2.threshold(bg_subtracted, 200, 255, cv2.THRESH_BINARY)[1]

        keypoints,heir= cv2.findContours(bg_subtracted_threshold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        # Order keypoints by size and largest first
        keypoints = sorted(keypoints, key=lambda x: cv2.contourArea(x), reverse=True)
        # Remove any keypoints that are too small
        keypoints = [kp for kp in keypoints if cv2.contourArea(kp) > self.minimum_area]

        # Update the trace with detected keypoints
        self.update_trace(keypoints)
        
        # Display the key points
        keypoint_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for kp in keypoints:
            cv2.circle(keypoint_frame, (int(calculate_keypoint_center(kp)[0]), int(calculate_keypoint_center(kp)[1])), 5, (0, 0, 255), 2)
        cv2.imshow("Key Points", keypoint_frame)

        # Combine the trace with the original frame
        # combined_frame = cv2.add(frame, self.tracing_frame)
        trace_frame = self.tracing_frame.copy()

        return trace_frame

    def update_trace(self, keypoints):
        """Update trace based on detected keypoints."""
        if keypoints:
            self.current_time = datetime.now()
            if self.last_keypoint_time:
                time_diff = (self.current_time - self.last_keypoint_time).total_seconds()
                if time_diff < 0.5:
                    pt1 = tuple(map(int, calculate_keypoint_center(self.keypoints[-1])))
                    pt2 = tuple(map(int, calculate_keypoint_center(keypoints[0])))
                    # If pt1 and pt2 are certain distance apart, don't draw a line
                    if not (abs(pt1[0] - pt2[0]) > self.max_trace_distance or abs(pt1[1] - pt2[1]) > self.max_trace_distance):
                        cv2.line(self.tracing_frame, pt1, pt2, 255, 2)
            self.last_keypoint_time = self.current_time
            self.keypoints.append(keypoints[0])
        # Erase the trace if after certain amount of seconds of no key points
        else:
            self.current_time = datetime.now()
            if self.last_keypoint_time:
                time_diff = (self.current_time - self.last_keypoint_time).total_seconds()
                if time_diff > self.time_till_clear:
                    global save_tracing_frames
                    save_tracing_frames = True

    # def is_trace_valid(self):
    #     """Check if the current trace is valid."""
    #     if len(self.keypoints) < 35:
    #         return False
    #     area = self._calculate_trace_area()
    #     return area > 500

    # def _calculate_trace_area(self):
    #     """Calculate the area of the bounding box around the trace."""
    #     points = [kp.pt for kp in self.keypoints]
    #     x_coords, y_coords = zip(*points)
    #     width = max(x_coords) - min(x_coords)
    #     height = max(y_coords) - min(y_coords)
    #     return width * height

    def clear_trace(self):
        """Clear the trace frame of all keypoints and their data."""
        self.tracing_frame = np.zeros_like(self.tracing_frame)
        self.keypoints.clear()
        self.last_keypoint_time = None


def main():
    processor = WandProcessor(640, 480)
    
    # OpenCV window setup
    cv2.namedWindow("Infrared Frame", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Color Frame", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Flags to track if the windows are resized
    infrared_window_resized = False
    color_window_resized = False

    print("Press 'esc' to exit.")
    try:
        while True:
            # Check if there is a new infrared frame
            if kinect.has_new_infrared_frame():
                
                # Get the infrared frame
                infrared_frame = kinect.get_last_infrared_frame()
                # Process the frame
                processed_infrared_frame = process_infrared_frame(infrared_frame)

                # Resize the infrared window on the first frame
                if not infrared_window_resized:
                    cv2.resizeWindow("Infrared Frame", processed_infrared_frame.shape[1], processed_infrared_frame.shape[0])
                    infrared_window_resized = True

                # Recieve the frame of the trace line made by the wand processor
                trace_frame = processor.process_frame(processed_infrared_frame)

                # Combine the trace with the original infared frame
                combined_frame = cv2.add(processed_infrared_frame, trace_frame)

                # Display the combined frame in the OpenCV window
                cv2.imshow("Infrared Frame", combined_frame)

                # Save the old tracing frame for debugging, if one already exists trace_image_count up the number
                global trace_image_count
                global save_tracing_frames
                if save_tracing_frames:
                    while os.path.exists(f"tracing_frame_{trace_image_count}.png"):
                        trace_image_count += 1
                    cv2.imwrite(f"tracing_frame_{trace_image_count}.png", trace_frame)
                    save_tracing_frames = False
                    processor.clear_trace()
                    # light.toggle_all_bulbs()
                    cnn.run_cnn(trace_frame)

            # Check if there is a new color frame
            if kinect.has_new_color_frame():
                
                # Get the color frame
                color_frame = kinect.get_last_color_frame()
                # Process the frame
                processed_color_frame = process_color_frame(color_frame)

                # Resize the color window on the first frame
                if not color_window_resized:
                    cv2.resizeWindow("Color Frame", int(processed_color_frame.shape[1]/2), int(processed_color_frame.shape[0]/2))
                    color_window_resized = True

                # Display the frame in the OpenCV window
                cv2.imshow("Color Frame", processed_color_frame)
            
            # Break the loop if 'esc' is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("Stopping the application.")

    finally:
        # Release resources
        kinect.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()