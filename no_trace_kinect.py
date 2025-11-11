import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# Initialize Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Infrared)

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

            # Display the frame in the OpenCV window
            cv2.imshow("Infrared Frame", processed_infrared_frame)
        
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
            break

except KeyboardInterrupt:
    print("Stopping the application.")

finally:
    # Release resources
    kinect.close()
    cv2.destroyAllWindows()
