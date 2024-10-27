#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
from datetime import datetime
import time

class ImageSaver:
    def __init__(self):
        # Initialize the node
        rospy.init_node('image_saver', anonymous=True)
        
        # Create a CV bridge
        self.bridge = CvBridge()
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.expanduser('~/saved_images')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Initialize counter for saved images
        self.count = 0
        
        # Rate limiting parameters (in seconds)
        self.save_interval = rospy.get_param('~save_interval', 2.0)  # Default to 2 seconds
        self.last_save_time = 0
        
        # Create the subscriber
        # Change 'image_topic' to your actual topic name (e.g., '/camera/image_raw')
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.callback)
        
        rospy.loginfo("Image saver node initialized. Saving images to: %s", self.output_dir)
        rospy.loginfo("Saving images every %.1f seconds", self.save_interval)
        
    def should_save_image(self):
        """Check if enough time has passed to save a new image"""
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.last_save_time = current_time
            return True
        return False
        
    def callback(self, data):
        # Check if we should save this image based on the time interval
        if not self.should_save_image():
            return
            
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CV Bridge error: %s", e)
            return
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}_{self.count}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the image
        try:
            cv2.imwrite(filepath, cv_image)
            rospy.loginfo("Saved image to: %s", filepath)
            self.count += 1
        except Exception as e:
            rospy.logerr("Error saving image: %s", e)

def main():
    try:
        image_saver = ImageSaver()
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down image saver node")
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()