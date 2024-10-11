#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
from ultralytics import YOLO
import numpy as np

class YOLODetectorNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('yolo_detector_node', anonymous=True)
        
        # Load YOLOv8 model (or use your own trained weights)
        self.model = YOLO("/home/vincent/git/cave_explorer/config/model_1.pt")  # Replace with "best.pt" for your custom weights
        
        # CVBridge for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the camera topic
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

        # Publisher to publish detection results as images
        self.detection_pub = rospy.Publisher('/yolo_detections', Image, queue_size=10)

        rospy.loginfo("YOLO Object Detection Node Initialized")

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Perform YOLO inference
        results = self.model(cv_image)

        # Draw bounding boxes on the image
        annotated_image = results[0].plot()

        try:
            # Convert annotated OpenCV image back to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Publish the detection results
        self.detection_pub.publish(ros_image)

    def run(self):
        # Keep the node running
        rospy.spin()

if __name__ == '__main__':
    try:
        yolo_node = YOLODetectorNode()
        yolo_node.run()
    except rospy.ROSInterruptException:
        pass
