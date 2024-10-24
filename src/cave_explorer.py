#!/usr/bin/env python3

import sys
import os
import rospy
import roslib
import math
import cv2 # OpenCV2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
import tf
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
import actionlib
import random
import copy
from threading import Lock
from enum import Enum
import torch
from ultralytics import YOLO
from visualization_msgs.msg import Marker
import threading


def wrap_angle(angle):
    # Function to wrap an angle between 0 and 2*Pi
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def pose2d_to_pose(pose_2d):
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose

def get_yaw(quaternion):
    """
    Extract the yaw (rotation around the z-axis) from a quaternion.
    
    :param quaternion: A quaternion [x, y, z, w].
    :return: The yaw angle in radians.
    """
    x, y, z, w = quaternion
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    # Add more!

class CaveExplorer:
    def __init__(self):

        # Variables/Flags for perception
        self.localised_ = False
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False
        self.goal_counter_ = 0 # gives each goal sent to move_base a unique ID

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Wait for the transform to become available
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            print("Waiting for transform... Have you launched a SLAM node?")        

        # Advertise "cmd_vel" publisher to control the robot manually -- though usually we will be controller via the following action client
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # Action client for move_base
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")

        # Publisher for the camera detections
        self.image_detections_pub_ = rospy.Publisher('detections_image', Image, queue_size=1)

        # # # Read in computer vision model (simple starting point)
        # self.computer_vision_model_filename_ = rospy.get_param("~computer_vision_model_filename")
        # self.computer_vision_model_ = cv2.CascadeClassifier(self.computer_vision_model_filename_)

        self.model_path_ = rospy.get_param("~model_path", "/home/student/git/SR_A3/config/model_4.pt")
        rospy.loginfo(f"Loading model from: {self.model_path_}")
        self.model = YOLO(self.model_path_)

        # Subscribe to the camera topic
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.frame_count = 0  # Initialise frame counter
        self.detection_interval = 1  # Only run detection every n frames 

        # Subscribe to the depth image topic
        self.depth_sub_ = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1)
        # Store the latest depth image
        self.depth_image_ = None
        # Mutex for safe access to depth image
        self.depth_image_mutex = threading.Lock()

        # Publisher for the markers
        self.marker_pub_= rospy.Publisher('/artifact_markers', Marker, queue_size=10)

    def get_pose_2d(self, timestamp):  

        # Lookup the latest transform
        (trans,rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))

        # try:
        #     # Query the transform for the given timestamp
        #     (trans, rot) = self.tf_listener_.lookupTransform('map', 'base_link', timestamp)
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     # If there's an error with the timestamp, log it and use the latest available pose
        #     rospy.logwarn("TF lookup failed at timestamp. Using latest available pose.")
        #     (trans, rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))  # Fallback

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = trans[0]
        pose.y = trans[1]

        qw = rot[3]
        qz = rot[2]

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw))

        print("pose: ", pose)

        return pose
    
    
    def transform_local_3d_to_global(self, local_position_3d, pose2d):
        """
        Transforms a 3D position from the robot's local frame (base_link) to the global frame (map).
        
        :param local_position_3d: A numpy array [x, y, z] representing the 3D coordinates in the robot's local frame.
        :param pose2d: the current 2D position of the robot
        :return: A numpy array [x, y, z] representing the 3D coordinates in the global frame (map).
        """
        
        # Get the robot's current position (x, y) and orientation (theta) in the global frame (map)
        robot_x = pose2d.x
        robot_y = pose2d.y
        robot_theta = pose2d.theta  # Directly using theta from pose2d
        
        # Rotation matrix to convert from local frame to global frame using the robot's current orientation (theta)
        rotation_matrix = np.array([
            [math.cos(robot_theta), -math.sin(robot_theta), 0],
            [math.sin(robot_theta),  math.cos(robot_theta), 0],
            [0, 0, 1]
        ])

        # Apply the rotation and translation to convert the local 3D point to global coordinates
        local_position_3d = np.array(local_position_3d)
        global_position_3d = rotation_matrix.dot(local_position_3d) + np.array([robot_x, robot_y, 0])

        return global_position_3d

    def depth_callback(self, depth_msg):
         # Lock the mutex when accessing the depth image
        with self.depth_image_mutex:
            # Convert ROS depth image to OpenCV format
            self.depth_image_ = depth_msg
    
    def image_callback(self, image_msg):
        # This method is called when a new RGB image is received
        # image timestamp
        image_timestamp = image_msg.header.stamp
        # Increment frame counter
        self.frame_count += 1

        # Check if the current frame is one where detection should be performed
        if self.frame_count % self.detection_interval != 0:
            return  # Skip detection for this frame to save computation
        
        # Use this method to detect artifacts of interest
        #
        # A simple method has been provided to begin with for detecting stop signs (which is not what we're actually looking for) 
        # adapted from: https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/

        # Copy the image message to a cv image
        # see http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

         # Lock the depth image mutex and copy the depth image at the very beginning
        with self.depth_image_mutex:
            # # Ensure the latest depth image is also from the same timestamp
            # if self.depth_image is None or self.depth_image.header.stamp != image_timestamp:
            #     rospy.logwarn("Depth and RGB image timestamps do not match. Skipping this frame.")
            #     return
            depth_image_copy = self.cv_bridge_.imgmsg_to_cv2(self.depth_image_, desired_encoding="passthrough")

         
        pose2d = self.get_pose_2d(image_timestamp)  # Pass the image timestamp

        # Run YOLO inference
        results = self.model(image, conf=0.3)

        # Check if objects were detected
        if len(results) > 0 and depth_image_copy is not None:
            self.artifact_found_ = True

            # Annotate the image with bounding boxes
            annotated_image = results[0].plot()  # Annotate with bounding boxes

            for result in results:
                # Extract bounding box info
                for bbox in result.boxes.xyxy:
                    x_center = int((bbox[0] + bbox[2]) / 2)  # X center of bounding box
                    y_center = int((bbox[1] + bbox[3]) / 2)  # Y center of bounding box

                    # Get depth value at the center of the bounding box
                    depth_value = depth_image_copy[y_center, x_center]

                    # Check if the depth value is valid (not 0 and not out of range)
                    if depth_value == 0 or np.isnan(depth_value):
                        rospy.logwarn(f"No valid depth at pixel ({x_center}, {y_center}). Artifact out of range.")
                        continue  # Skip this artifact as it is out of range

                    # Convert pixel (x_center, y_center) and depth value to 3D point
                    position_3d_local = self.pixel_to_3d((x_center, y_center), depth_value)

                    # Convert the 3d local coordinate into global coordinate
                    position_3d = self.transform_local_3d_to_global(position_3d_local, pose2d)

                    # Publish the marker in RViz for visualization
                    self.publish_marker(position_3d)

                    rospy.loginfo(f"Artifact at 3D position: {position_3d}")
        else:
            self.artifact_found_ = False
            rospy.loginfo("No objects detected")
            annotated_image = image  # If no objects, return the original image


        # Publish the image with the detection bounding boxes
        image_detection_message = self.cv_bridge_.cv2_to_imgmsg(annotated_image, encoding="rgb8")
        self.image_detections_pub_.publish(image_detection_message)

        rospy.loginfo('image_callback')
        rospy.loginfo('artifact_found_: ' + str(self.artifact_found_))

    def pixel_to_3d(self, pixel, depth_value):
        """ Convert a 2D pixel and depth value into a 3D point using camera intrinsics """
        # From camera_info topic
        fx, fy =  207.8449,  207.8449  # Focal lengths in pixels 
        cx, cy = 360.5, 240.5  # Principal points in pixels
        
        x, y = pixel

        # Calculate the real-world 3D coordinates from the depth image
        z = depth_value 
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy

        return np.array([x, y, z])

    def publish_marker(self, position):
        """ Publish a marker in RViz to show the artifact location """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 0.1

        marker.color.a = 1.0  
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker.pose.orientation.w = 1.0
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        
        self.marker_pub_.publish(marker)
        rospy.loginfo("Published artifact marker")


    def planner_move_forwards(self, action_state):
        # Simply move forward by 10m

        # Only send this once before another action
        if action_state == actionlib.GoalStatus.LOST:

            pose_2d = self.get_pose_2d()

            rospy.loginfo('Current pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

            # Move forward 10m
            pose_2d.x += 10 * math.cos(pose_2d.theta)
            pose_2d.y += 10 * math.sin(pose_2d.theta)

            rospy.loginfo('Target pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            self.move_base_action_client_.send_goal(action_goal.goal)


    def planner_go_to_first_artifact(self, action_state):
        # Go to a pre-specified artifact (alien) location

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a pre-specified goal location
            pose_2d = Pose2D()
            pose_2d.x = 18.0
            pose_2d.y = 25.0
            pose_2d.theta = -math.pi/2

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            self.move_base_action_client_.send_goal(action_goal.goal)



    def planner_return_home(self, action_state):
        # Go to the origin

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a pre-specified goal location
            pose_2d = Pose2D()
            pose_2d.x = 0
            pose_2d.y = 0
            pose_2d.theta = 0

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            self.move_base_action_client_.send_goal(action_goal.goal)

    def planner_random_walk(self, action_state):
        # Go to a random location, which may be invalid

        min_x = -5
        max_x = 50
        min_y = -5
        max_y = 50

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a random location
            pose_2d = Pose2D()
            pose_2d.x = random.uniform(min_x, max_x)
            pose_2d.y = random.uniform(min_y, max_y)
            pose_2d.theta = random.uniform(0, 2*math.pi)

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            self.move_base_action_client_.send_goal(action_goal.goal)

    def planner_random_goal(self, action_state):
        # Go to a random location out of a predefined set

        # Hand picked set of goal locations
        random_goals = [[53.3,40.7],[44.4, 13.3],[2.3, 33.4],[9.9, 37.3],[3.4, 18.5],[6.0, 0.4],[28.3, 11.8],[43.7, 12.8],[38.9,43.0],[47.4,4.7],[31.5,3.2],[36.6,32.5]]

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a random location
            idx = random.randint(0,len(random_goals)-1)
            pose_2d = Pose2D()
            pose_2d.x = random_goals[idx][0]
            pose_2d.y = random_goals[idx][1]
            pose_2d.theta = random.uniform(0, 2*math.pi)

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            self.move_base_action_client_.send_goal(action_goal.goal)

    def main_loop(self):

        while not rospy.is_shutdown():

            #######################################################
            # Get the current status
            # See the possible statuses here: https://docs.ros.org/en/noetic/api/actionlib_msgs/html/msg/GoalStatus.html
            action_state = self.move_base_action_client_.get_state()
            rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
            rospy.loginfo('action_state number:' + str(action_state))

            if (self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT) and (action_state == actionlib.GoalStatus.SUCCEEDED):
                print("Successfully reached first artifact!")
                self.reached_first_artifact_ = True
            if (self.planner_type_ == PlannerType.RETURN_HOME) and (action_state == actionlib.GoalStatus.SUCCEEDED):
                print("Successfully returned home!")
                self.returned_home_ = True




            #######################################################
            # Select the next planner to execute
            # Update this logic as you see fit!
            # self.planner_type_ = PlannerType.MOVE_FORWARDS
            if not self.reached_first_artifact_:
                self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
            elif not self.returned_home_:
                self.planner_type_ = PlannerType.RETURN_HOME
            else:
                self.planner_type_ = PlannerType.RANDOM_GOAL


            #######################################################
            # Execute the planner by calling the relevant method
            # The methods send a goal to "move_base" with "self.move_base_action_client_"
            # Add your own planners here!
            print("Calling planner:", self.planner_type_.name)
            if self.planner_type_ == PlannerType.MOVE_FORWARDS:
                self.planner_move_forwards(action_state)
            elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
                self.planner_go_to_first_artifact(action_state)
            elif self.planner_type_ == PlannerType.RETURN_HOME:
                self.planner_return_home(action_state)
            elif self.planner_type_ == PlannerType.RANDOM_WALK:
                self.planner_random_walk(action_state)
            elif self.planner_type_ == PlannerType.RANDOM_GOAL:
                self.planner_random_goal(action_state)


            #######################################################
            # Delay so the loop doesn't run too fast
            rospy.sleep(0.2)



if __name__ == '__main__':

    # Create the ROS node
    rospy.init_node('cave_explorer')

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    # Loop forever while processing callbacks
    cave_explorer.main_loop()




