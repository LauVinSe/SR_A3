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
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
import actionlib
import random
import copy
from threading import Lock
from enum import Enum
import torch
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray
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

class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    FRONTIER_EXPLORATION = 6

class CaveExplorer:
    def __init__(self):

        # Variables/Flags for perception
        self.localised_ = False
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.FRONTIER_EXPLORATION  # Set to FRONTIER_EXPLORATION
        self.reached_first_artifact_ = False
        self.returned_home_ = False
        self.goal_counter_ = 0  # Unique ID for each goal sent to move_base

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Wait for the transform to become available
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            print("Waiting for transform... Have you launched a SLAM node?")

        # Publishers and action clients
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")

        # Camera detection publisher
        self.image_detections_pub_ = rospy.Publisher('detections_image', Image, queue_size=1)

        # Load YOLO model
        self.model_path_ = rospy.get_param("~model_path", "/home/student/git/SR_A3/config/model_4.pt")
        rospy.loginfo(f"Loading model from: {self.model_path_}")
        self.model = YOLO(self.model_path_)

        # Subscribers
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.depth_sub_ = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1)

        # Frame and detection interval for computational efficiency
        self.frame_count = 0  # Frame counter
        self.detection_interval = 3  # Run detection every n frames

        # Depth image and mutex for safe access
        self.depth_image_ = None
        self.depth_image_mutex = threading.Lock()

        # Marker publishers
        self.marker_pub_ = rospy.Publisher('/artifact_marker', Marker, queue_size=10)
        self.marker_array_pub_ = rospy.Publisher("/artifact_markers", MarkerArray, queue_size=10)

        # Artifact detection management
        self.detected_artifacts = []
        self.next_artifact_id = 0
        self.marker_array = MarkerArray()
        self.marker_lifetime = rospy.Duration(600.0)  # Marker cleanup time threshold (e.g., 5 minutes)

        # Define color mapping for different classes
        self.class_colors = {
            'alien': {'r': 1.0, 'g': 0.0, 'b': 0.0},         # Red
            'white_rock': {'r': 1.0, 'g': 1.0, 'b': 1.0},    # White
            'stop': {'r': 0.0, 'g': 0.0, 'b': 1.0},          # Blue
            'mushroom': {'r': 1.0, 'g': 0.0, 'b': 1.0},      # Magenta
            'green_crystal': {'r': 0.0, 'g': 1.0, 'b': 0.0}, # Green
            'white_ball': {'r': 1.0, 'g': 1.0, 'b': 0.0}     # Yellow
        }

        # Map subscriber
        self.map_sub_ = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.occupancy_grid_ = None

        # Goal commitment variables
        self.current_goal = None
        self.goal_reached_threshold = 0.5  # Distance threshold in meters

    def get_pose_2d(self):  

        # Lookup the latest transform
        (trans,rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))

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
    
    
    def transform_local_to_global(self, local_position_3d):
        """
        Transforms a 3D position from the robot's local frame (base_link) to the global frame (map).
        
        :param local_position_3d: A numpy array [x, y, z] representing the 3D coordinates in the robot's local frame.
        :return: A numpy array [x, y, z] representing the 3D coordinates in the global frame (map).
        """
        # Get the transformation from camera to world (map)
        (world_camera_trans, world_camera_rot) = self.tf_listener_.lookupTransform('map', 'camera_rgb_optical_frame', rospy.Time(0))
        
        # Convert the rotation from quaternion to a 4x4 transformation matrix
        world_camera_rot_matrix = tf.transformations.quaternion_matrix(world_camera_rot)
        
        # Set the translation in the transformation matrix
        world_camera_rot_matrix[0:3, 3] = world_camera_trans
        
        # Create a 4x1 homogeneous coordinate vector for the point in the camera frame
        local_point_camera_homogeneous = np.array([local_position_3d[0], local_position_3d[1], local_position_3d[2], 1.0])
        
        # Apply the transformation matrix to the local point in the camera frame
        global_point = np.dot(world_camera_rot_matrix, local_point_camera_homogeneous)
        
        # Return the transformed point (x, y, z)
        return global_point[0:3]

    # Callback to update the occupancy grid map
    def map_callback(self, map_msg):
        self.occupancy_grid_ = map_msg

    def depth_callback(self, depth_msg):
         # Lock the mutex when accessing the depth image
        with self.depth_image_mutex:
            # Convert ROS depth image to OpenCV format
            self.depth_image_ = depth_msg

    def image_callback(self, image_msg):
        # This method is called when a new RGB image is received
        # Increment frame counter
        self.frame_count += 1

        # Check if the current frame is one where detection should be performed
        if self.frame_count % self.detection_interval != 0:
            return  # Skip detection for this frame to save computation

        # Copy the image message to a cv image
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        # Lock the depth image mutex and copy the depth image at the very beginning
        with self.depth_image_mutex:
            depth_image_copy = self.cv_bridge_.imgmsg_to_cv2(self.depth_image_, desired_encoding="passthrough")

            # Initialize a copy of the depth image for marking centroids
            # depth_image_vis = depth_image_copy.copy()

        # Run YOLO inference
        results = self.model(image, conf=0.75)

        # Check if objects were detected
        if len(results) > 0 and depth_image_copy is not None:
            self.artifact_found_ = True
            current_time = rospy.Time.now()

            # Create a new MarkerArray for this detection cycle
            current_markers = MarkerArray()

            # Annotate the image with bounding boxes
            annotated_image = results[0].plot()  # Annotate with bounding boxes

            for result in results:
                for i, bbox in enumerate(result.boxes.xyxy):
                    x_center = int((bbox[0] + bbox[2]) / 2)
                    y_center = int((bbox[1] + bbox[3]) / 2)
                    depth_value = depth_image_copy[y_center, x_center]

                    # Get the class name for this detection
                    class_id = int(result.boxes.cls[i])
                    class_name = result.names[class_id]

                    if depth_value == 0 or np.isnan(depth_value) or depth_value > 4.5 or depth_value < 0.5:
                        rospy.logwarn(f"No valid depth at pixel ({x_center}, {y_center}). Artifact out of range.")
                        continue

                    position_3d_local = self.pixel_to_3d((x_center, y_center), depth_value)
                    position_3d = self.transform_local_to_global(position_3d_local)

                    # Check if this is a new artifact and get its ID
                    artifact_id = self.process_detection(position_3d, current_time, class_name)
                    
                    if artifact_id is not None:  # New or updated artifact
                        # Add visualization markers
                        marker = self.create_marker(position_3d, artifact_id, class_name, current_time)
                        current_markers.markers.append(marker)

            # Update and publish the marker array
            self.update_marker_array(current_markers)
        else:
            self.artifact_found_ = False
            rospy.loginfo("No objects detected")
            annotated_image = image  # If no objects, return the original image

        # # Normalize depth image for visualization
        # depth_image_vis = cv2.cvtColor(depth_image_vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # Convert to BGR format

        # # Concatenate RGB image and depth image side by side
        # combined_image = cv2.hconcat([annotated_image, depth_image_vis])

        # Publish the image with the detection bounding boxes
        image_detection_message = self.cv_bridge_.cv2_to_imgmsg(annotated_image, encoding="rgb8")
        self.image_detections_pub_.publish(image_detection_message)

        # # Display the combined image using OpenCV imshow
        # cv2.imshow("RGB and Depth Image", combined_image)
        # cv2.waitKey(1)  # Display the image for 1 ms, adjust delay as needed

        rospy.loginfo('image_callback')
        rospy.loginfo('artifact_found_: ' + str(self.artifact_found_))


    def process_detection(self, position_3d, current_time, class_name, distance_threshold=5.0):
        """
        Process a new detection and update running average of positions
        """
        position_array = np.array(position_3d)
        
        # Check for nearby existing artifacts of the same class
        for artifact in self.detected_artifacts:
            if artifact['class_name'] != class_name:
                continue
                
            distance = np.linalg.norm(position_array - np.array(artifact['position']))
            if distance < distance_threshold:
                # Update existing artifact with running average
                detection_count = artifact['detection_count']
                new_average = (
                    (np.array(artifact['position']) * detection_count + position_array) / 
                    (detection_count + 1)
                )
                
                artifact.update({
                    'position': tuple(new_average),
                    'last_seen': current_time,
                    'detection_count': detection_count + 1
                })
                
                rospy.loginfo(f"Updated {class_name} artifact {artifact['id']}")
                return artifact['id']
        
        # If no nearby artifact found, create new one
        new_artifact = {
            'id': self.next_artifact_id,
            'class_name': class_name,
            'position': position_3d,
            'first_seen': current_time,
            'last_seen': current_time,
            'detection_count': 1
        }
        self.detected_artifacts.append(new_artifact)
        self.next_artifact_id += 1
        return new_artifact['id']

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

    def create_marker(self, position, artifact_id, class_name, current_time):
        """
        Create a marker with class-specific color
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = current_time
        marker.ns = "artifacts"
        marker.id = artifact_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set marker position
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0

        # Set consistent size
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 0.3

        # Set color based on class
        color = self.class_colors.get(class_name, {'r': 0.5, 'g': 0.5, 'b': 0.5})  # Default gray if class not found
        marker.color.r = color['r']
        marker.color.g = color['g']
        marker.color.b = color['b']
        marker.color.a = 1.0

        marker.lifetime = self.marker_lifetime
        return marker

    def update_marker_array(self, current_markers):
        """
        Update the marker array and remove old markers
        """
        current_time = rospy.Time.now()
        
        # Remove old artifacts from tracking
        self.detected_artifacts = [
            artifact for artifact in self.detected_artifacts
            if (current_time - artifact['last_seen']) < self.marker_lifetime
        ]
        
        # Update marker array
        self.marker_array = current_markers
        self.marker_array_pub_.publish(self.marker_array)

    # Frontier-based exploration planner with goal commitment
    def planner_frontier_exploration(self, action_state):
        rospy.loginfo("Starting frontier-based exploration...")

        if self.occupancy_grid_ is None:
            rospy.logwarn("Occupancy grid map is not yet available.")
            return

        # If a current goal exists and is not reached, keep moving toward it
        if self.current_goal and self.get_distance_to_goal(self.current_goal) > self.goal_reached_threshold:
            rospy.loginfo('Continuing toward the current goal...')
            return  # Continue without setting a new goal

        # Process the occupancy grid to find frontiers (unexplored borders)
        grid = np.array(self.occupancy_grid_.data).reshape((self.occupancy_grid_.info.height,
                                                            self.occupancy_grid_.info.width))

        unexplored = -1
        free_space = 0
        frontiers = []
        safety_distance = 10  # Cells away from an obstacle to be considered safe

       # Precompute a safe mask to determine if a cell is within the safety distance of an obstacle
        obstacle_mask = (grid > free_space).astype(np.uint8)
        safety_mask = cv2.dilate(obstacle_mask, np.ones((2 * safety_distance + 1, 2 * safety_distance + 1), np.uint8))

        for x in range(1, grid.shape[0] - 1):
            for y in range(1, grid.shape[1] - 1):
                if grid[x, y] == unexplored and safety_mask[x, y] == 0:
                    neighbors = [grid[x + i, y + j] for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
                    if free_space in neighbors:
                        frontiers.append((x, y))


        if not frontiers:
            rospy.logwarn("No frontiers found for exploration.")
            return

        frontier_goal = random.choice(frontiers)
        goal_x = frontier_goal[1] * self.occupancy_grid_.info.resolution + self.occupancy_grid_.info.origin.position.x
        goal_y = frontier_goal[0] * self.occupancy_grid_.info.resolution + self.occupancy_grid_.info.origin.position.y

        self.current_goal = Pose2D(goal_x, goal_y, random.uniform(0, 2 * math.pi))

        action_goal = MoveBaseActionGoal()
        action_goal.goal.target_pose.header.frame_id = "map"
        action_goal.goal_id = self.goal_counter_
        self.goal_counter_ += 1
        action_goal.goal.target_pose.pose = pose2d_to_pose(self.current_goal)

        rospy.loginfo('Sending frontier goal...')
        self.move_base_action_client_.send_goal(action_goal.goal)

    # Calculate distance to the goal
    def get_distance_to_goal(self, goal):
        pose = self.get_pose_2d()
        return math.sqrt((pose.x - goal.x) ** 2 + (pose.y - goal.y) ** 2)

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
                self.planner_type_ = PlannerType.FRONTIER_EXPLORATION


            #######################################################
            # Execute the planner by calling the relevant method
            # The methods send a goal to "move_base" with "self.move_base_action_client_"
            # Add your own planners here!
            print("Calling planner:", self.planner_type_.name)
            if self.planner_type_ == PlannerType.FRONTIER_EXPLORATION:
                self.planner_frontier_exploration(action_state)


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




