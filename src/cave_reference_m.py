#!/usr/bin/env python3

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

# Function to wrap an angle between 0 and 2*Pi
def wrap_angle(angle):
    while angle < 0.0:
        angle += 2 * math.pi
    while angle > 2 * math.pi:
        angle -= 2 * math.pi
    return angle

# Function to convert Pose2D to Pose
def pose2d_to_pose(pose_2d):
    pose = Pose()
    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y
    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)
    return pose

# Find the frontier point closest to the centroid
def distance_to_point(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

# Enum class for different planner types
class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    FRONTIER_EXPLORATION = 6  # Add FRONTIER_EXPLORATION

# CaveExplorer class
class CaveExplorer:
    def __init__(self):
        # Perception flags
        self.localised_ = False
        self.artifact_found_ = False

        # Planning flags
        self.planner_type_ = PlannerType.FRONTIER_EXPLORATION  # Set to FRONTIER_EXPLORATION
        self.reached_first_artifact_ = False
        self.returned_home_ = False
        self.finished_mapping_ = False
        self.goal_counter_ = 0

        # Initialize CvBridge
        self.cv_bridge_ = CvBridge()

        # Wait for transform
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()
        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            print("Waiting for transform... Have you launched a SLAM node?")

        # Publishers
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")

        # Computer vision
        self.image_detections_pub_ = rospy.Publisher('detections_image', Image, queue_size=1)
        self.computer_vision_model_filename_ = rospy.get_param("~computer_vision_model_filename")
        self.computer_vision_model_ = cv2.CascadeClassifier(self.computer_vision_model_filename_)
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)

        # Map subscriber
        self.map_sub_ = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.occupancy_grid_ = None

        # Goal commitment variables
        self.current_goal = None
        self.goal_reached_threshold = 1.0  # Distance threshold in meters

    # Callback to update the occupancy grid map
    def map_callback(self, map_msg):
        self.occupancy_grid_ = map_msg

    # Method to get the robot's current pose in 2D
    def get_pose_2d(self):
        (trans, rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))
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

    # Image callback for processing camera images
    def image_callback(self, image_msg):
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough').copy()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stop_sign_model = self.computer_vision_model_
        detections = stop_sign_model.detectMultiScale(image, minSize=(20, 20))
        num_detections = len(detections)

        if num_detections > 0:
            self.artifact_found_ = True
        else:
            self.artifact_found_ = False

        for (x, y, width, height) in detections:
            cv2.rectangle(image, (x, y), (x + height, y + width), (0, 255, 0), 5)

        image_detection_message = self.cv_bridge_.cv2_to_imgmsg(image, encoding="rgb8")
        self.image_detections_pub_.publish(image_detection_message)

        rospy.loginfo('image_callback')
        rospy.loginfo('artifact_found_: ' + str(self.artifact_found_))

    # Cluster frontiers using connected components
    def find_connected_frontiers(self,frontiers, max_distance=5):
        clusters = []
        used = set()
        
        for point in frontiers:
            if point in used:
                continue
                
            cluster = []
            stack = [point]
            
            while stack:
                current = stack.pop()
                if current not in used:
                    used.add(current)
                    cluster.append(current)
                    
                    # Check nearby points
                    for other in frontiers:
                        if other not in used:
                            dx = abs(current[0] - other[0])
                            dy = abs(current[1] - other[1])
                            if dx <= max_distance and dy <= max_distance:
                                stack.append(other)
            
            if len(cluster) > 0:
                clusters.append(cluster)
        
        return clusters

    # Frontier-based exploration planner with goal commitment
    def planner_frontier_exploration(self, action_state):
        rospy.loginfo("Starting frontier-based exploration...")

        if self.occupancy_grid_ is None:
            rospy.logwarn("Occupancy grid map is not yet available.")
            return

        # If a current goal exists and is not reached, keep moving toward it
        if self.current_goal and self.get_distance_to_goal(self.current_goal) > self.goal_reached_threshold:
            rospy.loginfo('Continuing toward the current goal...')
            return

        # Process the occupancy grid
        grid = np.array(self.occupancy_grid_.data).reshape((self.occupancy_grid_.info.height,
                                                            self.occupancy_grid_.info.width))

        unexplored = -1
        free_space = 0
        obstacle = 100
        frontiers = []

        # Get current robot position in grid coordinates
        current_pose = self.get_pose_2d()
        robot_x = (current_pose.x - self.occupancy_grid_.info.origin.position.x) / self.occupancy_grid_.info.resolution
        robot_y = (current_pose.y - self.occupancy_grid_.info.origin.position.y) / self.occupancy_grid_.info.resolution
        robot_grid_pos = (int(robot_y), int(robot_x))  # Note: grid coordinates are (row, col)

        # Find frontier points
        for x in range(1, grid.shape[0] - 1):
            for y in range(1, grid.shape[1] - 1):
                if grid[x, y] == unexplored:
                    # Check all 8 neighboring cells
                    neighbors = [(grid[x + i, y + j], (x + i, y + j)) 
                            for i in [-1, 0, 1] 
                            for j in [-1, 0, 1] 
                            if not (i == 0 and j == 0)]
                    
                    # Check if this unexplored cell is adjacent to free space
                    if any(n[0] == free_space for n in neighbors):
                        # Check if it's not too close to obstacles
                        if not any(n[0] == obstacle for n in neighbors):
                            frontiers.append((x, y))

        if not frontiers:
            rospy.logwarn("No frontiers found for exploration.")
            return

        # Find clusters and filter by size
        frontier_clusters = self.find_connected_frontiers(frontiers)
        MIN_CLUSTER_SIZE = 1000
        large_clusters = [c for c in frontier_clusters if len(c) >= MIN_CLUSTER_SIZE]

        if not frontier_clusters:
            rospy.logwarn("No valid frontier clusters found.")
            return

        # For each large cluster, calculate its centroid and distance to robot
        cluster_info = []
        for cluster in large_clusters:
            # Calculate cluster centroid
            centroid_x = sum(point[0] for point in cluster) / len(cluster)
            centroid_y = sum(point[1] for point in cluster) / len(cluster)
            centroid = (centroid_x, centroid_y)
            
            # Calculate distance from robot to cluster centroid
            distance = distance_to_point(robot_grid_pos, centroid)
            
            cluster_info.append({
                'cluster': cluster,
                'centroid': centroid,
                'distance': distance,
                'size': len(cluster)
            })

        # Sort clusters first by size (descending) to get the top 3 largest clusters
        cluster_info.sort(key=lambda x: x['size'], reverse=True)
        best_cluster = cluster_info[:3]  # Take top 3 largest clusters

        if not best_cluster:
            rospy.logwarn("No valid frontier clusters found.")
            self.finished_mapping_ = True
            return
        
        # Among the largest clusters, select the closest one
        selected_cluster_info = min(best_cluster, key=lambda x: x['distance'])
        frontier_goal = selected_cluster_info['centroid']

        rospy.loginfo(f"Selected best cluster - Size: {selected_cluster_info['size']}, Distance: {selected_cluster_info['distance']:.2f}")

        # Convert to world coordinates
        goal_x = frontier_goal[1] * self.occupancy_grid_.info.resolution + self.occupancy_grid_.info.origin.position.x
        goal_y = frontier_goal[0] * self.occupancy_grid_.info.resolution + self.occupancy_grid_.info.origin.position.y

        # Calculate orientation towards the center of the frontier
        # This aims the robot towards the unexplored area
        dx = goal_x - current_pose.x
        dy = goal_y - current_pose.y
        goal_theta = math.atan2(dy, dx)

        # Create goal with a random orientation
        self.current_goal = Pose2D(goal_x, goal_y, random.uniform(0, 2 * math.pi))

        action_goal = MoveBaseActionGoal()
        action_goal.goal.target_pose.header.frame_id = "map"
        action_goal.goal_id = self.goal_counter_
        self.goal_counter_ += 1
        action_goal.goal.target_pose.pose = pose2d_to_pose(self.current_goal)

        rospy.loginfo('Sending frontier goal...')
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

    # Calculate distance to the goal
    def get_distance_to_goal(self, goal):
        pose = self.get_pose_2d()
        return math.sqrt((pose.x - goal.x) ** 2 + (pose.y - goal.y) ** 2)

    # Main loop for decision-making
    def main_loop(self):
        while not rospy.is_shutdown():
            action_state = self.move_base_action_client_.get_state()
            rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
            rospy.loginfo('action_state number: ' + str(action_state))

            if not self.reached_first_artifact_ and not self.finished_mapping_:
                self.planner_type_ = PlannerType.FRONTIER_EXPLORATION

            if self.planner_type_ == PlannerType.FRONTIER_EXPLORATION:
                self.planner_frontier_exploration(action_state)

            if self.finished_mapping_:
                rospy.loginfo('Finished mapping... Returning home')
                self.planner_type_ = PlannerType.RETURN_HOME
                self.planner_return_home(action_state)

            rospy.sleep(0.2)

if __name__ == '__main__':
    rospy.init_node('cave_explorer')
    cave_explorer = CaveExplorer()
    cave_explorer.main_loop()
