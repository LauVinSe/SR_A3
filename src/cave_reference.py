#!/usr/bin/env python3

import rospy
import roslib
import math
import cv2  # OpenCV2
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


def wrap_angle(angle):
    while angle < 0.0:
        angle = angle + 2 * math.pi
    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi
    return angle


def pose2d_to_pose(pose_2d):
    pose = Pose()
    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y
    pose.orientation.w = math.cos(pose_2d.theta)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)
    return pose


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    FRONTIER_EXPLORE = 6  # New Planner for frontier-based exploration


class CaveExplorer:
    def __init__(self):
        self.localised_ = False
        self.artifact_found_ = False
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False
        self.goal_counter_ = 0  # Unique goal ID for move_base

        # Initialize CvBridge
        self.cv_bridge_ = CvBridge()

        # Waiting for the transform
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            print("Waiting for transform... Have you launched a SLAM node?")

        # Publisher to control robot manually
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # Move_base action client
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")

        # Publisher for camera detections
        self.image_detections_pub_ = rospy.Publisher('detections_image', Image, queue_size=1)

        # Read computer vision model
        self.computer_vision_model_filename_ = rospy.get_param("~computer_vision_model_filename")
        self.computer_vision_model_ = cv2.CascadeClassifier(self.computer_vision_model_filename_)

        # Subscribe to camera topic
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)

        # Get Map
        self.map_ = rospy.wait_for_message("/map", OccupancyGrid)

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

    def image_callback(self, image_msg):
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
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

    def planner_move_forwards(self, action_state):
        if action_state == actionlib.GoalStatus.LOST:
            pose_2d = self.get_pose_2d()
            rospy.loginfo('Current pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

            pose_2d.x += 10 * math.cos(pose_2d.theta)
            pose_2d.y += 10 * math.sin(pose_2d.theta)

            rospy.loginfo('Target pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ += 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            self.move_base_action_client_.send_goal(action_goal.goal)

    def frontier_explore(self, action_state):
        if action_state != actionlib.GoalStatus.ACTIVE:
            unexplored_frontiers = self.get_frontiers()

            if unexplored_frontiers:
                chosen_goal = random.choice(unexplored_frontiers)

                action_goal = MoveBaseActionGoal()
                action_goal.goal.target_pose.header.frame_id = "map"
                action_goal.goal_id = self.goal_counter_
                self.goal_counter_ += 1
                action_goal.goal.target_pose.pose = pose2d_to_pose(chosen_goal)

                rospy.loginfo('Exploring frontier, sending goal...')
                self.move_base_action_client_.send_goal(action_goal.goal)

    def get_frontiers(self):
        # Simple frontier detection logic - find unknown regions in the map
        frontiers = []
        width = self.map_.info.width
        height = self.map_.info.height
        resolution = self.map_.info.resolution
        data = np.array(self.map_.data).reshape((height, width))

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if data[y, x] == -1:
                    if (0 <= data[y - 1, x] <= 50 or 0 <= data[y + 1, x] <= 50 or
                            0 <= data[y, x - 1] <= 50 or 0 <= data[y, x + 1] <= 50):
                        goal_pose = Pose2D()
                        goal_pose.x = x * resolution + self.map_.info.origin.position.x
                        goal_pose.y = y * resolution + self.map_.info.origin.position.y
                        goal_pose.theta = random.uniform(0, 2 * math.pi)
                        frontiers.append(goal_pose)

        rospy.loginfo(f"Frontiers found: {len(frontiers)}")
        return frontiers

    def main_loop(self):
        while not rospy.is_shutdown():
            action_state = self.move_base_action_client_.get_state()
            rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
            rospy.loginfo('action_state number:' + str(action_state))

            if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT and action_state == actionlib.GoalStatus.SUCCEEDED:
                print("Successfully reached first artifact!")
                self.reached_first_artifact_ = True
            if self.planner_type_ == PlannerType.RETURN_HOME and action_state == actionlib.GoalStatus.SUCCEEDED:
                print("Successfully returned home!")
                self.returned_home_ = True

            if not self.reached_first_artifact_:
                self.planner_type_ = PlannerType.FRONTIER_EXPLORE
            elif not self.returned_home_:
                self.planner_type_ = PlannerType.RETURN_HOME
            else:
                self.planner_type_ = PlannerType.RANDOM_GOAL

            print("Calling planner:", self.planner_type_.name)
            if self.planner_type_ == PlannerType.MOVE_FORWARDS:
                self.planner_move_forwards(action_state)
            elif self.planner_type_ == PlannerType.FRONTIER_EXPLORE:
                self.frontier_explore(action_state)
            rospy.sleep(1)


if __name__ == '__main__':
    rospy.init_node('cave_explorer')
    explorer = CaveExplorer()
    explorer.main_loop()
