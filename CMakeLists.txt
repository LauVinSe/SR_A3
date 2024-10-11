cmake_minimum_required(VERSION 3.1.3)

project(cave_explorer)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        image_transport
        cv_bridge
        sensor_msgs
        geometry_msgs
        nav_msgs
        tf2_ros
        std_srvs
        actionlib
        move_base_msgs
        turtlebot3_description
        gazebo_ros
    gazebo_plugins
        )
        
find_package(gazebo REQUIRED)

catkin_package()


