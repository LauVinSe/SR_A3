# Space Robotics - Cave Explorer

## Prerequisites

Before launching the project, ensure you have the following installed:

- **ROS Noetic**: Follow instructions to install ROS Noetic [here](http://wiki.ros.org/noetic/Installation/Ubuntu).
- **TurtleBot3 Packages**: These are required for simulation.
- **TEB Local Planner**: Install with the following command:
  
  ```bash
  sudo apt-get install ros-noetic-teb-local-planner
  ```
## Installation

### Cloning the Repository and Switching Branches

#### 1. Clone the Repository

To clone the repository to your local machine, follow these steps:

1. Open your terminal or command line.
2. Run the following command to clone the repository:
   ```bash
   git clone git@github.com:LauVinSe/SR_A3.git
   ```
   or
   
   ```bash
   git clone https://github.com/LauVinSe/SR_A3.git
   ```
   
4. After cloning, navigate into the repository folder:
   
    ```bash
    cd project-name
    ```
    
### 2. List Available Branches

To see all the available branches in the repository, run:

  ```bash
  git branch -a
  ```

### 3. Switch to a Different Branch

To switch to a specific branch, use the following command:

  ```bash
  git checkout <branch_name>
  ```

### 4. Pull the Latest Changes (Optional)

Once you've switched to a branch, it’s a good idea to pull the latest changes from the remote repository to ensure you're working with the most up-to-date code:

  ```bash
  git pull origin <branch_name>
   ```

### Set Up the Workspace: 

After cloning the repository, navigate to your ROS workspace and build the project:
```bash
  cd ~/catkin_ws/
  catkin build
  source ~/catkin_ws/devel/setup.bash
```

## Running the project

As written in the project documentation:

### Step 1: Start the Simulation

In the first terminal, run the simulation environment using the following command:

```bash
  roslaunch cave_explorer cave_explorer_startup.launch
```

This will start the Gazebo simulator containing the Mars rover in a cave-like environment, along with RVIZ for visualization.

### Step 2: Launch Navigation

In the second terminal, run the navigation system:

```bash
roslaunch cave_explorer cave_explorer_navigation.launch
```

This will enable the rover to navigate using SLAM and basic navigation.

### Step 3: Start Autonomy

In the third terminal, run the autonomy node:

```bash
roslaunch cave_explorer cave_explorer_autonomy.launch
```

This node runs the main Python code (cave_explorer.py) that controls the rover's autonomous decision-making and artifact detection.

### Note:

The first time you run the autonomy node, you may encounter a permission error. Fix it by running:

```bash
chmod u+x ~/catkin_ws/src/cave_explorer/src/cave_explorer.py\
```

## How to Control the Rover
- ### Manual Control:
  You can manually send goals to the rover by clicking the 2D Nav Goal button in RVIZ and selecting a location on the map.

- ### Autonomous Mode:
  The rover will autonomously explore the cave and search for artifacts based on the code in cave_explorer.py. The code includes basic decision-making and computer vision functionalities.
