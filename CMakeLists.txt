cmake_minimum_required(VERSION 3.18)
project(simple_controller_pkg LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# Compiler configuration
# -----------------------------------------------------------------------------

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Quadro P400 (Pascal GP107) = sm_61
set(CMAKE_CUDA_ARCHITECTURES 61)

# -----------------------------------------------------------------------------
# Package dependencies
# -----------------------------------------------------------------------------

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(geographic_msgs REQUIRED)
find_package(mvp_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(common_ros_dependencies
  rclcpp
  nav_msgs
  rcl_interfaces
  std_msgs
  tf2
  tf2_geometry_msgs
)

set(simple_controller_dependencies
  ${common_ros_dependencies}
  geometry_msgs
  geographic_msgs
  mvp_msgs
)

set(yaw_extractor_dependencies
  ${common_ros_dependencies}
)

set(swarm_weight_manager_dependencies
  rclcpp
  std_msgs
)

# -----------------------------------------------------------------------------
# CUDA RBF library
# -----------------------------------------------------------------------------

add_library(rbf_cuda SHARED src/rbf_cuda.cu)

target_include_directories(rbf_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Keep this PUBLIC so dependents inherit the CUDA runtime link.
target_link_libraries(rbf_cuda PUBLIC CUDA::cudart)

set_target_properties(rbf_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

# -----------------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------------

add_executable(simple_controller_node src/simple_controller.cpp)
ament_target_dependencies(
  simple_controller_node
  ${simple_controller_dependencies}
)
target_link_libraries(simple_controller_node rbf_cuda)

add_executable(yaw_extractor_node src/yaw_extractor_node.cpp)
ament_target_dependencies(
  yaw_extractor_node
  ${yaw_extractor_dependencies}
)

add_executable(swarm_weight_manager_node src/swarm_weight_manager_node.cpp)
ament_target_dependencies(
  swarm_weight_manager_node
  ${swarm_weight_manager_dependencies}
)

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------

install(
  TARGETS
    rbf_cuda
    simple_controller_node
    swarm_weight_manager_node
    yaw_extractor_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    include/
  DESTINATION include
)

install(
  DIRECTORY
    launch
    config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
