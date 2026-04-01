# # #######################################################################

# # cmake_minimum_required(VERSION 3.18)
# # project(simple_controller_pkg LANGUAGES CXX CUDA)

# # ####################################################################
# # ## setup compiler
# # ####################################################################

# # if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
# #   add_compile_options(-Wall -Wextra -Wpedantic)
# # endif()

# # # Standards
# # set(CMAKE_CXX_STANDARD 17)
# # set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # set(CMAKE_CUDA_STANDARD 17)
# # set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# # # GPU arch: Quadro P400 (GP107, Pascal) = sm_61
# # set(CMAKE_CUDA_ARCHITECTURES 61)

# # ####################################################################
# # ## setup packages
# # ####################################################################

# # find_package(ament_cmake REQUIRED)
# # find_package(rclcpp REQUIRED)
# # find_package(geometry_msgs REQUIRED)
# # find_package(nav_msgs REQUIRED)
# # find_package(geographic_msgs REQUIRED)
# # find_package(std_msgs REQUIRED)
# # find_package(mvp_msgs REQUIRED)
# # find_package(tf2 REQUIRED)
# # find_package(tf2_geometry_msgs REQUIRED)
# # find_package(CUDAToolkit REQUIRED)


# # # CUDA library
# # add_library(rbf_cuda SHARED src/rbf_cuda.cu)

# # target_include_directories(rbf_cuda
# #   PUBLIC
# #     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# #     $<INSTALL_INTERFACE:include>)
# # target_link_libraries(rbf_cuda PRIVATE CUDA::cudart)
# # set_target_properties(rbf_cuda PROPERTIES
# #   POSITION_INDEPENDENT_CODE ON
# #   CUDA_SEPARABLE_COMPILATION ON)

# # # # Node that uses CUDA lib
# # # add_executable(data_processor_node src/data_processor_node.cpp)
# # # ament_target_dependencies(data_processor_node
# # #   rclcpp nav_msgs geometry_msgs std_msgs tf2 tf2_geometry_msgs)
# # # target_link_libraries(data_processor_node rbf_cuda)

# # # # (Optional) another node
# # add_executable(simple_controller_node src/simple_controller.cpp)
# # ament_target_dependencies(simple_controller_node
# #   rclcpp
# #   geometry_msgs
# #   nav_msgs
# #   geographic_msgs
# #   std_msgs
# #   mvp_msgs
# #   tf2
# #   tf2_geometry_msgs
# # )
# # target_link_libraries(simple_controller_node rbf_cuda)


# # ####################################################################
# # ## setup dependencies
# # ####################################################################

# # set(dependencies
# #   rclcpp
# #   geometry_msgs
# #   nav_msgs
# #   geographic_msgs
# #   std_msgs
# #   mvp_msgs
# # )

# # ####################################################################
# # ## setup compiling
# # ####################################################################

# # # add_executable(simple_controller_node src/simple_controller.cpp)
# # # # ament_target_dependencies(simple_controller_node
# # # #   ${dependencies}
# # # # )




# # ####################################################################
# # ## setup install 
# # ####################################################################


# # install(TARGETS simple_controller_node
# #   DESTINATION lib/${PROJECT_NAME})

# # # install(DIRECTORY include/
# # #   DESTINATION include/
# # # )

# # install(DIRECTORY launch 
# #   DESTINATION share/${PROJECT_NAME})

# # install(DIRECTORY config 
# #   DESTINATION share/${PROJECT_NAME})

# # install(TARGETS rbf_cuda simple_controller_node
# #   ARCHIVE DESTINATION lib
# #   LIBRARY DESTINATION lib
# #   RUNTIME DESTINATION lib/${PROJECT_NAME})


# # # install(TARGETS rbf_cuda data_processor_node simple_controller_node
# # #   ARCHIVE DESTINATION lib
# # #   LIBRARY DESTINATION lib
# # #   RUNTIME DESTINATION lib/${PROJECT_NAME})

# # install(DIRECTORY include/ DESTINATION include)

# # # if(BUILD_TESTING)
# # #   find_package(ament_lint_auto REQUIRED)
# # #   # the following line skips the linter which checks for copyrights
# # #   # comment the line when a copyright and license is added to all source files
# # #   set(ament_cmake_copyright_FOUND TRUE)
# # #   # the following line skips cpplint (only works in a git repo)
# # #   # comment the line when this package is in a git repo and when
# # #   # a copyright and license is added to all source files
# # #   set(ament_cmake_cpplint_FOUND TRUE)
# # #   ament_lint_auto_find_test_dependencies()
# # # endif()

# # ament_package()





# # # add_executable(data_processor_node src/data_processor_node.cpp)
# # # ament_target_dependencies(data_processor_node
# # #   rclcpp
# # #   nav_msgs
# # #   std_msgs
# # #   geometry_msgs
# # #   geographic_msgs
# # #   my_interfaces
# # # )
# # # install(TARGETS data_processor_node
# # #   DESTINATION lib/${PROJECT_NAME})

# # ##########################################################################################

# cmake_minimum_required(VERSION 3.18)
# project(simple_controller_pkg LANGUAGES CXX CUDA)

# # ----------------------- Compiler / Standards -----------------------
# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_ARCHITECTURES 61)  # Quadro P400 (Pascal GP107)

# # ----------------------- Dependencies -----------------------
# find_package(ament_cmake REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(geometry_msgs REQUIRED)
# find_package(nav_msgs REQUIRED)
# find_package(geographic_msgs REQUIRED)
# find_package(std_msgs REQUIRED)
# find_package(mvp_msgs REQUIRED)
# find_package(tf2 REQUIRED)
# find_package(tf2_geometry_msgs REQUIRED)
# find_package(CUDAToolkit REQUIRED)

# # ----------------------- CUDA library -----------------------
# add_library(rbf_cuda SHARED src/rbf_cuda.cu)
# target_include_directories(rbf_cuda
#   PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:include>
# )
# # target_link_libraries(rbf_cuda PRIVATE CUDA::cudart)

# # was PRIVATE; make it PUBLIC so dependents get cudart automatically
# target_link_libraries(rbf_cuda PUBLIC CUDA::cudart)

# set_target_properties(rbf_cuda PROPERTIES
#   POSITION_INDEPENDENT_CODE ON
#   CUDA_SEPARABLE_COMPILATION ON
# )

# # ----------------------- Node -----------------------
# add_executable(simple_controller_node src/simple_controller.cpp)
# ament_target_dependencies(simple_controller_node
#   rclcpp
#   geometry_msgs
#   nav_msgs
#   geographic_msgs
#   std_msgs
#   mvp_msgs
#   tf2
#   tf2_geometry_msgs
# )

# add_executable(yaw_extractor_node src/yaw_extractor_node.cpp)
# ament_target_dependencies(yaw_extractor_node
#   rclcpp
#   nav_msgs
#   std_msgs
#   tf2
#   tf2_geometry_msgs
# )

# install(TARGETS
#   yaw_extractor_node
#   DESTINATION lib/${PROJECT_NAME})


# # ----------------------- Install -----------------------
# target_link_libraries(simple_controller_node rbf_cuda)
# install(TARGETS rbf_cuda simple_controller_node
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION lib/${PROJECT_NAME}
# )
# install(DIRECTORY include/ DESTINATION include)
# install(DIRECTORY launch  DESTINATION share/${PROJECT_NAME})
# install(DIRECTORY config  DESTINATION share/${PROJECT_NAME})

# ament_package()


cmake_minimum_required(VERSION 3.18)
project(simple_controller_pkg LANGUAGES CXX CUDA)

# ----------------------- Compiler / Standards -----------------------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Quadro P400 (Pascal GP107) = sm_61
set(CMAKE_CUDA_ARCHITECTURES 61)

# ----------------------- Dependencies -----------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geographic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(mvp_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ----------------------- CUDA library (RBF) -----------------------
add_library(rbf_cuda SHARED src/rbf_cuda.cu)

target_include_directories(rbf_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link to CUDA runtime; PUBLIC so dependents inherit it
target_link_libraries(rbf_cuda PUBLIC CUDA::cudart)

set_target_properties(rbf_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

# ----------------------- Executables -----------------------

# Main controller node
add_executable(simple_controller_node src/simple_controller.cpp)
ament_target_dependencies(simple_controller_node
  rclcpp
  geometry_msgs
  nav_msgs
  geographic_msgs
  std_msgs
  mvp_msgs
  tf2
  tf2_geometry_msgs
)
target_link_libraries(simple_controller_node rbf_cuda)

# Yaw extractor helper node
add_executable(yaw_extractor_node src/yaw_extractor_node.cpp)
ament_target_dependencies(yaw_extractor_node
  rclcpp
  nav_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
)

# ----------------------- Install -----------------------
install(TARGETS
  rbf_cuda
  simple_controller_node
  yaw_extractor_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# ----------------------- Ament package -----------------------
ament_package()
