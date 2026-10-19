cmake_minimum_required(VERSION 3.16)
project(hilti_slam_2026 VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Match the frozen H2 floating-point/optimization settings; no fast-math.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -pthread")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV 3.4 REQUIRED)
if(NOT OpenCV_VERSION VERSION_LESS 4)
  message(FATAL_ERROR "Frozen H2 requires OpenCV 3.4; see docs/BUILD.md")
endif()
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
find_package(G2O REQUIRED)
find_package(Glog REQUIRED)
find_package(CSparse REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

add_library(myslam SHARED
  src/frame.cpp
  src/mappoint.cpp
  src/map.cpp
  src/camera.cpp
  src/eucm_camera.cpp
  src/config.cpp
  src/feature.cpp
  src/frontend.cpp
  src/backend.cpp
  src/viewer.cpp
  src/hilti_dataset.cpp
  src/hilti_temporal_tracker.cpp
  src/bearing_geometry.cpp
  src/imu_preintegration.cpp
  src/nav_state.cpp
  src/vio_initializer.cpp
  src/hilti_initializer.cpp
  src/initialization/startup_visual_bootstrap.cpp
  src/initialization/visual_bootstrap_uncertainty.cpp
  src/initialization/uncertainty_metric_initializer.cpp
  src/initialization/staged_metric_initializer.cpp
  src/initialization/metric_alignment.cpp
  src/startup_visual_trajectory.cpp
  src/hilti_image_epochs.cpp
  src/hilti_frontend.cpp
  src/hilti_backend.cpp
  src/hilti_landmark_quarantine.cpp
  src/marginalization.cpp
  src/hilti_vi_linearization.cpp
  src/hilti_vi_optimizer.cpp
  src/hilti_rolling_shutter.cpp
  src/hilti_landmark_weighting.cpp
  src/hilti_local_map.cpp
  src/keyframe_database.cpp
  src/loop_geometry.cpp
  src/loop_closing.cpp
  src/pose_graph.cpp
  src/global_correction.cpp
  src/global_trajectory.cpp
  src/hilti_submission.cpp
  src/admission/snapshot.cpp
  src/admission/quality.cpp
  src/admission/camera_floor.cpp
  third_party/fbow/fbow.cpp
)
target_include_directories(myslam PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR} ${GLOG_INCLUDE_DIRS} ${CSPARSE_INCLUDE_DIR}
  PRIVATE src/initialization src/admission third_party/fbow
)
target_compile_options(myslam PRIVATE -Wall)
target_link_libraries(myslam PUBLIC Eigen3::Eigen ${OpenCV_LIBS}
  ${Sophus_LIBRARIES} ${Pangolin_LIBRARIES} ${GLOG_LIBRARIES}
  ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_TYPES_SBA}
  ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION}
  ${CSPARSE_LIBRARY} SQLite::SQLite3 Threads::Threads OpenMP::OpenMP_CXX
  OpenGL::GL OpenGL::GLU GLEW::GLEW GLUT::GLUT
)
set_source_files_properties(src/keyframe_database.cpp third_party/fbow/fbow.cpp
  PROPERTIES COMPILE_DEFINITIONS FBOW_ORB_ONLY)
# These admission functions used -O2 in the frozen H2 build.
set_source_files_properties(src/admission/quality.cpp src/admission/camera_floor.cpp
  src/admission/snapshot.cpp PROPERTIES COMPILE_OPTIONS "-O2;-pthread;-std=c++14")
foreach(program run_hilti_sequence export_hilti_submission)
  add_executable(${program} app/${program}.cpp)
  target_include_directories(${program} PRIVATE src/initialization)
  target_link_libraries(${program} PRIVATE myslam)
endforeach()

include(CTest)
if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  add_executable(test_h2 tests/horizon.cpp tests/admission.cpp)
  set_target_properties(test_h2 PROPERTIES CXX_STANDARD 14)
  target_include_directories(test_h2 PRIVATE tests src/admission)
  target_compile_definitions(test_h2 PRIVATE H2_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(test_h2 PRIVATE myslam GTest::gtest GTest::gtest_main)
  add_test(NAME h2_configuration_horizon_admission COMMAND test_h2)
  set_tests_properties(h2_configuration_horizon_admission PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;OPENCV_FOR_THREADS_NUM=1")
endif()
