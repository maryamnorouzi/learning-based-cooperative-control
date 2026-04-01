
#include <array>
#include <cmath>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <limits>
#include <memory>
#include <string>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <cstdint> 
#include <cstdio>
#include <algorithm>
#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>
#include "rbf_cuda.hpp"

#include <std_msgs/msg/float32_multi_array.hpp>

using std::placeholders::_1;

// ============================================================================
// World and state definitions
// ============================================================================
// World frame: mauv_1/world = ENU (East, North, Up)
// X: +X = East, −X = West  (Surge) --> Roll
// Y: +Y = North, −Y = South (Sway) --> Yaw
// Z: +Z = Up, −Z = Down    (Heave) --> Pitch

struct FullState { 
  rclcpp::Time stamp;
  std::string frame_id;
  std::string child_frame_id;
  geometry_msgs::msg::Point position;          // x y z
  geometry_msgs::msg::Quaternion orientation;  // qx qy qz qw
  geometry_msgs::msg::Vector3 lin_vel;         // vx vy vz (in child frame: base_link)
  geometry_msgs::msg::Vector3 ang_vel;         // wx wy wz (in child frame: base_link)

  // *********************************************************************************
  // We use a struct here because it’s a simple data bundle.
  // Struct is like a class with public members.
  // It holds pieces of data together so you can pass them around as one thing.
  // **********************************************************************************
  };

// DOF indices (world)
enum { X=0, Y=1, Z=2, ROLL=3, PITCH=4, YAW=5 };

// Waypoints (leader state): q = [X, Y, Z, Pitch]
enum { 
  WX     = 0,   // X position
  WY     = 1,   // Y position  (used in waypoint generator, LOS, logging)
  WYAW   = 1,   // alias for index 1 when it represents YAW in controller vectors
  WZ     = 2,   // Z position
  WPITCH = 3    // Pitch angle
};

// Controller 4-D vectors:
// index 0 -> X : controlled by SURGE_T
// index 1 -> Y,  equivalent to YAW : controlled by SWAY_BOW_T
// index 2 -> Z : controlled by HEAVE_BOW_T
// index 3 -> PITCH : controlled by HEAVE_STERN_T

// Thrusters
enum { SURGE_T = 0, HEAVE_BOW_T = 1, HEAVE_STERN_T = 2, SWAY_BOW_T = 3 };

constexpr double PI = 3.14159265358979323846;

// ============================================================================
// Node definition
// ============================================================================

class DataProcessorNode : public rclcpp::Node
{
public:
  DataProcessorNode() : Node("data_processor_node")
  {
    setup_tf_();
    setup_params_();
    setup_io_();
    setup_math_();
    start_worker_();
  }

  ~DataProcessorNode() override {
    if (processing_thread_.joinable()) {
      processing_thread_.join();
    }
  }

private:
  // --------------------------------------------------------------------------
  // TF buffer and listener
  // --------------------------------------------------------------------------
  void setup_tf_() { 
    tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  }

  // --------------------------------------------------------------------------
  // Rotation matrix for Angular velocity
  // --------------------------------------------------------------------------
  Eigen::Matrix3d f_angular_velocity_transform(const geometry_msgs::msg::TransformStamped& tf) 
  {
      tf2::Quaternion quat;
      quat.setW(tf.transform.rotation.w);
      quat.setX(tf.transform.rotation.x);
      quat.setY(tf.transform.rotation.y);
      quat.setZ(tf.transform.rotation.z);

      Eigen::Vector3d orientation;
      tf2::Matrix3x3(quat).getRPY(orientation.x(), orientation.y(), orientation.z());

      Eigen::Matrix3d transform = Eigen::Matrix3d::Zero();

      double cosy = cos(orientation.y());
      double tany = tan(orientation.y());

      if(cosy >-0.0001 && cosy <0.0001){
          cosy = 0.0001;
      }

      tany = std::min(std::max(tany, -1000.0), 1000.0);

      transform(0,0) = 1.0;
      transform(0,1) = sin(orientation.x()) * tany;
      transform(0,2) = cos(orientation.x()) * tany;
      transform(1,0) = 0.0;
      transform(1,1) = cos(orientation.x());
      transform(1,2) = -sin(orientation.x());
      transform(2,0) = 0.0;
      transform(2,1) = sin(orientation.x()) / cosy;
      transform(2,2) = cos(orientation.x()) / cosy;

      return transform;
  }

    // Eigen::Matrix3d f_angular_velocity_transform(const geometry_msgs::msg::TransformStamped& tf)
    // {
    //   tf2::Quaternion quat;
    //   quat.setW(tf.transform.rotation.w);
    //   quat.setX(tf.transform.rotation.x);
    //   quat.setY(tf.transform.rotation.y);
    //   quat.setZ(tf.transform.rotation.z);

    //   double roll, pitch, yaw;
    //   tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);

    //   Eigen::Matrix3d transform = Eigen::Matrix3d::Zero();

    //   double cosy = std::cos(pitch);
    //   double tany = std::tan(pitch);

    //   if (cosy > -0.0001 && cosy < 0.0001) cosy = 0.0001;
    //   tany = std::min(std::max(tany, -1000.0), 1000.0);

    //   transform(0,0) = 1.0;
    //   transform(0,1) = std::sin(roll) * tany;
    //   transform(0,2) = std::cos(roll) * tany;

    //   transform(1,0) = 0.0;
    //   transform(1,1) = std::cos(roll);
    //   transform(1,2) = -std::sin(roll);

    //   transform(2,0) = 0.0;
    //   transform(2,1) = std::sin(roll) / cosy;
    //   transform(2,2) = std::cos(roll) / cosy;

    //   return transform;
    // }

  // --------------------------------------------------------------------------
  // Parameters
  // --------------------------------------------------------------------------
  void setup_params_() {
    global_frame_ = this->declare_parameter<std::string>("global_frame", ""); 

    // RBF / grid params
    zetta_ne_ = this->declare_parameter<int>("zetta_ne", 7);
    lambda_   = this->declare_parameter<double>("lambda", 0.2);

    // // Freeze-learning / init weights
    // wbar_bin_path_ = this->declare_parameter<std::string>("wbar_bin_path","/home/soslab-p330/ros2_ws/wbar.bin");

    // Scaling values
    scale_pos_    = this->declare_parameter<double>("scale_pos",    scale_pos_);
    scale_ang_    = this->declare_parameter<double>("scale_ang",    scale_ang_);
    scale_lin_    = this->declare_parameter<double>("scale_lin",    scale_lin_);
    scale_angvel_ = this->declare_parameter<double>("scale_angvel", scale_angvel_);
    scale_tau_    = this->declare_parameter<double>("scale_tau",    scale_tau_);

    auto lo = this->declare_parameter<std::vector<double>>(
         "rbf_lo8", std::vector<double>(8, -1.0));
    auto hi = this->declare_parameter<std::vector<double>>(
         "rbf_hi8", std::vector<double>(8,  1.0));

    // safety check
    if (lo.size() != 8 || hi.size() != 8) {
      // Print the error
      RCLCPP_ERROR(get_logger(),
        "rbf_lo8 and rbf_hi8 must each have exactly 8 elements. Got lo=%zu hi=%zu. Using defaults.",
        lo.size(), hi.size()); 

        // Resets both to safe defaults
        lo.assign(8, -1.0);
        hi.assign(8,  1.0);
    }

    for (int i = 0; i < 8; ++i) {
      rbf_lo8_[i] = static_cast<float>(lo[i]);
      rbf_hi8_[i] = static_cast<float>(hi[i]);
    }

    // ****************************************************************************
    // Frames: we parametrize the global frame, but auto-detect odom from messages.
    // global_frame_ ~ "mauv_1/world"
    // ****************************************************************************
  }
  
  // --------------------------------------------------------------------------
  // Input / output (ROS interfaces)
  // --------------------------------------------------------------------------
  void setup_io_() {
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odometry/filtered", 
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
      std::bind(&DataProcessorNode::odom_callback, this, _1)); 

    // Publishers for P_world, p_ , and weights
    pub_world_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("world_odom", 10);
    pub_wp_odom_    = this->create_publisher<nav_msgs::msg::Odometry>("waypoint_odom", 10);
    pub_z1_odom_    = this->create_publisher<nav_msgs::msg::Odometry>("z1_odom", 10);
    pub_pi_p_       = this->create_publisher<nav_msgs::msg::Odometry>("pi_p", 10);
    pub_W_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("rbf_weights", 10);
    pub_w_norms_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("rbf_weight_norms", 10);

    // ************************************************************
    // Note: "odometry/filtered" is a topic name, not a frame name.
    // ************************************************************
  }

  // --------------------------------------------------------------------------
  // Math and controller setup
  // --------------------------------------------------------------------------
  void setup_math_() {
    // Waypoint dynamics (leader) in [X, Y, Z, Pitch] 
    // Ensure clean zeros first
    A10_.fill(0.0);
    A0_.fill(0.0);
    B0_.fill(0.0);

    // A10_  
    A10_[WX*4 + WX]         = 1.0;
    A10_[WY*4 + WY]         = 1.0;
    A10_[WZ*4 + WZ]         = 1.0;
    A10_[WPITCH*4 + WPITCH] = 1.0;

    // A0_ 
    A0_[WZ*4 + WZ]         = -kd_z;       
    A0_[WPITCH*4 + WPITCH] = -zeta_pitch;

    // B0_ 
    B0_[WX*4 + WX]         = -wx*wx;         // X'' = -wx^2 X
    B0_[WY*4 + WY]         = -wy*wy;         // Y'' = -wy^2 Y
    B0_[WZ*4 + WZ]         = -kp_z;          // Z'' += -kp_z * Z
    B0_[WPITCH*4 + WPITCH] = -w_pitch;       // Pitch'' += -w_pitch * Pitch

    // Controller / model matrices & gains 
    H1_.fill(0.0);
    H2_.fill(0.0);

    // H1: position → virtual velocity (acts like a position gain inside z2)
    H1_[WX*4 + WX]         = 1.0; //0.5
    H1_[WYAW *4 + WYAW ]   = 1.0;   
    H1_[WZ*4 + WZ]         = 1.0;   
    H1_[WPITCH*4 + WPITCH] = 1.0;    

    // H2: velocity error → τ  (damping + position gain)
    H2_[WX*4 + WX]         = 200.0; //20.0
    H2_[WYAW *4 + WYAW ]   = 200.0;  
    H2_[WZ*4 + WZ]         = 300.0;  
    H2_[WPITCH*4 + WPITCH] = 100.0;

    // Initial conditions for the waypoint generator
    // circle
    // const std::array<double,4> q0  = { 0.0, 10, z_ref, 0.0}; //z_ref
    // const std::array<double,4> qd0 = { 0.1, 0.0, 0.0, 0.0 }; 

    // // Number "8" trajectory
    const double A = 100.0;   // X amplitude (meters)
    const double B = 50.0;    // Y amplitude (meters)
    const std::array<double,4> q0  = { 0.0, 0.0, z_ref, 0.0 };   // start at origin in XY
    const std::array<double,4> qd0 = { A*wx, B*wy, 0.0, 0.0};      

    // Waypoint states 
    for (int i=0; i<4; ++i) {
      X_waypoint_[i]    = q0[i];
      X_waypoint_[4+i]  = qd0[i];
      p_[i] = X_waypoint_[i];   
      v_[i] = X_waypoint_[4 + i];
    }

    // GPU helper (RBF)
    // rbf_ = std::make_unique<CudaRBF>(zetta_ne_, rbf_lo_, rbf_hi_, (float)lambda_);
    float lo8[8], hi8[8];
    for (int i = 0; i < 8; ++i) {
      lo8[i] = rbf_lo8_[i];
      hi8[i] = rbf_hi8_[i];
    } 

    // Create RBF
    rbf_ = std::make_unique<CudaRBF>(zetta_ne_, lo8, hi8, (float)lambda_);

    // // ---- ALWAYS load W from .bin (quick test) ----
    // if (wbar_bin_path_.empty()) {
    //   RCLCPP_WARN(get_logger(), "wbar_bin_path is empty -> using default-initialized weights");
    // } else {
    //   const uint64_t N = rbf_->num_points();
    //   const size_t expected_bytes = size_t(4) * size_t(N) * sizeof(float);

    //   // optional sanity check
    //   try {
    //     auto fs = std::filesystem::file_size(wbar_bin_path_);
    //     if (fs != expected_bytes) {
    //       RCLCPP_ERROR(get_logger(),
    //         "wbar file size mismatch: got %llu bytes, expected %zu bytes (4*N floats).",
    //         (unsigned long long)fs, expected_bytes);
    //     }
    //   } catch (...) {
    //     RCLCPP_WARN(get_logger(), "Could not read file size for %s", wbar_bin_path_.c_str());
    //   }

    //   std::vector<float> W(4 * N);
    //   std::ifstream f(wbar_bin_path_, std::ios::binary);
    //   if (!f) {
    //     RCLCPP_ERROR(get_logger(), "Failed to open wbar file: %s", wbar_bin_path_.c_str());
    //   } else {
    //     f.read(reinterpret_cast<char*>(W.data()), expected_bytes);
    //     if (!f) {
    //       RCLCPP_ERROR(get_logger(), "Failed to read full wbar (short read / mismatch)");
    //     } else {
    //       auto err = rbf_->upload_W(W.data());
    //       if (err != cudaSuccess) {
    //         RCLCPP_ERROR(get_logger(), "upload_W failed: %s", cudaGetErrorString(err));
    //       } else {
    //         RCLCPP_INFO(get_logger(), "Loaded initial weights from %s", wbar_bin_path_.c_str());
    //       }
    //     }
    //   }
    // }


    // Log RBF memory footprint
    uint64_t N = 1;
    for (int i = 0; i < 8; ++i) N *= static_cast<uint64_t>(zetta_ne_);
    double mb = static_cast<double>(N) * sizeof(float) / (1024.0*1024.0);
    RCLCPP_INFO(this->get_logger(), "RBF: ne=%d -> N=ne^8=%llu (%.2f MB buffer)",
                zetta_ne_, static_cast<unsigned long long>(N), mb);

    // if (!rbf_ || !rbf_->ready()) {
    //   RCLCPP_ERROR(this->get_logger(),
    //     "CUDA RBF allocation failed: ne=%d N=%llu bytes=%zu cudaError=%s",
    //     zetta_ne_,
    //     static_cast<unsigned long long>(rbf_ ? rbf_->num_points() : 0ULL),
    //     static_cast<size_t>(rbf_ ? rbf_->bytes_S() : 0),
    //     cudaGetErrorString(rbf_ ? rbf_->last_status() : cudaErrorUnknown));
    //   } 
    // else {
    //   RCLCPP_INFO(this->get_logger(), "CUDA RBF ready: N=%llu (ne=%d)",
    //     static_cast<unsigned long long>(rbf_->num_points()), zetta_ne_);
    //   }
  }

  // --------------------------------------------------------------------------
  // Worker thread (a second thread for running processing_loop)
  // --------------------------------------------------------------------------
  void start_worker_() {   
    processing_thread_ = std::thread(&DataProcessorNode::processing_loop, this);
  }

  // --------------------------------------------------------------------------
  // Odometry callback
  // --------------------------------------------------------------------------
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Cache the newest odometry and learn the frame name to use in processing loop 
    FullState s;
    s.stamp          = rclcpp::Time(msg->header.stamp);
    s.frame_id       = msg->header.frame_id;       // e.g., "mauv_1/odom"
    s.child_frame_id = msg->child_frame_id;        // e.g., "mauv_1/base_link"
    s.position       = msg->pose.pose.position;
    s.orientation    = msg->pose.pose.orientation;
    s.lin_vel        = msg->twist.twist.linear;    // in body frame (base_link)
    s.ang_vel        = msg->twist.twist.angular;   // in body frame (base_link)

    // Only one thread touches latest_state_ at a time
    {    
      std::lock_guard<std::mutex> lk(state_mutex_); 
      latest_state_ = s;
    } 

    // Latch odom frame name
    odom_frame_ = msg->header.frame_id;

    // Capture base frame and TF prefix
    if (base_frame_.empty()) {
        base_frame_ = msg->child_frame_id;     // "mauv_1/base_link" or just "base_link"
        auto slash = base_frame_.find('/');              
        tf_prefix_ = (slash == std::string::npos) 
                      ? ""   // no prefix present
                      : base_frame_.substr(0, slash); // "mauv_1"
        // RCLCPP_INFO(get_logger(), 
        //             "Base frame: %s | TF prefix: %s",
        //             base_frame_.c_str(), tf_prefix_.c_str());
      }

    // create thruster publishers once we know the prefix
    if (!pub_heave_bow_) {
      const std::string ns = tf_prefix_.empty() ? "" : ("/" + tf_prefix_);
      pub_heave_bow_   = 
          this->create_publisher<std_msgs::msg::Float64>(ns + "/control/thruster/heave_bow", 10);
      pub_heave_stern_ = 
          this->create_publisher<std_msgs::msg::Float64>(ns + "/control/thruster/heave_stern", 10);
      pub_surge_       = 
          this->create_publisher<std_msgs::msg::Float64>(ns + "/control/thruster/surge", 10);
      pub_sway_bow_    = 
          this->create_publisher<std_msgs::msg::Float64>(ns + "/control/thruster/sway_bow", 10);
     
      // RCLCPP_INFO(get_logger(), 
      //             "Thruster publishers created on namespace '%s'", ns.c_str());
    }
  }

  // --------------------------------------------------------------------------
  // Global frame auto-detection
  // --------------------------------------------------------------------------
  void try_auto_global_frame_() 
  {
    // Check if already set
    if (global_frame_.size()) {
      return; 
    }
    
    // Derive a prefix from odom_frame_ like "mauv_1" from "mauv_1/odom"
    std::string prefix;
    if (!odom_frame_.empty()) {
      auto slash = odom_frame_.find('/');
      if (slash != std::string::npos) {
      prefix = odom_frame_.substr(0, slash);
      }
    }

    // Add world to the derived prefix
    std::vector<std::string> candidates;
    if (!prefix.empty()) {
      candidates = {
        prefix + "/world",
      };
    }
    
    // Also try un-prefixed fallbacks
    candidates.insert(candidates.end(), {"world"}); 
    
    for (const auto& f : candidates) {
      if (tf_buffer_->canTransform(
            f, odom_frame_, rclcpp::Time(0),
            rclcpp::Duration::from_seconds(0.1))) {
        global_frame_ = f;
        // RCLCPP_INFO(get_logger(), 
        //             "Auto-selected global_frame: %s", 
        //             global_frame_.c_str());
        return;
      }
    }

    // If nothing works, fall back to odom (so the node can still run)
    if (!odom_frame_.empty()) {
      global_frame_ = odom_frame_;
      // RCLCPP_WARN(get_logger(), get_clock(), 50000,
      //   "Could not auto-detect a global frame; falling back to odom_frame '%s'.",
      //   global_frame_.c_str());
    }
  }

  // --------------------------------------------------------------------------
  // Processing loop (runs in worker thread)
  // --------------------------------------------------------------------------
  void processing_loop() {
    rclcpp::Rate rate(10); // 10 Hz
    while (rclcpp::ok()) {
      process_data();
      rate.sleep();
    }
  }

  // --------------------------------------------------------------------------
  // Thruster allocation: solve B_sub * f = tau (4x4 system)
  // --------------------------------------------------------------------------
  bool solve_thruster_forces_ls_(const double tau[4], double f_out[4]) {
    
    // Build augmented matrix: A = [B_sub | tau]: 
    double A[4][5];
    for (int j = 0; j < 4; ++j) {
      A[0][j] = B_[X][j];     // Fx 
      A[1][j] = B_[YAW][j];   // Mz 
      A[2][j] = B_[Z][j];     // Fz 
      A[3][j] = B_[PITCH][j]; // My (note: index 4)
    }
    for (int i = 0; i < 4; ++i) {
      A[i][4] = scale_tau_ * tau[i];   // RHS
    }

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 4; ++col) {
      int piv = col;
      double best = std::fabs(A[piv][col]);
      for (int r = col + 1; r < 4; ++r) {
        double v = std::fabs(A[r][col]);
        if (v > best) { 
          best = v; 
          piv = r; 
        }
      }

      if (best < 1e-10) {
        return false;  // singular / ill-conditioned
      }

      if (piv != col) {
        for (int j = col; j < 5; ++j){
          std::swap(A[piv][j], A[col][j]);
        }
      }

      double diag = A[col][col];
      for (int j = col; j < 5; ++j) {
        A[col][j] /= diag;
      }

      // eliminate below
      for (int r = col + 1; r < 4; ++r) {
        double factor = A[r][col];
        if (factor == 0.0) {
          continue;
        }
        for (int j = col; j < 5; ++j) {
          A[r][j] -= factor * A[col][j];
        }
      }
    }

    // Back substitution
    for (int i = 3; i >= 0; --i) {
      double s = A[i][4];
      for (int j = i + 1; j < 4; ++j) s -= A[i][j] * f_out[j];
      f_out[i] = s;  //  diagonal is 1: (A[i][i] = 1)
    }
    return true;

    // *************************************************************************************
    // B_ = [ d ; r × d ]: Thruster's force applied to CG of the AUV expressed in body frame
    // B_sub uses rows: (0,1,2,4) of B_ corresponding to: Fx, Mz, Fz, My
    // B_sub*f = tau (Solving system of linear algebraic equations) --->
    //    Augmented matrix: A = [B_ | tau]: 
    //      * 4 rows: Fx, Mz, Fz, My
    //      * 5 columns: 4 Thrusters (we should solve) + tau (commanded by the controller)
    //    Solve 4 equations 4 unknowns --> T1, T2, T3, T4
    // **************************************************************************************
  }

  // --------------------------------------------------------------------------
  // Main per-step processing
  // --------------------------------------------------------------------------
  void process_data()
  {
    // Time step-2.112
    rclcpp::Time now = this->get_clock()->now();
    double dt = (last_step_.nanoseconds() == 0) 
                  ? 0.1 
                  : (now - last_step_).seconds();
    if (dt <= 0.0 || dt > 1.0) {
      dt = 0.1;
    }
    last_step_ = now;

    // Integrate waypoint generator (simple ODE)
    rk4_step(sim_time_, dt, X_waypoint_);
    sim_time_ += dt;
    for (int i = 0; i < 4; ++i) { 
      p_[i] = X_waypoint_[i]; 
      v_[i] = X_waypoint_[4 + i];
    }

    // Need an odom frame before TF lookup
    if (odom_frame_.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 50000,
        "Waiting for first odom message to learn odom frame...");
      return;
    }

    // Auto-detect global frame if needed
    if (global_frame_.empty()) {
      try_auto_global_frame_();
      if (global_frame_.empty()) {
        // RCLCPP_WARN_THROTTLE(get_logger(), 
        //   *get_clock(), 50000,
        //   "Still auto-detecting global frame (have odom='%s').",
        //   odom_frame_.c_str());
        return;  
      }
      // ************************************************
      // Note: don't proceed until we have a global frame
      // ************************************************
    }

    RCLCPP_INFO_ONCE(get_logger(), 
                     "Using global_frame = '%s'", global_frame_.c_str());

    // Copy buffered state
    FullState s_copy;
    { 
      std::lock_guard<std::mutex> lk(state_mutex_); 
      s_copy = latest_state_;

      // ************************************************************** 
      // 1-Locks state_mutex_ + 2-copy + 3-unlocks when the block ends 
      // The braces: make the unlock happen immediately after the copy.
      // **************************************************************
    } 

    // Transform pose from odom → world
    geometry_msgs::msg::PoseStamped ps_odom, ps_world;
    ps_odom.header.stamp     = s_copy.stamp;
    ps_odom.header.frame_id  = odom_frame_;
    ps_odom.pose.position    = s_copy.position;
    ps_odom.pose.orientation = s_copy.orientation;

    try {
      auto T_g_o = tf_buffer_->lookupTransform(
        global_frame_, odom_frame_, s_copy.stamp, tf2::durationFromSec(0.2));
        
        // 1) Print translation + quaternion
        // const auto& tr = T_g_o.transform.translation;
        // const auto& q  = T_g_o.transform.rotation;
        // RCLCPP_INFO(this->get_logger(),
        //   "TF %s <- %s | t = [%.3f %.3f %.3f], q = [%.3f %.3f %.3f %.3f]",
        //   global_frame_.c_str(), odom_frame_.c_str(),
        //   tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w);

        // 2) print roll/pitch/yaw
        // tf2::Quaternion q_go;
        // tf2::fromMsg(q, q_go);
        // double r, p, y;
        // tf2::Matrix3x3(q_go).getRPY(r, p, y);
        // RCLCPP_INFO(this->get_logger(),
        //   "TF %s <- %s | RPY = [%.3f %.3f %.3f] rad",
        //   global_frame_.c_str(), odom_frame_.c_str(), r, p, y);

      tf2::doTransform(ps_odom, ps_world, T_g_o);
    } 

    catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF %s<-%s unavailable (%s). Skipping.",
        global_frame_.c_str(), odom_frame_.c_str(), ex.what());
      return;
    }

    // Pose in world: [x, y, z, roll, pitch, yaw]
    // Convert orientation (world frame) quaternion → roll/pitch/yaw
    tf2::Quaternion q_wb;
    tf2::fromMsg(ps_world.pose.orientation, q_wb);
    double roll_w = 0.0, pitch_w = 0.0, yaw_w = 0.0;
    tf2::Matrix3x3(q_wb).getRPY(roll_w, pitch_w, yaw_w);

    // Twist: body → world
    tf2::Matrix3x3 R_wb(q_wb);   // rotation: world <- base_link

    tf2::Vector3 v_b(
      s_copy.lin_vel.x, s_copy.lin_vel.y, s_copy.lin_vel.z);
    tf2::Vector3 w_b(
      s_copy.ang_vel.x, s_copy.ang_vel.y, s_copy.ang_vel.z);

    // **************************************************************************
    // s_copy.lin_vel and s_copy.ang_vel are given in the body frame (base_link).
    // R_wb maps vectors expressed in body frame to world frame.
    // **************************************************************************

    tf2::Vector3 v_w = R_wb * v_b;   // linear velocity in world frame
    geometry_msgs::msg::TransformStamped tf_wb_msg;
    tf_wb_msg.transform.rotation = ps_world.pose.orientation;  // world<-base orientation

    Eigen::Matrix3d T = f_angular_velocity_transform(tf_wb_msg);

    // body angular velocity (p,q,r)
    Eigen::Vector3d w_body(s_copy.ang_vel.x, s_copy.ang_vel.y, s_copy.ang_vel.z);

    // Euler angle rates: [roll_dot, pitch_dot, yaw_dot]
    Eigen::Vector3d euler_dot = T * w_body;
    // Eigen::Vector3d euler_dot = T.inverse() * w_body;


    // tf2::Vector3 w_w = R_wb * w_b;   // angular velocity in world frame

    // Pack pose and twist in World frame
    std::array<double,6> P_world = {
      ps_world.pose.position.x,
      ps_world.pose.position.y,
      ps_world.pose.position.z,
      roll_w, pitch_w, yaw_w
    };

    std::array<double,6> V_world = {
      v_w.x(), v_w.y(), v_w.z(),
      euler_dot(0), euler_dot(1), euler_dot(2)   
    };

   
    // // ----------------------------------------------------------------------
    // // Desired yaw from "path heading + cross-track" law
    // // ----------------------------------------------------------------------
    double A = 10;
    double PI_d_; 
    double rho_d = 1; //clockwise: +1, ccw: -1
    double R_p = 7.0; 
    double X_orbit = std::atan2(P_world[Y], P_world[X]);
    X_orbit = wrapToPi(X_orbit);
    double pc_p = std::hypot(P_world[X], P_world[Y]);


    //  LOS
    double e_max = std::abs(A + R_p);
    double e_min = std::abs(A - R_p);
    double e_y = std::abs(P_world[Y] - p_[WY]);
    
    double AX_val = (std::pow(pc_p,2) + std::pow(R_p,2) - std::pow(A,2)) / (2.0*pc_p*R_p);
    double AX_ang = std::asin(AX_val);
    AX_ang = wrapToPi(AX_ang);
    // if (pc_p > e_max) PI_d_ = X_orbit + PI;
    //     // PI_d_ = wrapToPi(PI_d_);
    // else if (pc_p < e_min) PI_d_ = X_orbit; //P_world[YAW];   //X_orbit;
    // // else if (ey < 0.5) PI_d_ = P_world[YAW];
    // else 
    //    PI_d_ = X_orbit + rho_d*(PI/2 - AX_ang); 
    
    // PI_d_ = wrapToPi(PI_d_);

    // //  Vector Field
    // double kc = 0.5;
    // double xmc = PI/4; 
    // double dc = pc_p - A; 
    // double ax_ang = std::atan(kc*dc);
    // double AX_ang = 2/PI*xmc*ax_ang;
    // PI_d_ = X_orbit + rho_d*(PI/2 + AX_ang);





    // ----------------------------------------------------------------------
    // Desired yaw from "path heading + cross-track" law
    // ----------------------------------------------------------------------
    double PI_p_; 
    // double PI_d_;
    const double Delta = 5.0;         // lookahead distance (tune)
    const double k_p   = 1.0 / Delta;  // gain  (matches kp = 1/delta)

    // 1) Path heading from waypoint motion: chi_p = atan2(dY, dX)
    double vxy = std::hypot(v_[WX], v_[WY]);
      // this is the "waypoint yaw" (path heading)
    PI_p_ = std::atan2(v_[WY], v_[WX]);
    // PI_d_ = std::atan2(V_world[Y], V_world[X]);
    // if (vxy > 1e-4) PI_p_ = std::atan2(v_[WY], v_[WX]);
   
    // else        PI_p_ = P_world[YAW];
    

    // print **********************
    PI_p_ = wrapToPi(PI_p_);

    // 2) Cross-track error in path frame
    // ex, ey: error in world frame
    double ex = P_world[X] - p_[WX];
    double ey = P_world[Y] - p_[WY];

    // Rotate into path frame (x_e along path, y_e lateral)
    double cos = std::cos(PI_p_);
    double sin = std::sin(PI_p_);
    double x_e =  cos * ex + sin * ey;
    double y_e = -sin * ex + cos * ey;   // lateral (cross-track) error
    // double y_e  =  ey;
    
    // 3) Heading correction term X_p_ = -atan(kp * y_e)
    double X_p_ ;
    double psi_d_raw;  
    X_p_ = -std::atan(k_p * y_e);

    // // y_e is your cross-track error in the path frame (you called it ey sometimes)
    // const double enter_hold = 5e-3;   // when |y_e| gets very small -> lock to PI_p_
    // const double exit_hold  = 2.0;    // stay locked until |y_e| grows beyond 3 m

    // static bool hold_PI = false;      // latch state

    // // Update latch
    // if (!hold_PI) {
    //   if (std::abs(y_e) <= enter_hold) hold_PI = true;
    // } else {
    //   if (std::abs(y_e) > exit_hold) hold_PI = false;
    // }

    // // Compute psi_d_raw based on latch
    // double psi_d_raw;
    // if (hold_PI) {
    //   psi_d_raw = PI_p_;
    // } else {
    //   // your "normal/orbit" law when NOT holding
    //   // choose ONE of these depending on what you want:

    //   // (A) LOS correction:
    //   psi_d_raw = wrapToPi(PI_p_ + X_p_);

    //   // (B) Orbit law (if you really want this instead of LOS):
    //   // psi_d_raw = wrapToPi(X_orbit + rho_d * (PI/2 - AX_ang));
    // }




    if (std::abs(y_e) > 5e-3) 
        // if (pc_p > e_max) psi_d_raw = X_orbit + PI;
        //     // PI_d_ = wrapToPi(PI_d_);
        // else if (pc_p < e_min) psi_d_raw = X_orbit; //P_world[YAW];   //X_orbit;
        // // else if (ey < 0.5) PI_d_ = P_world[YAW];
        // else 
        //   psi_d_raw = X_orbit + rho_d*(PI/2 - AX_ang); 
        
        // PI_d_ = wrapToPi(PI_d_);

        psi_d_raw = wrapToPi(PI_p_ + X_p_);
    else
    psi_d_raw = PI_p_;
 
    

    if (pub_pi_p_) {
    nav_msgs::msg::Odometry o;
    o.header.stamp = s_copy.stamp;
    o.header.frame_id = global_frame_;
    o.child_frame_id = "pi_p";

    // put it at the waypoint position (or at the vehicle position, your choice)
    o.pose.pose.position.x = p_[WX];
    o.pose.pose.position.y = p_[WY];
    o.pose.pose.position.z = p_[WZ];

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, psi_d_raw);
    o.pose.pose.orientation = tf2::toMsg(q);

    pub_pi_p_->publish(o);
    }


    // // 4) Raw desired heading
    // double psi_d_raw = wrapToPi(PI_p_ - 2*X_p_);

    // // ----------------------------------------------------------------------
    // // 4) Low-pass filter on desired heading + derivative
    // // // ----------------------------------------------------------------------
    // static double psi_d_prev = 0.0;
    // double psi_dot_d = wrapToPi(psi_d_raw - psi_d_prev) / dt;
    // psi_d_prev = psi_d_raw;

    // // // use psi_d as the filtered desired yaw
    double psi_d = psi_d_raw;
    // // --- Smooth desired yaw (wrap-safe low-pass) ---
    // static bool psi_init = false;
    // static double psi_d_filt = 0.0;   // filtered desired yaw

    // const double tau_psi = 1.0;       // seconds (tune: bigger = smoother)
    // double alpha_psi = dt / (tau_psi + dt);
    // if (alpha_psi < 0.0) alpha_psi = 0.0;
    // if (alpha_psi > 1.0) alpha_psi = 1.0;

    // if (!psi_init) {
    //   psi_d_filt = psi_d_raw;         // initialize once
    //   psi_init = true;
    // } else {
    //   // filter the shortest angular difference
    //   double e = wrapToPi(psi_d_raw - psi_d_filt);
    //   psi_d_filt = wrapToPi(psi_d_filt + alpha_psi * e);
    // }

    // // filtered desired yaw
    // double psi_d = psi_d_raw; //psi_d_filt;

    // filtered desired yaw rate
    static double psi_d_prev_filt = psi_d;
    double psi_dot_d = wrapToPi(psi_d - psi_d_prev_filt) / dt;
    psi_d_prev_filt = psi_d;

    // Publish world odometry
    if (pub_world_odom_) {
      nav_msgs::msg::Odometry world_msg;
      world_msg.header.stamp    = s_copy.stamp;       // same as odom
      world_msg.header.frame_id = global_frame_;      // e.g. "mauv_1/world"
      world_msg.child_frame_id  = base_frame_;        // e.g. "mauv_1/base_link"

      world_msg.pose.pose.position.x = P_world[X];
      world_msg.pose.pose.position.y = P_world[Y];
      world_msg.pose.pose.position.z = P_world[Z];

      tf2::Quaternion q_rpy;
      q_rpy.setRPY(P_world[ROLL], P_world[PITCH], P_world[YAW]);
      world_msg.pose.pose.orientation = tf2::toMsg(q_rpy);

      world_msg.twist.twist.linear.x  = V_world[X];
      world_msg.twist.twist.linear.y  = V_world[Y];
      world_msg.twist.twist.linear.z  = V_world[Z];
      world_msg.twist.twist.angular.x = V_world[ROLL];
      world_msg.twist.twist.angular.y = V_world[PITCH];
      world_msg.twist.twist.angular.z = V_world[YAW];

      pub_world_odom_->publish(world_msg);
    }

    // Publish waypoint (leader) in world frame
    if (pub_wp_odom_) {
      nav_msgs::msg::Odometry wp_msg;
      wp_msg.header.stamp = s_copy.stamp;
      wp_msg.header.frame_id = global_frame_;  // same as P_world
      wp_msg.child_frame_id  = "waypoint";

      // waypoint position [X, Y, Z, Pitch]
      wp_msg.pose.pose.position.x = p_[WX];
      wp_msg.pose.pose.position.y = p_[WY];
      wp_msg.pose.pose.position.z = p_[WZ];

      // Waypoint orientation:
      // roll_ref = 0
      // pitch_ref = p_[WPITCH]
      // yaw_ref   = psi_d  (desired yaw from LOS)
      tf2::Quaternion q_wp;
      q_wp.setRPY(0.0, p_[WPITCH], psi_d); 
      // q_wp.setRPY(0.0, pitch_d, psi_d);

      wp_msg.pose.pose.orientation = tf2::toMsg(q_wp);

      // waypoint velocities
      wp_msg.twist.twist.linear.x  = v_[WX];
      wp_msg.twist.twist.linear.y  = v_[WY];
      wp_msg.twist.twist.linear.z  = v_[WZ];
      wp_msg.twist.twist.angular.y = v_[WPITCH];
      // wp_msg.twist.twist.angular.y = pitch_dot_d;
      wp_msg.twist.twist.angular.z = psi_dot_d;

      pub_wp_odom_->publish(wp_msg);
    }

    // Control errors in World
    z1_[WX]     = P_world[X] - p_[WX];                   // ex
    z1_[WYAW]   = P_world[Y] - p_[WY]; // wrapToPi(P_world[YAW] - psi_d);        // e_yaw
    z1_[WZ]     = P_world[Z] - p_[WZ];                   // ez
    z1_[WPITCH] = wrapToPi(P_world[PITCH] - p_[WPITCH]); // e_pitch

    // ************************************************************
    // z1 = [ex, e_yaw, ez, e_pitch]
    // ex      = x - x_ref
    // e_yaw   = yaw - psi_d          (psi_d: desired yaw from LOS)
    // ez      = z - z_ref
    // e_pitch = pitch - pitch_ref
    // ************************************************************

    double H1z1[4]{};
    mat4_mul_vec(H1_, z1_.data(), H1z1);

    // v_ref = [ẋ_ref, ψ̇_ref, ż_ref, θ̇_ref]
    // double v_ref[4] = { v_[WX], psi_dot_d, v_[WZ], v_[WPITCH]};
    double v_ref[4] = { v_[WX], v_[WY], v_[WZ], v_[WPITCH]};


    for (int i=0; i<4; ++i) {
      beta_[i] = -H1z1[i] + v_ref[i];
    }

    // double v_t =  cos * V_world[X] + sin * V_world[Y];    // actual speed along path
    // double v_t_ref = vxy;                              // desired along-path speed (or project v_ too)

    // double v_ref[4] = { v_t_ref, psi_dot_d, v_[WZ], v_[WPITCH] };
    //     for (int i=0; i<4; ++i) {
    //   beta_[i] = -H1z1[i] + v_ref[i];
    // }

    // // after beta is computed:
    // z2_[WX] = v_t - beta_[WX];

    // z2 = [eẋ, e_ψ̇, eż, e_θ̇]
    z2_[WX]     = V_world[X]     - beta_[WX];     // eẋ
    z2_[WYAW]   = V_world[Y]     - beta_[WY];   // e_ψ̇ (yaw rate error)
    z2_[WZ]     = V_world[Z]     - beta_[WZ];     // eż
    z2_[WPITCH] = V_world[PITCH] - beta_[WPITCH]; // e_θ̇ (pitch rate error)

    // Publish z1 (position error) with z2 as its velocity error
    if (pub_z1_odom_) {
      nav_msgs::msg::Odometry z1_msg;
      z1_msg.header.stamp    = s_copy.stamp;
      z1_msg.header.frame_id = global_frame_;   // same frame as P_world
      z1_msg.child_frame_id  = "z1";

      // Position error
      z1_msg.pose.pose.position.x = z1_[WX];    // ex
      z1_msg.pose.pose.position.y = z1_[WY];    //P_world[Y] - p_[WY]; // unused in control, but used in error visualization
      z1_msg.pose.pose.position.z = z1_[WZ];    // ez

      // Orientation encodes BOTH pitch and yaw errors (roll = 0)
      // RPY(z1): [0, e_pitch, e_yaw]
      tf2::Quaternion q_z1;
      q_z1.setRPY(0.0,              // roll error = 0
                  z1_[WPITCH],      // pitch error
                  z1_[WYAW]);       // yaw error
      z1_msg.pose.pose.orientation = tf2::toMsg(q_z1);

      // Velocity error = z2
      z1_msg.twist.twist.linear.x  = z2_[WX];      // evx
      z1_msg.twist.twist.linear.y  = z2_[WY];          // unused, 
      z1_msg.twist.twist.linear.z  = z2_[WZ];      // evz

      z1_msg.twist.twist.angular.x = 0.0;
      z1_msg.twist.twist.angular.y = z2_[WPITCH];  // eω_pitch
      z1_msg.twist.twist.angular.z = z2_[WYAW];    // eω_yaw
      pub_z1_odom_->publish(z1_msg);
    }

    // Pack world state (for RBF)
    std::array<double,12> X_world = {
      P_world[0], P_world[1], P_world[2],
      P_world[3], P_world[4], P_world[5],
      V_world[0], V_world[1], V_world[2],
      V_world[3], V_world[4], V_world[5]
    };

    float x_f[8];
    x_f[0] = static_cast<float>(X_world[X] / 100.0);           // scale_pos_
    x_f[1] = static_cast<float>(X_world[Y] / 50.0);         // scale_ang_
    x_f[2] = static_cast<float>(X_world[Z]/10.0);           // scale_pos_
    x_f[3] = static_cast<float>(X_world[PITCH] * 5.0);       // scale_ang_
    x_f[4] = static_cast<float>(X_world[X + 6]);       // scale_lin_
    x_f[5] = static_cast<float>(X_world[Y + 6]);     // scale_angvel_
    x_f[6] = static_cast<float>(X_world[Z + 6] *2);       // scale_lin_
    x_f[7] = static_cast<float>(X_world[PITCH + 6] *10.0);   // scale_angvel_

    // RBF evaluation and weight update
    if (rbf_ && rbf_->ready()) {

      // 1) compute S(x) on GPU
      rbf_->compute_S(x_f);
      
      // // Optional debug of S(x)
      // {
      //   static int dbg_counter = 0;
      //   if (dbg_counter++ % 50 == 0) {   // print every 50 cycles
      //     const std::uint64_t N = rbf_->num_points();
      //     std::vector<float> S(N);
      //     rbf_->copy_S_to_host(S.data());

      //     double minS = 1e300, maxS = -1e300;
      //     std::uint64_t idx_max = 0;

      //     for (std::uint64_t i = 0; i < N; ++i) {
      //       const double v = (double)S[i];
      //       if (v < minS) minS = v;
      //       if (v > maxS) { maxS = v; idx_max = i; }
      //     }

      //     float c[8];
      //     if (rbf_->download_center(idx_max, c) == cudaSuccess) {
      //       // RCLCPP_INFO(get_logger(),
      //       //   "S: min=%.3e max=%.3e idx_max=%llu center=[%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f]",
      //       //   minS, maxS, (unsigned long long)idx_max,
      //       //   c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
      //     }
      //   }
      // }

      // 2) F = [w1^T S, ..., w4^T S]
      float Ff[4];
      rbf_->dot_F(Ff);
      for (int i = 0; i < 4; ++i) {
        F_[i] = static_cast<double>(Ff[i]);
      } 

      // 3) update weights using z2[i]
      float z2f[4];
      for (int i=0; i<4;++i) {
        z2f[i] = static_cast<float>(z2_[i]);
      }

      static const float gamma1 = 2.2e-4f;   //  0.0;
      static const float sigma  = 5.0e-1f;   //  0.0;

      rbf_->update_w(z2f, gamma1, sigma);

      // Publish L2 norms every 10th loop
      static int norm_counter = 0;
      if (pub_w_norms_ && (++norm_counter % 10 == 0)) {  // ~1 Hz if loop is 10 Hz
        const uint64_t N = rbf_->num_points();
        std::vector<float> W(4 * N);

        if (rbf_->download_W(W.data()) == cudaSuccess) {
          double sum_sq[4] = {0.0, 0.0, 0.0, 0.0};

          for (uint64_t j = 0; j < N; ++j) {
            for (int i = 0; i < 4; ++i) {
              float wij = W[i * N + j];   // row-major blocks: [w0(0..N-1), w1(..), ...]
              sum_sq[i] += static_cast<double>(wij) * static_cast<double>(wij);
            }
          }

          std_msgs::msg::Float32MultiArray msg;
          msg.layout.dim.resize(1);
          msg.layout.dim[0].label  = "norms";
          msg.layout.dim[0].size   = 4;
          msg.layout.dim[0].stride = 4;

          msg.data.resize(4);
          for (int i = 0; i < 4; ++i) {
            msg.data[i] = static_cast<float>(std::sqrt(sum_sq[i])); // ||wi||
          }

          pub_w_norms_->publish(msg);
        }
      }
    }

    // // OPTIONAL: publish full weight matrix W (4×N) at low rate
    // static size_t w_tick = 0;
    // if (rbf_ && rbf_->ready() && pub_W_ && (w_tick++ % 50 == 0)){   // publish every 50 cycles (adjust as you like)
    //   std::vector<float> W_host;
    //   rbf_->copy_W_to_host(W_host);         // size = 4 * N

    //   std_msgs::msg::Float32MultiArray msg;
    //   msg.layout.dim.resize(2);

    //   // dim[0] = output index (0..3)
    //   msg.layout.dim[0].label  = "output";
    //   msg.layout.dim[0].size   = 4;
    //   msg.layout.dim[0].stride = 4 * static_cast<uint32_t>(rbf_->num_points());

    //   // dim[1] = RBF index (0..N-1)
    //   msg.layout.dim[1].label  = "rbf";
    //   msg.layout.dim[1].size   = static_cast<uint32_t>(rbf_->num_points());
    //   msg.layout.dim[1].stride = static_cast<uint32_t>(rbf_->num_points());

    //   msg.data = std::move(W_host);   // flatten: length = 4 * N

    //   pub_W_->publish(msg);
    // }

    // tau = F - H2*z2 - z1, in world frame
    double H2z2[4]{};
    mat4_mul_vec(H2_, z2_.data(), H2z2);
    for (int i = 0; i < 4; ++i) {
      tau_[i] = F_[i] - H2z2[i] - z1_[i];   
    }

    // double Ft = tau_[WX];   // force along path tangent
    // double Fx_w =  c * Ft;
    // double Fy_w =  s * Ft;
    
    // Map tau to body frame: world → base_link
    tf2::Matrix3x3 R_bw = R_wb.transpose();

    // τ in world frame:
    // tau_[WX]    -> Fx (surge)
    // tau_[WZ]    -> Fz (heave)
    // tau_[WY]    -> Fy (yaw moment)
    // tau_[WPITCH]-> My (pitch moment)
    tf2::Vector3 F_world(tau_[WX], tau_[WY], tau_[WZ]);
    // tf2::Vector3 F_world(tau_[WX],    0.0,    tau_[WZ]);
    tf2::Vector3 M_world(0.0, tau_[WPITCH], 0.0);   // tau_[WYAW]

    tf2::Vector3 F_body = R_bw * F_world;
    // tf2::Vector3 M_body = R_bw * M_world;
    Eigen::Vector3d M_WOLRD(0.0, tau_[WPITCH], 0.0);
    Eigen::Vector3d M_body = T.inverse() * M_WOLRD;
    // Eigen::Vector3d M_body = T * M_WOLRD;

    
    

    tau_body_[WX]     = F_body.x();   // body Fx
    tau_body_[WY]     = F_body.y();   // body Fy
    tau_body_[WZ]     = F_body.z();   // body Fz
    tau_body_[WPITCH] = M_body.y();   // body My (pitch)

    // Thruster allocation
    if (!alloc_ready_) {
    try {
      (void)compute_allocation_from_tf_();
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Alloc TF lookup failed: %s", ex.what());
      return; // try again next cycle
    }
    if (!alloc_ready_) {
      return; // safety
    }
  }
  // Solve for nominal thruster forces (no limits/nonlinearity yet)
  double f_ls[4];
  if (solve_thruster_forces_ls_(tau_body_.data(), f_ls)) {
    for (int i = 0; i < 4; ++i) {
      thr_force_[i] = f_ls[i];
    }
  } else {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Allocation solve failed (BᵀB near singular).");
    // fall back to zero force or keep previous:
    for (int i = 0; i < 4; ++i) {
      thr_force_[i] = 0.0;
    }
  }

  // Thrusters command (filtered)
  for (int i = 0; i < 4; ++i) {
    double u_raw = force_to_cmd_(thr_force_[i], i);

    double a = alpha_;  // default smoothing

    thr_cmd_[i] = u_raw;


  //   if (i == SURGE_T) {
  //     // desired waypoint Y in [-5, 5] -> forward only
  //     // const bool forward_zone = (p_[WY] >= -.1 && p_[WY] <= 0.1);
  //     const bool forward_zone = (v_[WX] >= -.02 && v_[WX] <= 0.02);

  //     double u_target = u_raw;
  //     if (forward_zone) {
  //       u_target = std::max(0.16, u_raw);   // forward-only
  //     }

  //     a = alpha_surge_;
  //     thr_cmd_filt_[i] = (1.0 - a) * thr_cmd_filt_[i] + a * u_target;

  //     // final safety clamp
  //     if (thr_cmd_filt_[i] < -1.0) thr_cmd_filt_[i] = -1.0;
  //     if (thr_cmd_filt_[i] >  1.0) thr_cmd_filt_[i] =  1.0;

  //     thr_cmd_[i] = thr_cmd_filt_[i];
  //     continue;
  //   }

  //   if (i == HEAVE_BOW_T || i == HEAVE_STERN_T) {
  //     a = alpha_heave_;  
  //   }
  //   else {
  //     a = 2*alpha_sway_; 
  //   }

  //   thr_cmd_filt_[i] = (1.0 - a) * thr_cmd_filt_[i] + a * u_raw;
  //   thr_cmd_[i]      = 0.6*(thr_cmd_filt_[i]);
  
  }

  // Debug logs (throttled) 
  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 20000,
    "X_d=[%.3f, %.3f, %.3f,%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]", 
     X_waypoint_[0],X_waypoint_[1], psi_d, X_waypoint_[2],X_waypoint_[3],
     X_waypoint_[4],X_waypoint_[5], psi_dot_d, X_waypoint_[6],X_waypoint_[7]);

  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 20000,
    "X_w=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]", 
    P_world[0],P_world[1], P_world[5], P_world[2],P_world[4],
    V_world[0],V_world[1], V_world[5], V_world[2],V_world[4]);

  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 20000,
    "x_f: [%.1f %.3f %.3f %.3f %.3f %.3f %.3f %.3f]",
     x_f[0], x_f[1], x_f[2], x_f[3],
     x_f[4], x_f[5], x_f[6], x_f[7]);

  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 20000,
    "z1=[%.3e, %.3e, %.3e, %.3e]", 
    z1_[0],z1_[1],z1_[2],z1_[3]); 

  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 20000,
    "z2=[%.3f, %.3f, %.3f %.3f]", 
    z2_[0],z2_[1],z2_[2],z2_[3]); 

  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 20000,
    "beta=[%.3e, %.3e, %.3e, %.3e]", 
    beta_[0],beta_[1],beta_[2],beta_[3]); 

  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 20000,
    "F=[%.3e, %.3e, %.3e, %.3e]", 
    F_[0],F_[1],F_[2],F_[3]);

  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 20000,
    "tau = [%.3f %.3f %.3f %.3f ]", 
    tau_[0], tau_[1], tau_[2], tau_[3]);

  RCLCPP_INFO_THROTTLE(
    this->get_logger(), *this->get_clock(), 20000,
    "tau_body = [%.3f %.3f %.3f %.3f ]",
    tau_body_[0], tau_body_[1], tau_body_[2], tau_body_[3]);
  
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 20000,
    "thr_cmd = [%.2f %.2f %.2f %.2f]", 
    thr_cmd_[0], thr_cmd_[1], thr_cmd_[2], thr_cmd_[3]);

  // Publish thruster commands
  if (pub_heave_bow_) {
    std_msgs::msg::Float64 msg;
    msg.data = thr_cmd_[SURGE_T];       pub_surge_->publish(msg);
    msg.data = thr_cmd_[HEAVE_BOW_T];   pub_heave_bow_->publish(msg);
    msg.data = thr_cmd_[HEAVE_STERN_T]; pub_heave_stern_->publish(msg);
    msg.data = thr_cmd_[SWAY_BOW_T];    pub_sway_bow_->publish(msg);
    }
  }

  // --------------------------------------------------------------------------
  // Thruster force → command mapping: T = c |u| u, solve for u
  // --------------------------------------------------------------------------
  inline double force_to_cmd_(double T, int thruster_id) const
  {
    const double c = (thruster_id == SURGE_T) ? thrust_coeff_surge_ : thrust_coeff_other_;

    if (c <= 0.0) {
      return 0.0;
    }
    double a = std::abs(T) / c;
    if (a < 0.0) {
      a = 0.0;
    }
    double u = std::sqrt(a);
    if (u > 1.0) {
      u = 1.0;
    }
    return std::copysign(u, T);
  }

  // --------------------------------------------------------------------------
  // Math helpers (4x4 multiply + angle wrapping)
  // --------------------------------------------------------------------------
  static inline void mat4_mul_vec(const std::array<double,16>& M,
                                  const double v[4], 
                                  double out[4]) 
  {
    for (int r=0; r<4; ++r) {
      double s=0.0;
      for (int c=0; c<4; ++c) {
        s += M[r*4 + c]*v[c];
      }
      out[r]=s;
    }
  }

    static inline double wrapToPi(double a)
  {
    while (a >  M_PI) a -= 2.0*M_PI;
    while (a < -M_PI) a += 2.0*M_PI;
    return a;
  }

  static inline double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
  }

  // 5th-order smooth move (min-jerk): position/velocity/acc = 0 at endpoints
  static inline double smooth5(double s) {
    s = clamp(s, 0.0, 1.0);
    return 10*s*s*s - 15*s*s*s*s + 6*s*s*s*s*s;
  }
  static inline double smooth5_d1(double s) {
    s = clamp(s, 0.0, 1.0);
    return 30*s*s - 60*s*s*s + 30*s*s*s*s;
  }

  // Z reference with dwell + smooth moves, and pitch coupled to vertical velocity
  inline void z_pitch_ref(double t, double &z_ref, double &dz_ref,
                          double &pitch_ref, double &dpitch_ref) const
  {
    const double z_min = z_mid - z_amp;   // -> -3
    const double z_max = z_mid + z_amp;   // -> -1
    const double dz = (z_max - z_min);

    const double pitch_max = pitch_amp;   // pitch limit


    const double T_hold = 200.0;   // seconds at top/bottom (tune)
    const double T_move = 400.0;  // seconds to move between (tune: bigger => smaller vz)

    const double Tcycle = 2*T_hold + 2*T_move;
    double tau = std::fmod(t, Tcycle);
    if (tau < 0.0) tau += Tcycle;

    // defaults
    z_ref      = z_min; 
    dz_ref     = 0.0;
    pitch_ref  = 0.0; 
    dpitch_ref = 0.0;

    auto move = [&](double s01, bool up) {
      // s in [0,1]
      double p  = smooth5(s01);
      double dp = smooth5_d1(s01);

      // position
      z_ref = up ? (z_min + dz*p) : (z_max - dz*p);

      // dz/dt
      const double sign = up ? +1.0 : -1.0;
      dz_ref = sign * dz * (dp / T_move);
    };

    if (tau < T_hold) {
      // hold bottom
      z_ref = z_min;
    } else if (tau < T_hold + T_move) {
      // move up
      move((tau - T_hold)/T_move, true);
    } else if (tau < T_hold + T_move + T_hold) {
      // hold top
      z_ref = z_max;
    } else {
      // move down
      move((tau - (T_hold + T_move + T_hold))/T_move, false);
    }

    // Pitch follows vertical velocity; => pitch=0 at holds + turning points
    // normalize so max |pitch| <= pitch_max
    // (smooth5_d1 peak is 1.875 at s=0.5)
    const double dz_peak = std::max(1e-9, std::abs(dz) * (1.875 / T_move));
    // pitch_ref = clamp(pitch_max * (dz_ref / dz_peak), -pitch_max, +pitch_max);
    pitch_ref = 0.0;

    // dpitch_ref (optional; set 0 if you don’t need it)
    dpitch_ref = 0.0;
  }

  // --------------------------------------------------------------------------
  // Allocation matrix B from TF
  // --------------------------------------------------------------------------
  bool compute_allocation_from_tf_() 
  {
    if (base_frame_.empty()) {
      return false;
    }

    for (int i = 0; i < 4; ++i) {
      // Compose fully qualified thruster frame (with prefix if any)
      const std::string link = tf_prefix_.empty()
        ? thr_links_[i]
        : (tf_prefix_ + "/" + thr_links_[i]);

      // Transform from thruster link to base frame
      auto T_b_t = tf_buffer_->lookupTransform(
          base_frame_, link, rclcpp::Time(0), tf2::durationFromSec(0.2));

      // r_i: position of the thruster in base frame
      const auto& tr = T_b_t.transform.translation;
      tf2::Vector3 r(tr.x, tr.y, tr.z);

      // d_i: +X axis of thruster, expressed in base frame
      geometry_msgs::msg::Vector3Stamped ex_t, ex_b;
      ex_t.header.frame_id = link;
      ex_t.vector.x = 1.0; 
      ex_t.vector.y = 0.0; 
      ex_t.vector.z = 0.0;
      tf2::doTransform(ex_t, ex_b, T_b_t);

      tf2::Vector3 d(ex_b.vector.x, ex_b.vector.y, ex_b.vector.z);
      if (d.length2() == 0.0) {
        return false;
      }
      d.normalize();

      // Fill B = [ d ; r × d ]
      B_[X][i] = d.x();  
      B_[Y][i] = d.y();  
      B_[Z][i] = d.z();
      tf2::Vector3 m = r.cross(d);
      B_[ROLL][i]  = m.x();  
      B_[PITCH][i] = m.y();  
      B_[YAW][i]   = m.z();  
    }

    alloc_ready_ = true;

    // (Optional) print B
    // RCLCPP_INFO(get_logger(),
    //   "Allocation B = [\n"
    //   " %.3f %.3f %.3f %.3f;\n"
    //   " %.3f %.3f %.3f %.3f;\n"
    //   " %.3f %.3f %.3f %.3f;\n"
    //   " %.3f %.3f %.3f %.3f;\n"
    //   " %.3f %.3f %.3f %.3f]", 
    //   B_[X][0],     B_[X][1],     B_[X][2],     B_[X][3],
    //   B_[Y][0],     B_[Y][1],     B_[Y][2],     B_[Y][3],
    //   B_[Z][0],     B_[Z][1],     B_[Z][2],     B_[Z][3],
    //   B_[PITCH][0], B_[PITCH][1], B_[PITCH][2], B_[PITCH][3],
    //   B_[YAW][0], B_[YAW][1], B_[YAW][2], B_[YAW][3]);

    return true;
  }

  // --------------------------------------------------------------------------
  // Waypoint generator dynamics xdot = f(t, x)
  // --------------------------------------------------------------------------
  void compute_xdot(double t, 
                    const std::array<double,8>& x,
                    std::array<double,8>& xdot) 
  {
    const double* q  = x.data();        // [X, Y, Z, Pitch]
    const double* qd = x.data() + 4;    // [dX, dY, dZ, dPitch]

    double dq[4];
    double A0_qd[4];
    double B0_q[4];

    mat4_mul_vec(A10_, qd, dq);
    mat4_mul_vec(A0_,  qd, A0_qd);
    mat4_mul_vec(B0_,  q,  B0_q);

    // double u[4] = {0.0, 0.0, kp_z * z_ref, 0.0}; 
    double zref, dzref, thref, dthref;
    z_pitch_ref(t, zref, dzref, thref, dthref);

    double u[4] = {0.0, 0.0, 0.0, 0.0};

    // Track Zref(t)
    u[WZ] = kp_z * zref;

    // Track pitch_ref(t)
    u[WPITCH] = w_pitch * thref;


    for (int i=0;i<4;++i) {
      xdot[i]     = dq[i];                     // dq
      xdot[4 + i] = A0_qd[i] + B0_q[i] + u[i]; // dqdot
    }
  }

  // --------------------------------------------------------------------------
  // RK4 integration step
  // --------------------------------------------------------------------------
  void rk4_step(double t, double dt, std::array<double,8>& x) 
  {
    std::array<double,8> k1{}, k2{}, k3{}, k4{}, xt{};

    compute_xdot(t, x, k1);

    for (int i = 0; i < 8; ++i) {
      xt[i] = x[i] + 0.5 * dt * k1[i];
    }
    compute_xdot(t + 0.5 * dt, xt, k2);

    for (int i = 0; i < 8; ++i) {
      xt[i] = x[i] + 0.5 * dt * k2[i];
    }
    compute_xdot(t + 0.5 * dt, xt, k3);

    for (int i = 0; i < 8; ++i) {
      xt[i] = x[i] + dt * k3[i];
    }
    compute_xdot(t + dt, xt, k4);

    for (int i = 0; i < 8; ++i) {
      x[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
  }



  // --------------------------------------------------------------------------
  // Member variables
  // --------------------------------------------------------------------------

  // Simulation & timing 
  double   sim_time_{0.0};
  rclcpp::Time last_step_;

  // RBF / learning 
  int    zetta_ne_;                // samples per dimension (-> ne^8 total points)
  double lambda_;                  // Gaussian width parameter
  std::unique_ptr<CudaRBF> rbf_;   // CUDA/RBF
  // bool freeze_weights_{false};
  std::string wbar_bin_path_{"/home/soslab-p330/ros2_ws/wbar.bin"};


  // Latest state
  FullState latest_state_;
  std::mutex state_mutex_;

  /// Waypoint dynamics matrices (4x4, row-major)
  std::array<double,16> A10_{};
  std::array<double,16> A0_{};
  std::array<double,16> B0_{};

  // Waypoint generator pose/vel 
  std::array<double,4>  p_{};
  std::array<double,4>  v_{};
  std::array<double,8> X_waypoint_{};

  // Control variables 
  std::array<double,4>  z1_{};
  std::array<double,16> H1_{};
  std::array<double,4>  beta_{};
  std::array<double,4>  z2_{};

  // Gains and outputs 
  std::array<double,16> H2_{};
  std::array<double,4>  tau_{};  //tau = F - H2*z2 - z1, in world frame
  std::array<double,4> tau_body_{};
  std::array<double,4>  F_{};  

  // TF-related variables
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;                // a box that stores all transforms received
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;   // the little worker that listens to /tf and fills the buffer
  std::string global_frame_{""};                              //  mauv_1/world
  std::string odom_frame_{""};                                // learned from first odom message
  std::string base_frame_{};
  std::string tf_prefix_{};

  // Thruster command filtering
  std::array<double,4> thr_cmd_filt_{};  // initialized to {0,0,0,0}
  double alpha_{0.2};                    // 0<alpha<=1; smaller = smoother
  double alpha_heave_{0.05};             // smoother filtering for heave (tune this)
  double alpha_surge_{0.02};             // smoother filtering for surge (tune this)  
  double alpha_sway_{0.35};

  // Normalization scales
  double scale_pos_{1.0};      // 100.0   
  double scale_ang_{1.0};      // PI
  double scale_lin_{1.0};      // 2.0     
  double scale_angvel_{1.0};   // 2.0
  double scale_tau_{0.5};      // 0.5

  // Centers 
  // float rbf_lo_{-10.0f};
  // float rbf_hi_{ 10.0f};
  std::array<float, 8> rbf_lo8_{{-1,-1,-1,-1,-1,-1,-1,-1}};
  std::array<float, 8> rbf_hi8_{{ 1, 1, 1, 1, 1, 1, 1, 1}};


  // waypoints parameters
  const double wx         = 0.005;
  const double wy         = 2.0 * wx;   // 2.0 * wx;    
  const double z_mid      = -2.0;   // center of [-3, -1]
  const double z_amp      = 1.0;   // amplitude -> [-3, -1]
  const double pitch_amp  = 0.15;  // rad -> +/-0.05

  // Frequencies (small => small vz and pitch rate)
  const double wz_traj    = 0.01;   // rad/s  (period ~ 314 s)
  const double wp_traj    = 0.01;   // rad/s

  // Make Z and Pitch undamped oscillators
  const double kd_z       = 1.0;
  const double zeta_pitch = 4.0;

  // These are the "spring constants" in your B0_ matrix
  const double kp_z       = 0.25; //wz_traj * wz_traj;     // = ωz^2
  const double w_pitch    = 4.0; //wp_traj * wp_traj;     // = ωp^2

  // This is the equilibrium point for Z
  const double z_ref      = z_mid;  // important: center at -2

  // const double kp_z       = 1.0;
  // const double kd_z       = 2.0;               
  // const double z_ref      = -2.0;                    
  // const double w_pitch    = 0.0;
  // const double zeta_pitch = 2.0;     
  
  // Subscriptions
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  // Thruster command publishers
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_heave_bow_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_heave_stern_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_surge_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_sway_bow_;

  // Visualization publishers
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_world_odom_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_wp_odom_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_z1_odom_; 
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_pi_p_;

  // NN Weights (W:4*N) publishers
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_W_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_w_norms_;

  // Worker thread
  std::thread processing_thread_;

  // Thruster frames (unprefixed; we'll prepend tf_prefix_ if present)
  std::array<std::string,4> thr_links_{
    "surge_thruster_link",
    "heave_bow_thruster_link",
    "heave_stern_thruster_link",
    "sway_bow_thruster_link"
  };

  // 6x4 allocation matrix B (body frame). Row-major: B[row:position dimension][col:thruster]
  double B_[6][4]{};      
  bool   alloc_ready_{false};

  // thrust model (T = c |u| u) 
  // double                thrust_coeff_{99.12};  
  double thrust_coeff_surge_{99.12};
  double thrust_coeff_other_{57.26};
 
  std::array<double, 4> thr_force_{};           
  std::array<double, 4> thr_cmd_{};             

  // Waypoint CSV logging (ONLY p_ and v_)
  // std::ofstream  wp_csv_;
  // bool           wp_log_{true};  // enable/disable
  // std::string    wp_csv_path_{"/tmp/waypoint.csv"};
  // int            wp_decim_{1};   // write every Nth loop (1 = every loop)
  // size_t         wp_tick_{0};
};

// ============================================================================
// main
// ============================================================================

int main(int argc , char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DataProcessorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}


