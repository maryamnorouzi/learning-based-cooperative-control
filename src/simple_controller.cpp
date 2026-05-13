#include <array>
#include <cctype>
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
#include <unordered_map>
#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/accel_stamped.hpp>
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
// State Types And Control Indices
// ============================================================================
// World frame: mauv_1/world = ENU (East, North, Up)
// X: +X = East, −X = West  (Surge) --> Roll
// Y: +Y = North, −Y = South (Sway) --> Yaw
// Z: +Z = Up, −Z = Down    (Heave) --> Pitch

struct FullState
{ 
  rclcpp::Time stamp;
  std::string frame_id;
  std::string child_frame_id;
  
  // Bundle odom position, orientation, and velocity 
  geometry_msgs::msg::Point position;          // x y z
  geometry_msgs::msg::Quaternion orientation;  // qx qy qz qw
  geometry_msgs::msg::Vector3 lin_vel;         // vx vy vz (expressed in child frame: base_link)
  geometry_msgs::msg::Vector3 ang_vel;         // wx wy wz (expressed in child frame: base_link)
  
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

// Controller 4-D channels:
// index 0 -> surge-axis force Fx (mapped mainly to SURGE_T)
// index 1 -> yaw/sway control channel (stored in the Y slot, mapped mainly to SWAY_BOW_T)
// index 2 -> heave-axis force Fz (allocated across the heave thrusters)
// index 3 -> pitch moment My (realized by differential heave thruster action)

// Thrusters
enum { SURGE_T = 0, HEAVE_BOW_T = 1, HEAVE_STERN_T = 2, SWAY_BOW_T = 3 };

constexpr double PI = 3.14159265358979323846;

// ============================================================================
// DataProcessorNode Controller
// ============================================================================
// This class owns the live ROS 2 controller node: setup, worker loop,
// waypoint tracking, adaptive RBF update, and thruster output.

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
  enum class LearningPhase {
    Learning,
    SteadyRecording,
    Frozen
  };

  enum class KnowledgeSource {
    LocalAverage,
    SwarmAverage
  };

  enum class FormationProfile {
    Training,
    Rotated
  };

  // --------------------------------------------------------------------------
  // TF setup
  // --------------------------------------------------------------------------
  void setup_tf_() {
    tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  }

  // --------------------------------------------------------------------------
  // Body-rate to world Euler-rate transform
  // --------------------------------------------------------------------------
  Eigen::Matrix3d angular_velocity_transform_matrix_(
      const geometry_msgs::msg::TransformStamped& tf) {
    tf2::Quaternion quat;
    quat.setW(tf.transform.rotation.w);
    quat.setX(tf.transform.rotation.x);
    quat.setY(tf.transform.rotation.y);
    quat.setZ(tf.transform.rotation.z);

    Eigen::Vector3d rpy;
    tf2::Matrix3x3(quat).getRPY(rpy.x(), rpy.y(), rpy.z());

    Eigen::Matrix3d transform = Eigen::Matrix3d::Zero();

    double cos_pitch = std::cos(rpy.y());
    double tan_pitch = std::tan(rpy.y());

    if (cos_pitch > -0.0001 && cos_pitch < 0.0001) {
      cos_pitch = 0.0001;
    }

    tan_pitch = std::clamp(tan_pitch, -1000.0, 1000.0);

    transform(0,0) = 1.0;
    transform(0,1) = std::sin(rpy.x()) * tan_pitch;
    transform(0,2) = std::cos(rpy.x()) * tan_pitch;
    transform(1,0) = 0.0;
    transform(1,1) = std::cos(rpy.x());
    transform(1,2) = -std::sin(rpy.x());
    transform(2,0) = 0.0;
    transform(2,1) = std::sin(rpy.x()) / cos_pitch;
    transform(2,2) = std::cos(rpy.x()) / cos_pitch;

    return transform;
  }

  // --------------------------------------------------------------------------
  // Parameter loading
  // --------------------------------------------------------------------------
  void setup_params_() {
    global_frame_ = this->declare_parameter<std::string>("global_frame", ""); 
    offset_x_ = this->declare_parameter<double>("offset_x", 0.0);
    offset_y_ = this->declare_parameter<double>("offset_y", 0.0);
    offset_z_ = this->declare_parameter<double>("offset_z", 0.0);
    offset_pitch_ = this->declare_parameter<double>("offset_pitch", 0.0);
    rotated_offset_x_ = this->declare_parameter<double>("rotated_offset_x", offset_x_);
    rotated_offset_y_ = this->declare_parameter<double>("rotated_offset_y", offset_y_);
    rotated_offset_z_ = this->declare_parameter<double>("rotated_offset_z", offset_z_);
    rotated_offset_pitch_ =
      this->declare_parameter<double>("rotated_offset_pitch", offset_pitch_);
    neighbor_names_ = this->declare_parameter<std::vector<std::string>>(
      "neighbor_names", std::vector<std::string>{});

    zetta_ne_ = this->declare_parameter<int>("zetta_ne", 8);
    lambda_   = this->declare_parameter<double>("lambda", 0.2);
    const int steady_record_stride =
      static_cast<int>(this->declare_parameter<int>("steady_record_stride", 10));
    steady_record_stride_ = static_cast<std::size_t>(
      std::max(1, steady_record_stride));

    const std::string learning_phase_name =
      this->declare_parameter<std::string>("learning_phase", "learning");
    if (!parse_learning_phase_(learning_phase_name, learning_phase_)) {
      RCLCPP_WARN(
        get_logger(),
        "Invalid initial learning_phase '%s'. Falling back to 'learning'.",
        learning_phase_name.c_str());
      learning_phase_ = LearningPhase::Learning;
    }

    const std::string knowledge_source_name =
      this->declare_parameter<std::string>("knowledge_source", "local_average");
    if (!parse_knowledge_source_(knowledge_source_name, knowledge_source_)) {
      RCLCPP_WARN(
        get_logger(),
        "Invalid initial knowledge_source '%s'. Falling back to 'local_average'.",
        knowledge_source_name.c_str());
      knowledge_source_ = KnowledgeSource::LocalAverage;
    }

    const std::string formation_profile_name =
      this->declare_parameter<std::string>("formation_profile", "training");
    if (!parse_formation_profile_(formation_profile_name, formation_profile_)) {
      RCLCPP_WARN(
        get_logger(),
        "Invalid initial formation_profile '%s'. Falling back to 'training'.",
        formation_profile_name.c_str());
      formation_profile_ = FormationProfile::Training;
    }

    // Scaling values
    scale_pos_    = this->declare_parameter<double>("scale_pos",    scale_pos_);
    scale_ang_    = this->declare_parameter<double>("scale_ang",    scale_ang_);
    scale_lin_    = this->declare_parameter<double>("scale_lin",    scale_lin_);
    scale_angvel_ = this->declare_parameter<double>("scale_angvel", scale_angvel_);
    scale_tau_    = this->declare_parameter<double>("scale_tau",    scale_tau_);

    auto lo = this->declare_parameter<std::vector<double>>(
         "rbf_lo6", std::vector<double>(kRbfDim, -1.0));
    auto hi = this->declare_parameter<std::vector<double>>(
         "rbf_hi6", std::vector<double>(kRbfDim,  1.0));

    if (lo.size() != static_cast<std::size_t>(kRbfDim) ||
        hi.size() != static_cast<std::size_t>(kRbfDim)) {
      RCLCPP_ERROR(get_logger(),
        "rbf_lo6 and rbf_hi6 must each have exactly %d elements. Got lo=%zu hi=%zu. Using defaults.",
        kRbfDim,
        lo.size(), hi.size()); 
      lo.assign(kRbfDim, -1.0);
      hi.assign(kRbfDim,  1.0);
    }

    for (int i = 0; i < kRbfDim; ++i) {
      rbf_lo_[i] = static_cast<float>(lo[i]);
      rbf_hi_[i] = static_cast<float>(hi[i]);
    }

    apply_formation_profile_();

    parameter_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&DataProcessorNode::on_parameters_set_, this, _1));
  }
  
  // --------------------------------------------------------------------------
  // ROS interfaces
  // --------------------------------------------------------------------------
  void setup_io_() {
    const auto latched_qos =
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                "odometry/filtered",
                rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
                std::bind(&DataProcessorNode::odom_callback, this, _1)); 

    pub_world_odom_  = this->create_publisher<nav_msgs::msg::Odometry>("world_odom", 10);
    pub_wp_odom_     = this->create_publisher<nav_msgs::msg::Odometry>("waypoint_odom", 10);
    pub_z1_odom_     = this->create_publisher<nav_msgs::msg::Odometry>("z1_odom", 10);
    pub_pi_p_        = this->create_publisher<nav_msgs::msg::Odometry>("pi_p", 10);
    pub_W_           = this->create_publisher<std_msgs::msg::Float32MultiArray>("rbf_weights", 10);
    pub_w_norms_     = this->create_publisher<std_msgs::msg::Float32MultiArray>("rbf_weight_norms", 10);
    pub_local_frozen_wbar_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
      "local_frozen_wbar", latched_qos);
    pub_z1_ang_      = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("z1_ang", 10);
    pub_actual_rpy_  = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("actual_rpy", 10);
    pub_desired_rpy_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("desired_rpy", 10);
    pub_f_comparison_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "f_comparison", 10);

    sub_actual_passive_accel_ = this->create_subscription<geometry_msgs::msg::AccelStamped>(
      "stonefish/passive_accel",
      10,
      std::bind(&DataProcessorNode::actual_passive_accel_callback_, this, _1));

    sub_shared_wbar_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
      "/swarm/shared_wbar",
      latched_qos,
      std::bind(&DataProcessorNode::shared_wbar_callback_, this, _1));

    for (const auto& neighbor_name : neighbor_names_) {
      const std::string topic_name = "/" + neighbor_name + "/rbf_weights";
      auto sub = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        topic_name,
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
        [this, neighbor_name](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
          this->neighbor_weights_callback_(neighbor_name, msg);
        });

      neighbor_weight_subs_.push_back(sub);
      RCLCPP_INFO(get_logger(), "Subscribed to neighbor weights: %s", topic_name.c_str());
    }
  }

  void actual_passive_accel_callback_(
    const geometry_msgs::msg::AccelStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(actual_passive_accel_mutex_);
    actual_passive_accel_[0] = msg->accel.linear.x;
    actual_passive_accel_[1] = msg->accel.linear.y;
    actual_passive_accel_[2] = msg->accel.linear.z;
    actual_passive_accel_[3] = msg->accel.angular.y;
    has_actual_passive_accel_ = true;
  }

  bool compute_average_rbf_output_(std::array<double, 4>& output) {
    if (!(rbf_ && rbf_->ready())) {
      return false;
    }

    const std::uint64_t point_count = rbf_->num_points();
    const std::size_t expected_weight_count =
      static_cast<std::size_t>(4) * static_cast<std::size_t>(point_count);

    std::vector<float> activation(point_count);
    rbf_->copy_S_to_host(activation.data());
    output.fill(0.0);

    if (learning_phase_ == LearningPhase::SteadyRecording) {
      if (steady_sample_count_ == 0 ||
          steady_weight_sum_.size() != expected_weight_count) {
        return false;
      }

      const double inv_sample_count =
        1.0 / static_cast<double>(steady_sample_count_);
      for (int channel = 0; channel < 4; ++channel) {
        const std::size_t offset =
          static_cast<std::size_t>(channel) * static_cast<std::size_t>(point_count);
        double value = 0.0;
        for (std::uint64_t j = 0; j < point_count; ++j) {
          const double averaged_weight =
            steady_weight_sum_[offset + static_cast<std::size_t>(j)] *
            inv_sample_count;
          value += averaged_weight * static_cast<double>(activation[j]);
        }
        output[channel] = value;
      }
      return true;
    }

    std::vector<float> averaged_weights;
    if (learning_phase_ == LearningPhase::Frozen) {
      if (knowledge_source_ == KnowledgeSource::SwarmAverage) {
        std::lock_guard<std::mutex> lk(shared_wbar_mutex_);
        if (!shared_weights_ready_ ||
            shared_frozen_weights_host_.size() != expected_weight_count) {
          return false;
        }
        averaged_weights = shared_frozen_weights_host_;
      } else {
        if (!frozen_weights_ready_ ||
            frozen_weights_host_.size() != expected_weight_count) {
          return false;
        }
        averaged_weights = frozen_weights_host_;
      }
    } else {
      return false;
    }

    for (int channel = 0; channel < 4; ++channel) {
      const std::size_t offset =
        static_cast<std::size_t>(channel) * static_cast<std::size_t>(point_count);
      double value = 0.0;
      for (std::uint64_t j = 0; j < point_count; ++j) {
        value += static_cast<double>(
                   averaged_weights[offset + static_cast<std::size_t>(j)]) *
                 static_cast<double>(activation[j]);
      }
      output[channel] = value;
    }
    return true;
  }

  void publish_f_comparison_(const std::array<double, 4>& estimate) {
    if (!pub_f_comparison_) {
      return;
    }

    std::array<double, 4> actual{};
    {
      std::lock_guard<std::mutex> lock(actual_passive_accel_mutex_);
      if (!has_actual_passive_accel_) {
        return;
      }
      actual = actual_passive_accel_;
    }

    std_msgs::msg::Float64MultiArray msg;
    msg.data = {
      estimate[0], actual[0], estimate[0] - actual[0],
      estimate[1], actual[1], estimate[1] - actual[1],
      estimate[2], actual[2], estimate[2] - actual[2],
      estimate[3], actual[3], estimate[3] - actual[3]
    };

    pub_f_comparison_->publish(msg);
  }

  // --------------------------------------------------------------------------
  // Controller and RBF initialization
  // --------------------------------------------------------------------------
  void setup_math_() {
    A10_.fill(0.0);
    A0_.fill(0.0);
    B0_.fill(0.0);

    A10_[WX*4 + WX]         = 1.0;
    A10_[WY*4 + WY]         = 1.0;
    A10_[WZ*4 + WZ]         = 1.0;
    A10_[WPITCH*4 + WPITCH] = 1.0;

    A0_[WZ*4 + WZ]         = -kd_z;       
    A0_[WPITCH*4 + WPITCH] = -zeta_pitch;

    B0_[WX*4 + WX]         = -wx*wx;         // X'' = -wx^2 X
    B0_[WY*4 + WY]         = -wy*wy;         // Y'' = -wy^2 Y
    B0_[WZ*4 + WZ]         = -kp_z;          // Z'' += -kp_z * Z
    B0_[WPITCH*4 + WPITCH] = -w_pitch;       // Pitch'' += -w_pitch * Pitch

    H1_.fill(0.0);
    H2_.fill(0.0);

    // H1: position -> virtual velocity.
    H1_[WX*4 + WX]         = 1.0;
    H1_[WYAW *4 + WYAW ]   = 1.0;   
    H1_[WZ*4 + WZ]         = 1.0;   
    H1_[WPITCH*4 + WPITCH] = 1.0;    

    // H2: velocity error -> control effort.
    H2_[WX*4 + WX]         = 200.0;
    H2_[WYAW *4 + WYAW ]   = 40.0;  
    H2_[WZ*4 + WZ]         = 300.0;  
    H2_[WPITCH*4 + WPITCH] = 100.0;

    // Initial conditions for the figure-eight waypoint generator.
    const double A = 100.0;   // X amplitude (meters)
    const double B = 50.0;    // Y amplitude (meters)
    const std::array<double,4> q0  = { 0.0, 0.0, z_ref_initial_, 0.0 };
    const std::array<double,4> qd0 = { A*wx, B*wy, 0.0, 0.0};      

    for (int i=0; i<4; ++i) {
      X_waypoint_[i]    = q0[i];
      X_waypoint_[4+i]  = qd0[i];
      p_[i] = X_waypoint_[i];   
      v_[i] = X_waypoint_[4 + i];
    }

    float lo[kRbfDim], hi[kRbfDim];
    for (int i = 0; i < kRbfDim; ++i) {
      lo[i] = rbf_lo_[i];
      hi[i] = rbf_hi_[i];
    } 

    rbf_ = std::make_unique<CudaRBF>(zetta_ne_, lo, hi, (float)lambda_);
    uint64_t N = 1;
    for (int i = 0; i < kRbfDim; ++i) N *= static_cast<uint64_t>(zetta_ne_);
    double mb = static_cast<double>(N) * sizeof(float) / (1024.0*1024.0);
    RCLCPP_INFO(this->get_logger(), "RBF: ne=%d -> N=ne^%d=%llu (%.2f MB buffer)",
                zetta_ne_, kRbfDim, static_cast<unsigned long long>(N), mb);

    if (learning_phase_ == LearningPhase::SteadyRecording) {
      begin_steady_recording_();
    } else if (learning_phase_ == LearningPhase::Frozen) {
      if (knowledge_source_ == KnowledgeSource::SwarmAverage) {
        RCLCPP_INFO(
          get_logger(),
          "Starting in 'frozen' mode with shared swarm knowledge. "
          "Waiting for /swarm/shared_wbar if it has not arrived yet.");
      } else {
        RCLCPP_WARN(
          get_logger(),
          "Initial learning_phase 'frozen' requires knowledge_source="
          "'swarm_average' when reusing saved weights. Starting in "
          "'learning' mode instead.");
        learning_phase_ = LearningPhase::Learning;
        this->set_parameter(rclcpp::Parameter("learning_phase", "learning"));
      }
    }
  }

  // --------------------------------------------------------------------------
  // Worker thread startup
  // --------------------------------------------------------------------------
  void start_worker_() {   
    processing_thread_ = std::thread(&DataProcessorNode::processing_loop, this);
  }

  // --------------------------------------------------------------------------
  // Odometry input
  // --------------------------------------------------------------------------
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    FullState s;
    s.stamp          = rclcpp::Time(msg->header.stamp);
    s.frame_id       = msg->header.frame_id;         // e.g., "mauv_1/odom"
    s.child_frame_id = msg->child_frame_id;          // e.g., "mauv_1/base_link"
    s.position       = msg->pose.pose.position;
    s.orientation    = msg->pose.pose.orientation;   // quaternion form
    s.lin_vel        = msg->twist.twist.linear;      // in body frame (base_link)
    s.ang_vel        = msg->twist.twist.angular;     // in body frame (base_link)

    {    
      std::lock_guard<std::mutex> lk(state_mutex_); 
      latest_state_ = s;
    } 

    odom_frame_ = msg->header.frame_id;

    if (base_frame_.empty()) {
        base_frame_ = msg->child_frame_id;     // "mauv_1/base_link" or just "base_link"
        auto slash = base_frame_.find('/');              
        tf_prefix_ = (slash == std::string::npos) 
                      ? ""   // no prefix present
                      : base_frame_.substr(0, slash); // "mauv_1"
      }

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
    }
  }

  // --------------------------------------------------------------------------
  // Cooperative weight input
  // --------------------------------------------------------------------------
  void neighbor_weights_callback_(
      const std::string& neighbor_name,
      const std_msgs::msg::Float32MultiArray::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lk(neighbor_weights_mutex_);
      neighbor_weights_[neighbor_name] = msg->data;
      neighbor_weight_stamps_[neighbor_name] = this->get_clock()->now();
    }

    const uint64_t expected_size = expected_weight_vector_size_();
    if (msg->data.size() != expected_size) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Neighbor '%s' weight size mismatch: got=%zu expected=%llu",
        neighbor_name.c_str(),
        msg->data.size(),
        static_cast<unsigned long long>(expected_size));
    } else {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "Received neighbor weights from '%s' (size=%zu)",
        neighbor_name.c_str(),
        msg->data.size());
    }

    bool all_neighbors_ready = false;
    {
      std::lock_guard<std::mutex> lk(neighbor_weights_mutex_);
      all_neighbors_ready = (neighbor_weights_.size() == neighbor_names_.size());
    }

    if (all_neighbors_ready) {
      RCLCPP_INFO_ONCE(
        get_logger(),
        "Received at least one rbf_weights message from all configured neighbors.");
    }
  }

  uint64_t expected_weight_vector_size_() const
  {
    uint64_t N = 1;
    for (int i = 0; i < kRbfDim; ++i) {
      N *= static_cast<uint64_t>(zetta_ne_);
    }
    return 4 * N;
  }

  static const char* knowledge_source_to_string_(KnowledgeSource source)
  {
    switch (source) {
      case KnowledgeSource::LocalAverage:
        return "local_average";
      case KnowledgeSource::SwarmAverage:
        return "swarm_average";
    }
    return "local_average";
  }

  static bool parse_knowledge_source_(
      const std::string& source_name,
      KnowledgeSource& source)
  {
    if (source_name == "local_average") {
      source = KnowledgeSource::LocalAverage;
      return true;
    }
    if (source_name == "swarm_average") {
      source = KnowledgeSource::SwarmAverage;
      return true;
    }
    return false;
  }

  static const char* formation_profile_to_string_(FormationProfile profile)
  {
    switch (profile) {
      case FormationProfile::Training:
        return "training";
      case FormationProfile::Rotated:
        return "rotated";
    }
    return "training";
  }

  static bool parse_formation_profile_(
      const std::string& profile_name,
      FormationProfile& profile)
  {
    if (profile_name == "training") {
      profile = FormationProfile::Training;
      return true;
    }
    if (profile_name == "rotated") {
      profile = FormationProfile::Rotated;
      return true;
    }
    return false;
  }

  static const char* learning_phase_to_string_(LearningPhase phase)
  {
    switch (phase) {
      case LearningPhase::Learning:
        return "learning";
      case LearningPhase::SteadyRecording:
        return "steady_recording";
      case LearningPhase::Frozen:
        return "frozen";
    }
    return "learning";
  }

  static bool parse_learning_phase_(
      const std::string& phase_name,
      LearningPhase& phase)
  {
    if (phase_name == "learning") {
      phase = LearningPhase::Learning;
      return true;
    }
    if (phase_name == "steady_recording") {
      phase = LearningPhase::SteadyRecording;
      return true;
    }
    if (phase_name == "frozen") {
      phase = LearningPhase::Frozen;
      return true;
    }
    return false;
  }

  void apply_formation_profile_()
  {
    switch (formation_profile_) {
      case FormationProfile::Training:
        active_offset_x_ = offset_x_;
        active_offset_y_ = offset_y_;
        active_offset_z_ = offset_z_;
        active_offset_pitch_ = offset_pitch_;
        break;

      case FormationProfile::Rotated:
        active_offset_x_ = rotated_offset_x_;
        active_offset_y_ = rotated_offset_y_;
        active_offset_z_ = rotated_offset_z_;
        active_offset_pitch_ = rotated_offset_pitch_;
        break;
    }
  }

  void shared_wbar_callback_(
      const std_msgs::msg::Float32MultiArray::SharedPtr msg)
  {
    const std::size_t expected_size =
      static_cast<std::size_t>(expected_weight_vector_size_());
    if (msg->data.size() != expected_size) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Shared swarm w_bar size mismatch: got=%zu expected=%zu",
        msg->data.size(),
        expected_size);
      return;
    }

    {
      std::lock_guard<std::mutex> lk(shared_wbar_mutex_);
      shared_frozen_weights_host_ = msg->data;
      shared_weights_ready_ = true;
      shared_weights_dirty_ = true;
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 10000,
      "Received shared swarm w_bar (size=%zu).",
      msg->data.size());
  }

  rcl_interfaces::msg::SetParametersResult on_parameters_set_(
      const std::vector<rclcpp::Parameter>& parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto& parameter : parameters) {
      if (parameter.get_name() == "learning_phase") {
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
          result.successful = false;
          result.reason = "learning_phase must be a string.";
          return result;
        }

        LearningPhase requested_phase = LearningPhase::Learning;
        if (!parse_learning_phase_(parameter.as_string(), requested_phase)) {
          result.successful = false;
          result.reason =
            "learning_phase must be one of: learning, steady_recording, frozen.";
          return result;
        }

        std::lock_guard<std::mutex> lk(runtime_update_mutex_);
        pending_learning_phase_ = requested_phase;
        has_pending_learning_phase_ = true;
      }

      if (parameter.get_name() == "knowledge_source") {
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
          result.successful = false;
          result.reason = "knowledge_source must be a string.";
          return result;
        }

        KnowledgeSource requested_source = KnowledgeSource::LocalAverage;
        if (!parse_knowledge_source_(parameter.as_string(), requested_source)) {
          result.successful = false;
          result.reason =
            "knowledge_source must be one of: local_average, swarm_average.";
          return result;
        }

        std::lock_guard<std::mutex> lk(runtime_update_mutex_);
        pending_knowledge_source_ = requested_source;
        has_pending_knowledge_source_ = true;
      }

      if (parameter.get_name() == "formation_profile") {
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
          result.successful = false;
          result.reason = "formation_profile must be a string.";
          return result;
        }

        FormationProfile requested_profile = FormationProfile::Training;
        if (!parse_formation_profile_(parameter.as_string(), requested_profile)) {
          result.successful = false;
          result.reason =
            "formation_profile must be one of: training, rotated.";
          return result;
        }

        std::lock_guard<std::mutex> lk(runtime_update_mutex_);
        pending_formation_profile_ = requested_profile;
        has_pending_formation_profile_ = true;
      }

    }

    return result;
  }

  void begin_steady_recording_()
  {
    const std::size_t weight_count =
      static_cast<std::size_t>(expected_weight_vector_size_());
    steady_weight_sum_.assign(weight_count, 0.0);
    steady_weight_snapshot_.clear();
    frozen_weights_host_.clear();
    steady_sample_count_ = 0;
    steady_record_tick_ = 0;
    frozen_weights_ready_ = false;
    learning_phase_ = LearningPhase::SteadyRecording;

    RCLCPP_INFO(
      get_logger(),
      "Learning phase switched to 'steady_recording'. Recording weight "
      "samples every %zu control step(s).",
      steady_record_stride_);
  }

  void publish_local_frozen_wbar_()
  {
    if (!pub_local_frozen_wbar_ || frozen_weights_host_.empty()) {
      return;
    }

    std_msgs::msg::Float32MultiArray msg;
    msg.layout.dim.resize(2);
    msg.layout.dim[0].label = "output";
    msg.layout.dim[0].size = 4;
    msg.layout.dim[0].stride =
      static_cast<uint32_t>(frozen_weights_host_.size());
    msg.layout.dim[1].label = "rbf";
    msg.layout.dim[1].size =
      static_cast<uint32_t>(frozen_weights_host_.size() / 4);
    msg.layout.dim[1].stride =
      static_cast<uint32_t>(frozen_weights_host_.size() / 4);
    msg.data = frozen_weights_host_;
    pub_local_frozen_wbar_->publish(msg);
  }

  void record_steady_weight_sample_()
  {
    if (learning_phase_ != LearningPhase::SteadyRecording) {
      return;
    }

    if (steady_record_tick_++ % steady_record_stride_ != 0) {
      return;
    }

    const std::size_t weight_count =
      static_cast<std::size_t>(expected_weight_vector_size_());
    if (steady_weight_sum_.size() != weight_count) {
      steady_weight_sum_.assign(weight_count, 0.0);
    }

    steady_weight_snapshot_.resize(weight_count);
    const cudaError_t err = rbf_->download_W(steady_weight_snapshot_.data());
    if (err != cudaSuccess) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Failed to record steady-phase weights: %s",
        cudaGetErrorString(err));
      return;
    }

    for (std::size_t idx = 0; idx < weight_count; ++idx) {
      steady_weight_sum_[idx] +=
        static_cast<double>(steady_weight_snapshot_[idx]);
    }
    ++steady_sample_count_;

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 20000,
      "Steady recording active: %zu sample(s) collected.",
      steady_sample_count_);
  }

  bool freeze_to_average_weights_()
  {
    if (!(rbf_ && rbf_->ready())) {
      RCLCPP_WARN(get_logger(), "Cannot freeze learning before the RBF is ready.");
      return false;
    }

    if (steady_sample_count_ == 0 || steady_weight_sum_.empty()) {
      RCLCPP_WARN(
        get_logger(),
        "Cannot freeze learning: no steady-phase weight samples recorded yet.");
      return false;
    }

    const std::size_t weight_count = steady_weight_sum_.size();
    frozen_weights_host_.resize(weight_count);
    for (std::size_t idx = 0; idx < weight_count; ++idx) {
      frozen_weights_host_[idx] = static_cast<float>(
        steady_weight_sum_[idx] / static_cast<double>(steady_sample_count_));
    }

    const cudaError_t err = rbf_->upload_W(frozen_weights_host_.data());
    if (err != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to upload averaged frozen weights: %s",
        cudaGetErrorString(err));
      return false;
    }

    frozen_weights_ready_ = true;
    learning_phase_ = LearningPhase::Frozen;
    publish_local_frozen_wbar_();

    RCLCPP_INFO(
      get_logger(),
      "Learning phase switched to 'frozen' using %zu averaged steady sample(s).",
      steady_sample_count_);
    return true;
  }

  void apply_shared_wbar_if_ready_()
  {
    if (!(rbf_ && rbf_->ready())) {
      return;
    }

    std::vector<float> shared_weights;
    {
      std::lock_guard<std::mutex> lk(shared_wbar_mutex_);
      if (!shared_weights_ready_) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 10000,
          "Waiting for shared swarm w_bar before applying swarm-average knowledge.");
        return;
      }

      if (!shared_weights_dirty_) {
        return;
      }

      shared_weights = shared_frozen_weights_host_;
      shared_weights_dirty_ = false;
    }

    const std::size_t expected_size =
      static_cast<std::size_t>(expected_weight_vector_size_());
    if (shared_weights.size() != expected_size) {
      RCLCPP_WARN(
        get_logger(),
        "Shared swarm w_bar has unexpected size: got=%zu expected=%zu",
        shared_weights.size(),
        expected_size);
      return;
    }

    const cudaError_t err = rbf_->upload_W(shared_weights.data());
    if (err != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to upload shared swarm w_bar: %s",
        cudaGetErrorString(err));
      return;
    }

    shared_weights_active_ = true;
    RCLCPP_INFO(
      get_logger(),
      "Applied shared swarm w_bar to the local controller.");
  }

  void apply_pending_runtime_updates_()
  {
    LearningPhase requested_phase = learning_phase_;
    KnowledgeSource requested_source = knowledge_source_;
    FormationProfile requested_profile = formation_profile_;
    bool has_pending_phase = false;
    bool has_pending_source = false;
    bool has_pending_profile = false;

    {
      std::lock_guard<std::mutex> lk(runtime_update_mutex_);
      has_pending_phase = has_pending_learning_phase_;
      has_pending_source = has_pending_knowledge_source_;
      has_pending_profile = has_pending_formation_profile_;

      requested_phase = pending_learning_phase_;
      requested_source = pending_knowledge_source_;
      requested_profile = pending_formation_profile_;

      has_pending_learning_phase_ = false;
      has_pending_knowledge_source_ = false;
      has_pending_formation_profile_ = false;
    }

    if (has_pending_profile && requested_profile != formation_profile_) {
      formation_profile_ = requested_profile;
      apply_formation_profile_();
      RCLCPP_INFO(
        get_logger(),
        "Formation profile switched to '%s'.",
        formation_profile_to_string_(formation_profile_));
    }

    if (has_pending_phase && requested_phase != learning_phase_) {
      switch (requested_phase) {
        case LearningPhase::Learning:
          learning_phase_ = LearningPhase::Learning;
          shared_weights_active_ = false;
          RCLCPP_INFO(get_logger(), "Learning phase switched to 'learning'.");
          break;

        case LearningPhase::SteadyRecording:
          begin_steady_recording_();
          break;

        case LearningPhase::Frozen:
          if (!freeze_to_average_weights_()) {
            this->set_parameter(rclcpp::Parameter(
              "learning_phase",
              learning_phase_to_string_(learning_phase_)));
          }
          break;
      }
    }

    if (has_pending_source && requested_source != knowledge_source_) {
      knowledge_source_ = requested_source;
      RCLCPP_INFO(
        get_logger(),
        "Knowledge source switched to '%s'.",
        knowledge_source_to_string_(knowledge_source_));

      if (learning_phase_ == LearningPhase::Frozen) {
        if (knowledge_source_ == KnowledgeSource::LocalAverage) {
          if (frozen_weights_ready_ && !frozen_weights_host_.empty()) {
            const cudaError_t err = rbf_->upload_W(frozen_weights_host_.data());
            if (err != cudaSuccess) {
              RCLCPP_ERROR(
                get_logger(),
                "Failed to restore local frozen w_bar: %s",
                cudaGetErrorString(err));
            } else {
              shared_weights_active_ = false;
            }
          } else {
            RCLCPP_WARN(
              get_logger(),
              "Local frozen w_bar is not ready yet; cannot switch to local_average.");
          }
        } else {
          apply_shared_wbar_if_ready_();
        }
      }
    }

    if (learning_phase_ == LearningPhase::Frozen &&
        knowledge_source_ == KnowledgeSource::SwarmAverage) {
      apply_shared_wbar_if_ready_();
    }
  }

  // --------------------------------------------------------------------------
  // Global frame discovery
  // --------------------------------------------------------------------------
  void try_auto_global_frame_() 
  {
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

    std::vector<std::string> candidates;
    if (!prefix.empty()) {
      candidates = {
        prefix + "/world",
      };
    }
    
    candidates.insert(candidates.end(), {"world"}); 
    
    for (const auto& f : candidates) {
      if (tf_buffer_->canTransform(
            f, odom_frame_, rclcpp::Time(0),
            rclcpp::Duration::from_seconds(0.1))) {
        global_frame_ = f;
        RCLCPP_INFO(get_logger(), 
                    "Auto-selected global_frame: %s", 
                    global_frame_.c_str());
        return;
      }
    }

    if (!odom_frame_.empty()) {
      global_frame_ = odom_frame_;
    }
  }

  // --------------------------------------------------------------------------
  // Worker loop
  // --------------------------------------------------------------------------
  void processing_loop() {
    rclcpp::Rate rate(10); // 10 Hz
    while (rclcpp::ok()) {
      process_data();
      rate.sleep();
    }
  }

  // --------------------------------------------------------------------------
  // Thruster allocation solver
  // --------------------------------------------------------------------------
  bool solve_thruster_forces_ls_(const double tau[4], double f_out[4]) {
    
    // Build augmented matrix A = [B_sub | tau].
    double A[4][5];
    for (int j = 0; j < 4; ++j) {
      A[0][j]  = B_[X][j];     // Fx 
      A[1][j]  = B_[YAW][j];   // Mz 
      A[2][j]  = B_[Z][j];     // Fz 
      A[3][j]  = B_[PITCH][j]; // My (note: index 4)
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
  }

  // --------------------------------------------------------------------------
  // Per-step data types and helpers
  // --------------------------------------------------------------------------

  struct WorldFrameState {
    FullState raw_state;
    std::array<double, 6> pose{};
    std::array<double, 6> twist{};
    tf2::Matrix3x3 rotation_world_from_body;
    Eigen::Matrix3d euler_rate_transform;
  };

  struct HeadingReference {
    double path_heading{0.0};
    double heading_correction{0.0};
    double desired_roll{0.0};
    double desired_pitch{0.0};
    double desired_yaw{0.0};
    double desired_yaw_rate{0.0};
    double cross_track_error{0.0};
  };

  double compute_step_dt_() {
    const rclcpp::Time now = this->get_clock()->now();
    double dt = (last_step_.nanoseconds() == 0)
                  ? 0.1
                  : (now - last_step_).seconds();
    if (dt <= 0.0 || dt > 1.0) {
      dt = 0.1;
    }
    last_step_ = now;
    return dt;
  }

  void update_waypoint_reference_(double dt) {
    rk4_step(sim_time_, dt, X_waypoint_);
    sim_time_ += dt;

    for (int i = 0; i < 4; ++i) {
      p_[i] = X_waypoint_[i];
      v_[i] = X_waypoint_[4 + i];
    }

    p_ref_[WX] = p_[WX] + active_offset_x_;
    p_ref_[WY] = p_[WY] + active_offset_y_;
    p_ref_[WZ] = p_[WZ] + active_offset_z_;
    p_ref_[WPITCH] = p_[WPITCH] + active_offset_pitch_;

    v_ref_local_[WX] = v_[WX];
    v_ref_local_[WY] = v_[WY];
    v_ref_local_[WZ] = v_[WZ];
    v_ref_local_[WPITCH] = v_[WPITCH];
  }

  bool ensure_frames_ready_() {
    if (odom_frame_.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 50000,
        "Waiting for first odom message to learn odom frame...");
      return false;
    }

    if (global_frame_.empty()) {
      try_auto_global_frame_();
      if (global_frame_.empty()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 50000,
          "Still auto-detecting global frame (have odom='%s').",
          odom_frame_.c_str());
        return false;
      }
    }

    RCLCPP_INFO_ONCE(
      get_logger(), "Using global_frame = '%s'", global_frame_.c_str());
    return true;
  }

  FullState copy_latest_state_() {
    std::lock_guard<std::mutex> lk(state_mutex_);
    return latest_state_;
  }

  bool build_world_state_(const FullState& raw_state,
                          WorldFrameState& world_state) {
    world_state.raw_state = raw_state;

    geometry_msgs::msg::PoseStamped pose_odom;
    geometry_msgs::msg::PoseStamped pose_world;
    pose_odom.header.stamp = raw_state.stamp;
    pose_odom.header.frame_id = odom_frame_;
    pose_odom.pose.position = raw_state.position;
    pose_odom.pose.orientation = raw_state.orientation;

    try {
      const auto transform_global_from_odom = tf_buffer_->lookupTransform(
        global_frame_, odom_frame_, raw_state.stamp, tf2::durationFromSec(0.2));
      tf2::doTransform(pose_odom, pose_world, transform_global_from_odom);
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "TF %s<-%s unavailable (%s). Skipping.",
        global_frame_.c_str(), odom_frame_.c_str(), ex.what());
      return false;
    }

    tf2::Quaternion q_world_from_body;
    tf2::fromMsg(pose_world.pose.orientation, q_world_from_body);

    double roll_world = 0.0;
    double pitch_world = 0.0;
    double yaw_world = 0.0;
    tf2::Matrix3x3(q_world_from_body).getRPY(
      roll_world, pitch_world, yaw_world);

    world_state.rotation_world_from_body = tf2::Matrix3x3(q_world_from_body);

    const tf2::Vector3 linear_velocity_body(
      raw_state.lin_vel.x,
      raw_state.lin_vel.y,
      raw_state.lin_vel.z);
    const tf2::Vector3 linear_velocity_world =
      world_state.rotation_world_from_body * linear_velocity_body;

    geometry_msgs::msg::TransformStamped world_from_body_transform;
    world_from_body_transform.transform.rotation = pose_world.pose.orientation;
    world_state.euler_rate_transform =
      angular_velocity_transform_matrix_(world_from_body_transform);

    const Eigen::Vector3d angular_velocity_body(
      raw_state.ang_vel.x,
      raw_state.ang_vel.y,
      raw_state.ang_vel.z);
    const Eigen::Vector3d euler_rates_world =
      world_state.euler_rate_transform * angular_velocity_body;

    world_state.pose = {
      pose_world.pose.position.x,
      pose_world.pose.position.y,
      pose_world.pose.position.z,
      roll_world,
      pitch_world,
      yaw_world
    };

    world_state.twist = {
      linear_velocity_world.x(),
      linear_velocity_world.y(),
      linear_velocity_world.z(),
      euler_rates_world(0),
      euler_rates_world(1),
      euler_rates_world(2)
    };

    return true;
  }

  HeadingReference compute_heading_reference_(
      const std::array<double, 6>& pose_world,
      double dt) {
    HeadingReference reference;
    constexpr double kLookaheadDistance = 5.0;
    const double lookahead_gain = 1.0 / kLookaheadDistance;

    reference.path_heading = wrapToPi(std::atan2(v_[WY], v_[WX]));

    const double ex = pose_world[X] - p_ref_[WX];
    const double ey = pose_world[Y] - p_ref_[WY];
    const double cos_path = std::cos(reference.path_heading);
    const double sin_path = std::sin(reference.path_heading);
    reference.cross_track_error = -sin_path * ex + cos_path * ey;

    reference.heading_correction =
      -std::atan(lookahead_gain * reference.cross_track_error);
    reference.desired_yaw = (std::abs(reference.cross_track_error) > 5e-3)
      ? wrapToPi(reference.path_heading + reference.heading_correction)
      : reference.path_heading;

    reference.desired_roll = 0.0;
    reference.desired_pitch = p_ref_[WPITCH];

    if (previous_desired_yaw_valid_ && dt > 0.0) {
      reference.desired_yaw_rate =
        wrapToPi(reference.desired_yaw - previous_desired_yaw_) / dt;
    }

    previous_desired_yaw_ = reference.desired_yaw;
    previous_desired_yaw_valid_ = true;

    return reference;
  }

  void publish_reference_topics_(const WorldFrameState& world_state,
                                const HeadingReference& reference) {
    if (pub_pi_p_) {
      nav_msgs::msg::Odometry msg;
      msg.header.stamp = world_state.raw_state.stamp;
      msg.header.frame_id = global_frame_;
      msg.child_frame_id = "pi_p";
      msg.pose.pose.position.x = p_ref_[WX];
      msg.pose.pose.position.y = p_ref_[WY];
      msg.pose.pose.position.z = p_ref_[WZ];

      tf2::Quaternion q_heading;
      q_heading.setRPY(0.0, 0.0, reference.desired_yaw);
      msg.pose.pose.orientation = tf2::toMsg(q_heading);
      pub_pi_p_->publish(msg);
    }

    if (pub_actual_rpy_) {
      geometry_msgs::msg::Vector3Stamped msg;
      msg.header.stamp = world_state.raw_state.stamp;
      msg.header.frame_id = global_frame_;
      msg.vector.x = world_state.pose[ROLL];
      msg.vector.y = world_state.pose[PITCH];
      msg.vector.z = world_state.pose[YAW];
      pub_actual_rpy_->publish(msg);
    }

    if (pub_desired_rpy_) {
      geometry_msgs::msg::Vector3Stamped msg;
      msg.header.stamp = world_state.raw_state.stamp;
      msg.header.frame_id = global_frame_;
      msg.vector.x = reference.desired_roll;
      msg.vector.y = reference.desired_pitch;
      msg.vector.z = reference.desired_yaw;
      pub_desired_rpy_->publish(msg);
    }
  }

  void publish_world_topics_(const WorldFrameState& world_state,
                            const HeadingReference& reference) {
    if (pub_world_odom_) {
      nav_msgs::msg::Odometry world_msg;
      world_msg.header.stamp = world_state.raw_state.stamp;
      world_msg.header.frame_id = global_frame_;
      world_msg.child_frame_id = base_frame_;
      world_msg.pose.pose.position.x = world_state.pose[X];
      world_msg.pose.pose.position.y = world_state.pose[Y];
      world_msg.pose.pose.position.z = world_state.pose[Z];

      tf2::Quaternion q_world;
      q_world.setRPY(
        world_state.pose[ROLL],
        world_state.pose[PITCH],
        world_state.pose[YAW]);
      world_msg.pose.pose.orientation = tf2::toMsg(q_world);

      world_msg.twist.twist.linear.x = world_state.twist[X];
      world_msg.twist.twist.linear.y = world_state.twist[Y];
      world_msg.twist.twist.linear.z = world_state.twist[Z];
      world_msg.twist.twist.angular.x = world_state.twist[ROLL];
      world_msg.twist.twist.angular.y = world_state.twist[PITCH];
      world_msg.twist.twist.angular.z = world_state.twist[YAW];
      pub_world_odom_->publish(world_msg);
    }

    if (pub_wp_odom_) {
      nav_msgs::msg::Odometry waypoint_msg;
      waypoint_msg.header.stamp = world_state.raw_state.stamp;
      waypoint_msg.header.frame_id = global_frame_;
      waypoint_msg.child_frame_id = "waypoint";
      waypoint_msg.pose.pose.position.x = p_ref_[WX];
      waypoint_msg.pose.pose.position.y = p_ref_[WY];
      waypoint_msg.pose.pose.position.z = p_ref_[WZ];

      tf2::Quaternion q_waypoint;
      q_waypoint.setRPY(0.0, p_ref_[WPITCH], reference.desired_yaw);
      waypoint_msg.pose.pose.orientation = tf2::toMsg(q_waypoint);

      waypoint_msg.twist.twist.linear.x = v_ref_local_[WX];
      waypoint_msg.twist.twist.linear.y = v_ref_local_[WY];
      waypoint_msg.twist.twist.linear.z = v_ref_local_[WZ];
      waypoint_msg.twist.twist.angular.x = 0.0;
      waypoint_msg.twist.twist.angular.y = v_ref_local_[WPITCH];
      waypoint_msg.twist.twist.angular.z = reference.desired_yaw_rate;
      pub_wp_odom_->publish(waypoint_msg);
    }
  }

  void publish_tracking_error_topics_(
      const WorldFrameState& world_state,
      const HeadingReference& reference,
      double z1_roll,
      double z1_pitch,
      double z1_yaw,
      const std::array<double, 4>& reference_velocity) {
    if (pub_z1_ang_) {
      geometry_msgs::msg::Vector3Stamped msg;
      msg.header.stamp = world_state.raw_state.stamp;
      msg.header.frame_id = global_frame_;
      msg.vector.x = z1_roll;
      msg.vector.y = z1_pitch;
      msg.vector.z = z1_yaw;
      pub_z1_ang_->publish(msg);
    }

    if (pub_z1_odom_) {
      nav_msgs::msg::Odometry z1_msg;
      z1_msg.header.stamp = world_state.raw_state.stamp;
      z1_msg.header.frame_id = global_frame_;
      z1_msg.child_frame_id = "z1";
      z1_msg.pose.pose.position.x = z1_[WX];
      z1_msg.pose.pose.position.y = z1_[WY];
      z1_msg.pose.pose.position.z = z1_[WZ];

      tf2::Quaternion q_error;
      q_error.setRPY(z1_roll, z1_pitch, z1_yaw);
      z1_msg.pose.pose.orientation = tf2::toMsg(q_error);

      z1_msg.twist.twist.linear.x = world_state.twist[X] - reference_velocity[WX];
      z1_msg.twist.twist.linear.y = world_state.twist[Y] - reference_velocity[WY];
      z1_msg.twist.twist.linear.z = world_state.twist[Z] - reference_velocity[WZ];
      z1_msg.twist.twist.angular.x = world_state.twist[ROLL];
      z1_msg.twist.twist.angular.y =
        world_state.twist[PITCH] - reference_velocity[WPITCH];
      z1_msg.twist.twist.angular.z =
        world_state.twist[YAW] - reference.desired_yaw_rate;
      pub_z1_odom_->publish(z1_msg);
    }
  }

  std::array<float, kRbfDim> build_rbf_input_(
      const std::array<double, 12>& world_state_vector) const {
    std::array<float, kRbfDim> rbf_input{};

    // Previous 8-D RBF input:
    // rbf_input[0] = static_cast<float>(z1_[WX] / 100.0); // world_state_vector[X]
    // rbf_input[1] = static_cast<float>(z1_[WY] / 50.0);  // world_state_vector[Y]
    // rbf_input[2] = static_cast<float>(world_state_vector[Z] / 10.0);
    // rbf_input[3] = static_cast<float>(world_state_vector[PITCH] * 5.0);
    // rbf_input[4] = static_cast<float>(world_state_vector[X + 6]);
    // rbf_input[5] = static_cast<float>(world_state_vector[Y + 6]);
    // rbf_input[6] = static_cast<float>(world_state_vector[Z + 6] * 2.0);
    // rbf_input[7] = static_cast<float>(world_state_vector[PITCH + 6] * 10.0);

    rbf_input[0] = static_cast<float>(world_state_vector[Z] / 10.0);
    rbf_input[1] = static_cast<float>(world_state_vector[PITCH] * 5.0);
    rbf_input[2] = static_cast<float>(world_state_vector[X + 6]);
    rbf_input[3] = static_cast<float>(world_state_vector[Y + 6]);
    rbf_input[4] = static_cast<float>(world_state_vector[Z + 6] * 2.0);
    rbf_input[5] = static_cast<float>(world_state_vector[PITCH + 6] * 10.0);
    return rbf_input;
  }

  void publish_weight_norms_() {
    if (!(rbf_ && rbf_->ready() && pub_w_norms_)) {
      return;
    }

    if (++weight_norm_publish_counter_ % 10 != 0) {
      return;
    }

    const uint64_t point_count = rbf_->num_points();
    std::vector<float> weights(4 * point_count);
    if (rbf_->download_W(weights.data()) != cudaSuccess) {
      return;
    }

    double sum_sq[4] = {0.0, 0.0, 0.0, 0.0};
    for (uint64_t j = 0; j < point_count; ++j) {
      for (int i = 0; i < 4; ++i) {
        const float weight = weights[i * point_count + j];
        sum_sq[i] += static_cast<double>(weight) * static_cast<double>(weight);
      }
    }

    std_msgs::msg::Float32MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = "norms";
    msg.layout.dim[0].size = 4;
    msg.layout.dim[0].stride = 4;
    msg.data.resize(4);

    for (int i = 0; i < 4; ++i) {
      msg.data[i] = static_cast<float>(std::sqrt(sum_sq[i]));
    }

    pub_w_norms_->publish(msg);
  }

  void publish_weights_for_neighbors_() {
    if (!(rbf_ && rbf_->ready() && pub_W_)) {
      return;
    }

    if (weight_publish_tick_++ % 10 != 0) {
      return;
    }

    std::vector<float> weights_host;
    rbf_->copy_W_to_host(weights_host);

    std_msgs::msg::Float32MultiArray msg;
    msg.layout.dim.resize(2);
    msg.layout.dim[0].label = "output";
    msg.layout.dim[0].size = 4;
    msg.layout.dim[0].stride =
      4 * static_cast<uint32_t>(rbf_->num_points());
    msg.layout.dim[1].label = "rbf";
    msg.layout.dim[1].size = static_cast<uint32_t>(rbf_->num_points());
    msg.layout.dim[1].stride =
      static_cast<uint32_t>(rbf_->num_points());
    msg.data = std::move(weights_host);
    pub_W_->publish(msg);
  }

  void update_rbf_controller_(const std::array<float, kRbfDim>& rbf_input) {
    if (!(rbf_ && rbf_->ready())) {
      return;
    }

    rbf_->compute_S(rbf_input.data());

    // Debug helper: inspect S(x) and the most active center when needed.
    // {
    //   static int dbg_counter = 0;
    //   if (dbg_counter++ % 50 == 0) {
    //     const std::uint64_t N = rbf_->num_points();
    //     std::vector<float> S(N);
    //     rbf_->copy_S_to_host(S.data());
    //
    //     double minS = 1e300;
    //     double maxS = -1e300;
    //     std::uint64_t idx_max = 0;
    //
    //     for (std::uint64_t i = 0; i < N; ++i) {
    //       const double value = static_cast<double>(S[i]);
    //       if (value < minS) minS = value;
    //       if (value > maxS) {
    //         maxS = value;
    //         idx_max = i;
    //       }
    //     }
    //
    //     float center[kRbfDim];
    //     if (rbf_->download_center(idx_max, center) == cudaSuccess) {
    //       // RCLCPP_INFO(get_logger(),
    //       //   "S: min=%.3e max=%.3e idx_max=%llu center=[%.1f %.1f %.1f %.1f %.1f %.1f]",
    //       //   minS, maxS, (unsigned long long)idx_max,
    //       //   center[0], center[1], center[2], center[3],
    //       //   center[4], center[5]);
    //     }
    //   }
    // }

    std::array<float, 4> learned_output_host{};
    rbf_->dot_F(learned_output_host.data());
    for (int i = 0; i < 4; ++i) {
      F_[i] = static_cast<double>(learned_output_host[i]);
    }

    std::array<float, 4> z2_host{};
    for (int i = 0; i < 4; ++i) {
      z2_host[i] = static_cast<float>(z2_[i]);
    }

    static const float gamma1 = 2.2e-4f;
    static const float gamma2 = 1.0e-4f;
    static const float sigma = 5.0e-1f;

    const uint64_t point_count = rbf_->num_points();
    const size_t expected_weight_count =
      static_cast<size_t>(4) * static_cast<size_t>(point_count);
    std::vector<float> neighbor_sum(expected_weight_count, 0.0f);
    int valid_neighbors = 0;

    {
      std::lock_guard<std::mutex> lk(neighbor_weights_mutex_);
      for (const auto& neighbor_name : neighbor_names_) {
        const auto it = neighbor_weights_.find(neighbor_name);
        if (it == neighbor_weights_.end()) {
          continue;
        }

        const auto& neighbor_weights = it->second;
        if (neighbor_weights.size() != expected_weight_count) {
          continue;
        }

        for (size_t idx = 0; idx < expected_weight_count; ++idx) {
          neighbor_sum[idx] += neighbor_weights[idx];
        }
        ++valid_neighbors;
      }
    }

    if (learning_phase_ != LearningPhase::Frozen) {
      if (valid_neighbors > 0) {
        const cudaError_t coop_err = rbf_->update_w_cooperative(
          z2_host.data(),
          neighbor_sum.data(),
          static_cast<float>(valid_neighbors),
          gamma1,
          gamma2,
          sigma);

        if (coop_err != cudaSuccess) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Cooperative weight update failed (%s). Falling back to local update.",
            cudaGetErrorString(coop_err));
          rbf_->update_w(z2_host.data(), gamma1, sigma);
        } else {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 20000,
            "Cooperative update active with %d neighbor(s).",
            valid_neighbors);
        }
      } else {
        rbf_->update_w(z2_host.data(), gamma1, sigma);
      }
    } else {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 20000,
        "Frozen learning active: reusing averaged RBF weights.");
    }

    if (learning_phase_ == LearningPhase::SteadyRecording) {
      record_steady_weight_sample_();
    }

    std::array<double, 4> averaged_output{};
    if (compute_average_rbf_output_(averaged_output)) {
      publish_f_comparison_(averaged_output);
    } else {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 20000,
        "Waiting for averaged RBF weights before publishing f_comparison.");
    }

    publish_weight_norms_();
  }

  void compute_control_effort_(const WorldFrameState& world_state) {
    double H2z2[4]{};
    mat4_mul_vec(H2_, z2_.data(), H2z2);
    for (int i = 0; i < 4; ++i) {
      tau_[i] = F_[i] - H2z2[i] - z1_[i];
    }

    const tf2::Matrix3x3 rotation_body_from_world =
      world_state.rotation_world_from_body.transpose();
    const tf2::Vector3 force_world(tau_[WX], tau_[WY], tau_[WZ]);
    const tf2::Vector3 force_body = rotation_body_from_world * force_world;
    const Eigen::Vector3d moment_world(0.0, tau_[WPITCH], 0.0);
    const Eigen::Vector3d moment_body =
      world_state.euler_rate_transform.inverse() * moment_world;

    tau_body_[WX] = force_body.x();
    tau_body_[WY] = force_body.y();
    tau_body_[WZ] = force_body.z();
    tau_body_[WPITCH] = moment_body.y();
  }

  bool ensure_allocation_ready_() {
    if (alloc_ready_) {
      return true;
    }

    try {
      (void)compute_allocation_from_tf_();
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Alloc TF lookup failed: %s", ex.what());
      return false;
    }

    return alloc_ready_;
  }

  bool update_thruster_commands_() {
    if (!ensure_allocation_ready_()) {
      return false;
    }

    std::array<double, 4> thruster_solution{};
    if (solve_thruster_forces_ls_(tau_body_.data(), thruster_solution.data())) {
      for (int i = 0; i < 4; ++i) {
        thr_force_[i] = thruster_solution[i];
      }
    } else {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Allocation solve failed (BᵀB near singular).");
      for (int i = 0; i < 4; ++i) {
        thr_force_[i] = 0.0;
      }
    }

    for (int i = 0; i < 4; ++i) {
      thr_cmd_[i] = force_to_cmd_(thr_force_[i], i);
    }

    return true;
  }

  void log_controller_state_(const WorldFrameState& world_state,
                            const HeadingReference& reference,
                            const std::array<float, kRbfDim>& rbf_input) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 20000,
      "learning_phase=%s steady_samples=%zu knowledge_source=%s formation_profile=%s",
      learning_phase_to_string_(learning_phase_),
      steady_sample_count_,
      knowledge_source_to_string_(knowledge_source_),
      formation_profile_to_string_(formation_profile_));

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 20000,
      "X_d=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
      p_ref_[WX], p_ref_[WY], reference.desired_yaw, p_ref_[WZ], p_ref_[WPITCH],
      v_ref_local_[WX], v_ref_local_[WY], reference.desired_yaw_rate,
      v_ref_local_[WZ], v_ref_local_[WPITCH]);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 20000,
      "X_w=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
      world_state.pose[X], world_state.pose[Y], world_state.pose[YAW],
      world_state.pose[Z], world_state.pose[PITCH],
      world_state.twist[X], world_state.twist[Y], world_state.twist[YAW],
      world_state.twist[Z], world_state.twist[PITCH]);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 20000,
      "x_f=[%.3f %.3f %.3f %.3f %.3f %.3f]",
      rbf_input[0], rbf_input[1], rbf_input[2],
      rbf_input[3], rbf_input[4], rbf_input[5]);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 20000,
      "z1=[%.3e, %.3e, %.3e, %.3e]",
      z1_[0], z1_[1], z1_[2], z1_[3]);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 20000,
      "z2=[%.3f, %.3f, %.3f, %.3f]",
      z2_[0], z2_[1], z2_[2], z2_[3]);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 20000,
      "beta=[%.3e, %.3e, %.3e, %.3e]",
      beta_[0], beta_[1], beta_[2], beta_[3]);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 20000,
      "F=[%.3e, %.3e, %.3e, %.3e]",
      F_[0], F_[1], F_[2], F_[3]);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 20000,
      "tau=[%.3f %.3f %.3f %.3f]",
      tau_[0], tau_[1], tau_[2], tau_[3]);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 20000,
      "tau_body=[%.3f %.3f %.3f %.3f]",
      tau_body_[0], tau_body_[1], tau_body_[2], tau_body_[3]);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 20000,
      "thr_cmd=[%.2f %.2f %.2f %.2f]",
      thr_cmd_[0], thr_cmd_[1], thr_cmd_[2], thr_cmd_[3]);
  }

  void publish_thruster_commands_() {
    if (!pub_heave_bow_) {
      return;
    }

    std_msgs::msg::Float64 msg;
    msg.data = thr_cmd_[SURGE_T];
    pub_surge_->publish(msg);
    msg.data = thr_cmd_[HEAVE_BOW_T];
    pub_heave_bow_->publish(msg);
    msg.data = thr_cmd_[HEAVE_STERN_T];
    pub_heave_stern_->publish(msg);
    msg.data = thr_cmd_[SWAY_BOW_T];
    pub_sway_bow_->publish(msg);
  }

  // --------------------------------------------------------------------------
  // Control step
  // --------------------------------------------------------------------------
  void process_data() {
    apply_pending_runtime_updates_();

    const double dt = compute_step_dt_();
    update_waypoint_reference_(dt);

    if (!ensure_frames_ready_()) {
      return;
    }

    const FullState raw_state = copy_latest_state_();
    WorldFrameState world_state;
    if (!build_world_state_(raw_state, world_state)) {
      return;
    }

    const HeadingReference reference =
      compute_heading_reference_(world_state.pose, dt);
    publish_reference_topics_(world_state, reference);
    publish_world_topics_(world_state, reference);

    const double z1_roll =
      wrapToPi(world_state.pose[ROLL] - reference.desired_roll);
    const double z1_pitch =
      wrapToPi(world_state.pose[PITCH] - reference.desired_pitch);
    const double z1_yaw =
      wrapToPi(world_state.pose[YAW] - reference.desired_yaw);

    z1_[WX] = world_state.pose[X] - p_ref_[WX];
    z1_[WY] = world_state.pose[Y] - p_ref_[WY];
    z1_[WZ] = world_state.pose[Z] - p_ref_[WZ];
    z1_[WPITCH] = wrapToPi(world_state.pose[PITCH] - p_ref_[WPITCH]);

    double H1z1[4]{};
    mat4_mul_vec(H1_, z1_.data(), H1z1);

    const std::array<double, 4> reference_velocity = {
      v_ref_local_[WX],
      v_ref_local_[WY],
      v_ref_local_[WZ],
      v_ref_local_[WPITCH]
    };

    for (int i = 0; i < 4; ++i) {
      beta_[i] = -H1z1[i] + reference_velocity[i];
    }

    z2_[WX] = world_state.twist[X] - beta_[WX];
    z2_[WYAW] = world_state.twist[Y] - beta_[WY];
    z2_[WZ] = world_state.twist[Z] - beta_[WZ];
    z2_[WPITCH] = world_state.twist[PITCH] - beta_[WPITCH];

    publish_tracking_error_topics_(
      world_state,
      reference,
      z1_roll,
      z1_pitch,
      z1_yaw,
      reference_velocity);

    const std::array<double, 12> world_state_vector = {
      world_state.pose[X],
      world_state.pose[Y],
      world_state.pose[Z],
      world_state.pose[ROLL],
      world_state.pose[PITCH],
      world_state.pose[YAW],
      world_state.twist[X],
      world_state.twist[Y],
      world_state.twist[Z],
      world_state.twist[ROLL],
      world_state.twist[PITCH],
      world_state.twist[YAW]
    };

    const std::array<float, kRbfDim> rbf_input =
      build_rbf_input_(world_state_vector);
    update_rbf_controller_(rbf_input);
    publish_weights_for_neighbors_();

    compute_control_effort_(world_state);
    if (!update_thruster_commands_()) {
      return;
    }

    log_controller_state_(world_state, reference, rbf_input);
    publish_thruster_commands_();
  }

  // --------------------------------------------------------------------------
  // Thruster command mapping
  // --------------------------------------------------------------------------
  inline double force_to_cmd_(double T, int thruster_id) const
  {
    const double c = (thruster_id == SURGE_T) ? thrust_coeff_surge_ : thrust_coeff_other_;

    if (c <= 0.0) {
      return 0.0;
    }
    double a = std::abs(T) / c;
    double u = std::sqrt(a);
    if (u > 1.0) {
      u = 1.0;
    }
    return std::copysign(u, T);
  }

  // --------------------------------------------------------------------------
  // Math utilities
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

  // Z reference with dwell + smooth moves.
  inline void z_pitch_ref(double t, double &z_ref, double &dz_ref,
                          double &pitch_ref, double &dpitch_ref) const
  {
    const double z_min = z_mid - z_amp;   // -> -3
    const double z_max = z_mid + z_amp;   // -> -1
    const double dz = (z_max - z_min);


    const double T_hold = 200.0;   // seconds at top/bottom (tune)
    const double T_move = 400.0;   // seconds to move between (tune: bigger => smaller vz)

    const double Tcycle = 2*T_hold + 2*T_move;
    double cycle_time = std::fmod(t, Tcycle);
    if (cycle_time < 0.0) cycle_time += Tcycle;

    z_ref      = z_min; 
    dz_ref     = 0.0;
    pitch_ref  = 0.0; 
    dpitch_ref = 0.0;

    auto move = [&](double s01, bool up) {
      double p  = smooth5(s01);
      double dp = smooth5_d1(s01);

      z_ref = up ? (z_min + dz*p) : (z_max - dz*p);

      const double sign = up ? +1.0 : -1.0;
      dz_ref = sign * dz * (dp / T_move);
    };

    if (cycle_time < T_hold) {
      z_ref = z_min;
    } else if (cycle_time < T_hold + T_move) {
      move((cycle_time - T_hold)/T_move, true);
    } else if (cycle_time < T_hold + T_move + T_hold) {
      z_ref = z_max;
    } else {
      move((cycle_time - (T_hold + T_move + T_hold))/T_move, false);
    }

    // Pitch tracking is currently disabled.
    pitch_ref = 0.0;
    dpitch_ref = 0.0;
  }

  // --------------------------------------------------------------------------
  // Thruster allocation matrix from TF
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

    return true;
  }

  // --------------------------------------------------------------------------
  // Waypoint generator dynamics
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
  // Waypoint integrator
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
  // Persistent node state
  // --------------------------------------------------------------------------

  // Lifecycle and timing
  double sim_time_{0.0};
  rclcpp::Time last_step_;
  std::thread processing_thread_;

  // Frame and TF state
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string global_frame_{""};
  std::string odom_frame_{""};
  std::string base_frame_{};
  std::string tf_prefix_{};

  // Configuration parameters
  double offset_x_{0.0};
  double offset_y_{0.0};
  double offset_z_{0.0};
  double offset_pitch_{0.0};
  double rotated_offset_x_{0.0};
  double rotated_offset_y_{0.0};
  double rotated_offset_z_{0.0};
  double rotated_offset_pitch_{0.0};
  double active_offset_x_{0.0};
  double active_offset_y_{0.0};
  double active_offset_z_{0.0};
  double active_offset_pitch_{0.0};
  std::vector<std::string> neighbor_names_;

  double scale_pos_{1.0};
  double scale_ang_{1.0};
  double scale_lin_{1.0};
  double scale_angvel_{1.0};
  double scale_tau_{0.5};
  std::size_t steady_record_stride_{10};

  std::array<float, kRbfDim> rbf_lo_{{-1, -1, -1, -1, -1, -1}};
  std::array<float, kRbfDim> rbf_hi_{{1, 1, 1, 1, 1, 1}};

  // Waypoint parameters
  const double wx         = 0.005;
  const double wy         = 2.0 * wx;
  const double z_mid      = -2.0;   // center of [-3, -1]
  const double z_amp      = 1.0;    // amplitude -> [-3, -1]

  // Z and pitch dynamics in the waypoint generator.
  const double kd_z       = 1.0;
  const double zeta_pitch = 4.0;

  const double kp_z       = 0.25;
  const double w_pitch    = 4.0;

  const double z_ref_initial_ = z_mid;
  double thrust_coeff_surge_{99.12};
  double thrust_coeff_other_{57.26};

  // Waypoint generator state
  std::array<double, 16> A10_{};
  std::array<double, 16> A0_{};
  std::array<double, 16> B0_{};
  std::array<double, 8> X_waypoint_{};
  std::array<double, 4> p_{};
  std::array<double, 4> v_{};
  std::array<double, 4> p_ref_{};
  std::array<double, 4> v_ref_local_{};

  // Adaptive controller state
  int zetta_ne_;                 // samples per dimension (-> ne^6 total points)
  double lambda_;                // Gaussian width parameter
  std::unique_ptr<CudaRBF> rbf_;
  std::array<double, 16> H1_{};
  std::array<double, 16> H2_{};
  std::array<double, 4> z1_{};
  std::array<double, 4> beta_{};
  std::array<double, 4> z2_{};
  std::array<double, 4> F_{};
  std::array<double, 4> tau_{};       // tau = F - H2*z2 - z1, in world frame
  std::array<double, 4> tau_body_{};
  LearningPhase learning_phase_{LearningPhase::Learning};
  KnowledgeSource knowledge_source_{KnowledgeSource::LocalAverage};
  FormationProfile formation_profile_{FormationProfile::Training};
  LearningPhase pending_learning_phase_{LearningPhase::Learning};
  KnowledgeSource pending_knowledge_source_{KnowledgeSource::LocalAverage};
  FormationProfile pending_formation_profile_{FormationProfile::Training};
  bool has_pending_learning_phase_{false};
  bool has_pending_knowledge_source_{false};
  bool has_pending_formation_profile_{false};
  std::mutex runtime_update_mutex_;
  std::vector<double> steady_weight_sum_;
  std::vector<float> steady_weight_snapshot_;
  std::vector<float> frozen_weights_host_;
  std::vector<float> shared_frozen_weights_host_;
  std::size_t steady_sample_count_{0};
  std::size_t steady_record_tick_{0};
  bool frozen_weights_ready_{false};
  bool shared_weights_ready_{false};
  bool shared_weights_dirty_{false};
  bool shared_weights_active_{false};
  std::mutex shared_wbar_mutex_;
  double previous_desired_yaw_{0.0};
  bool previous_desired_yaw_valid_{false};

  // Allocation and actuator state
  std::array<std::string, 4> thr_links_{
    "surge_thruster_link",
    "heave_bow_thruster_link",
    "heave_stern_thruster_link",
    "sway_bow_thruster_link"
  };
  double B_[6][4]{};
  bool alloc_ready_{false};
  std::array<double, 4> thr_force_{};
  std::array<double, 4> thr_cmd_{};

  // Cached incoming data
  FullState latest_state_;
  std::mutex state_mutex_;
  std::array<double, 4> actual_passive_accel_{};
  std::mutex actual_passive_accel_mutex_;
  bool has_actual_passive_accel_{false};
  std::unordered_map<std::string, std::vector<float>> neighbor_weights_;
  std::unordered_map<std::string, rclcpp::Time> neighbor_weight_stamps_;
  std::mutex neighbor_weights_mutex_;

  // ROS subscriptions
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
    parameter_callback_handle_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::AccelStamped>::SharedPtr
    sub_actual_passive_accel_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr>
    neighbor_weight_subs_;

  // ROS publishers: thruster commands
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_heave_bow_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_heave_stern_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_surge_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_sway_bow_;

  // ROS publishers: visualization and telemetry
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_world_odom_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_wp_odom_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_z1_odom_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_pi_p_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr
    pub_actual_rpy_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr
    pub_desired_rpy_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr pub_z1_ang_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
    pub_f_comparison_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_W_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_w_norms_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr
    pub_local_frozen_wbar_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
    sub_shared_wbar_;

  // Diagnostics and publish throttles
  int weight_norm_publish_counter_{0};
  std::size_t weight_publish_tick_{0};
};

// ============================================================================
// ROS 2 Entry Point
// ============================================================================

int main(int argc , char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DataProcessorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
