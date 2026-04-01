#include <array>
#include <cmath>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <limits>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>

#include "rbf_cuda.hpp"

using std::placeholders::_1;

// ------------------------------------------------------------------------
// Robot's states 
// ------------------------------------------------------------------------
// struct is just like a class but members are public by default.
// We use a struct here because it’s a simple data bundle
// a little box that holds several related pieces of data together so you can pass them around as one thing.
// ------------------------------------------------------------------------
struct FullState { 
  rclcpp::Time stamp;
  std::string frame_id;
  std::string child_frame_id;

  geometry_msgs::msg::Point position;          // x y z
  geometry_msgs::msg::Quaternion orientation;  // qx qy qz qw
  geometry_msgs::msg::Vector3 lin_vel;         // vx vy vz (in child frame)
  geometry_msgs::msg::Vector3 ang_vel;         // wx wy wz (in child frame)
};

// DOF indices in [x,y,z, roll,pitch,yaw]
enum { X=0, Y=1, Z=2, ROLL=3, PITCH=4, YAW=5 };
constexpr double PI = 3.14159265358979323846;

// ------------------------------------------------------------------------
// Node
// ------------------------------------------------------------------------
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
  // ========= TF BUFFER AND LISTENER =========
  void setup_tf_() {   // main thread: listening to topics (odom) and updating TF via the tf2_ros::TransformListener.
    tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  }

  void setup_params_() {
    // Frames: we parametrize the frame we want to work in (the global frame), 
    // but we auto-detect the odometry frame from the messages.
    global_frame_ = this->declare_parameter<std::string>("global_frame", "mauv_1/world");

    // RBF / grid params
    zetta_ne_ = this->declare_parameter<int>("zetta_ne", 3);
    lambda_   = this->declare_parameter<double>("lambda", 50.0);

    scale_pos_    = this->declare_parameter<double>("scale_pos",    100.0);
    scale_ang_    = this->declare_parameter<double>("scale_ang",    PI);
    scale_lin_    = this->declare_parameter<double>("scale_lin",    2.0);
    scale_angvel_ = this->declare_parameter<double>("scale_angvel", 1.0);

    // Initialize timer reference
    last_step_ = this->get_clock()->now();
  }

  void setup_io_() {
    // Subscriptions
    // "odometry/filtered" is a common convention, especially if you use the robot_localization package 
    // (its EKF publishes to /odometry/filtered by default).
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odometry/filtered", // "odometry/filtered" is a topic name, not a frame name
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
      std::bind(&DataProcessorNode::odom_callback, this, _1));
  }

  void setup_math_() {
    // === Controller / model matrices & gains ===
    // (Ensure clean zeros first; safe even if already zero-initialized)
    A10_.fill(0.0);
    A0_.fill(0.0);
    B0_.fill(0.0);
    H1_.fill(0.0);
    H2_.fill(0.0);

    // A10 = I for x,y,yaw (dq = qdot on those axes)
    A10_[X*6 + X]     = 1.0;
    A10_[Y*6 + Y]     = 1.0;
    A10_[YAW*6 + YAW] = 1.0;

    // B0 = diag([-1, 0, 0, 0, 0, -1]) on x and yaw
    B0_[X*6 + X]      = -1.0;
    B0_[YAW*6 + YAW]  = -1.0;

    // H1 gains (X, Y, Z, ROLL, PITCH, YAW)
    H1_[X*6 + X]      = 720.0;
    H1_[Y*6 + Y]      = 900.0;
    H1_[YAW*6 + YAW]  = 1350.0;

    // H2 diagonal
    H2_[X*6 + X]          = 1300.0;
    H2_[Y*6 + Y]          = 1300.0;
    H2_[Z*6 + Z]          = 1300.0;
    H2_[ROLL*6 + ROLL]    = 1500.0;
    H2_[PITCH*6 + PITCH]  = 1500.0;
    H2_[YAW*6 + YAW]      = 1500.0;

    // Initial conditions for the waypoint generator
    const std::array<double,6> q0  = { 0.0, 80.0, -200.0, 0.0, 0.0, 0.0 };
    const std::array<double,6> qd0 = { 80.0, 0.0, 0.0, 0.0, 0.0, 80.0 };
    for (int i=0; i<6; ++i) {
      X_waypoint_[i]    = q0[i];
      X_waypoint_[6+i]  = qd0[i];
      p_[i] = X_waypoint_[i];
      v_[i] = X_waypoint_[6 + i];
    }

    // Create the GPU helper (float)
    rbf_ = std::make_unique<CudaRBF>(zetta_ne_, -1.0f,  1.0f,  static_cast<float>(lambda_));

    // Log memory footprint
    uint64_t N = 1;
    for (int i = 0; i < 12; ++i) N *= static_cast<uint64_t>(zetta_ne_);
    double mb = static_cast<double>(N) * sizeof(float) / (1024.0*1024.0);
    RCLCPP_INFO(this->get_logger(), "RBF: ne=%d -> N=ne^12=%llu (%.2f MB buffer)",
                zetta_ne_, static_cast<unsigned long long>(N), mb);

    if (!rbf_ || !rbf_->ready()) {
      RCLCPP_ERROR(this->get_logger(),
        "CUDA RBF allocation failed: ne=%d N=%llu bytes=%zu cudaError=%s",
        zetta_ne_,
        static_cast<unsigned long long>(rbf_ ? rbf_->num_points() : 0ULL),
        static_cast<size_t>(rbf_ ? rbf_->bytes_S() : 0),
        cudaGetErrorString(rbf_ ? rbf_->last_status() : cudaErrorUnknown));
    } else {
      RCLCPP_INFO(this->get_logger(), "CUDA RBF ready: N=%llu (ne=%d)",
        static_cast<unsigned long long>(rbf_->num_points()), zetta_ne_);
    }
  }

  void start_worker_() {   // creates a second thread: runs processing_loop()
    processing_thread_ = std::thread(&DataProcessorNode::processing_loop, this);
  }

  // ========= ROS callbacks / loop =========
  // cache the newest odometry and learn the frame name so the processing loop can use it.
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Snapshot for processing thread
    FullState s;
    s.stamp          = rclcpp::Time(msg->header.stamp);
    s.frame_id       = msg->header.frame_id;   // e.g., "mauv_1/odom"
    s.child_frame_id = msg->child_frame_id;    // e.g., "mauv_1/base_link"
    s.position       = msg->pose.pose.position;
    s.orientation    = msg->pose.pose.orientation;
    s.lin_vel        = msg->twist.twist.linear;  // in child frame (base_link)
    s.ang_vel        = msg->twist.twist.angular; // in child frame (base_link)

    { std::lock_guard<std::mutex> lk(state_mutex_); latest_state_ = s; } // only one thread touches latest_state_ at a time.

    // Latch the odom frame name as soon as we see it
    odom_frame_ = msg->header.frame_id;
  }

  void processing_loop() {
    rclcpp::Rate rate(10); // 10 Hz
    while (rclcpp::ok()) {
      process_data();
      rate.sleep();
    }
  }

  void process_data()
  {
    // Time step
    rclcpp::Time now = this->get_clock()->now();
    double dt = (last_step_.nanoseconds() == 0) ? 0.1 : (now - last_step_).seconds();
    if (dt <= 0.0 || dt > 1.0) dt = 0.1;
    last_step_ = now;

    // Integrate the simple ODE waypoint generator
    rk4_step(sim_time_, dt, X_waypoint_);
    sim_time_ += dt;
    for (int i = 0; i < 6; ++i) { p_[i] = X_waypoint_[i]; v_[i] = X_waypoint_[6 + i]; }

    // Need an odom frame before we can look up TF
    if (odom_frame_.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Waiting for first odom message to learn odom frame...");
      return;
    }

    // Snapshot state
    FullState s_copy;
    // Locks state_mutex_, copies latest_state_ into s_copy. 
    // Automatically unlocks when the block ends 
    // The braces are just to make the unlock happen immediately after the copy.
    { std::lock_guard<std::mutex> lk(state_mutex_); s_copy = latest_state_; } 

    // Transform pose from ODOM -> WORLD
    geometry_msgs::msg::PoseStamped ps_odom, ps_world;
    ps_odom.header.stamp    = s_copy.stamp;
    ps_odom.header.frame_id = odom_frame_;
    ps_odom.pose.position   = s_copy.position;
    ps_odom.pose.orientation= s_copy.orientation;

    try {
      auto T_g_o = tf_buffer_->lookupTransform(
        global_frame_, odom_frame_, s_copy.stamp, tf2::durationFromSec(0.2));
      tf2::doTransform(ps_odom, ps_world, T_g_o);
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF %s<-%s unavailable (%s). Skipping.",
        global_frame_.c_str(), odom_frame_.c_str(), ex.what());
      return;
    }

    // Pose in WORLD as [x,y,z,r,p,y]
    tf2::Quaternion q_wb;
    tf2::fromMsg(ps_world.pose.orientation, q_wb);
    double roll_w=0, pitch_w=0, yaw_w=0;
    tf2::Matrix3x3(q_wb).getRPY(roll_w, pitch_w, yaw_w);

    // Rotate body-frame twist to WORLD
    tf2::Matrix3x3 R_wb(q_wb);      // WORLD <- base_link
    tf2::Vector3 v_b(s_copy.lin_vel.x, s_copy.lin_vel.y, s_copy.lin_vel.z);
    tf2::Vector3 w_b(s_copy.ang_vel.x, s_copy.ang_vel.y, s_copy.ang_vel.z);
    tf2::Vector3 v_w = R_wb * v_b;
    tf2::Vector3 w_w = R_wb * w_b;

    std::array<double,6> P_world = { ps_world.pose.position.x, ps_world.pose.position.y,
      ps_world.pose.position.z, roll_w, pitch_w, yaw_w};

    std::array<double,6> V_world = { v_w.x(), v_w.y(), v_w.z(), w_w.x(), w_w.y(), w_w.z() };

    // Control errors in WORLD
    z1_[X]     = P_world[X]     - p_[X];
    z1_[Y]     = P_world[Y]     - p_[Y];
    z1_[Z]     = P_world[Z]     - p_[Z];
    z1_[ROLL]  = P_world[ROLL]  - p_[ROLL];
    z1_[PITCH] = P_world[PITCH] - p_[PITCH];
    z1_[YAW]   = P_world[YAW]   - p_[YAW];

    double H1z1[6]{};
    mat6_mul_vec(H1_, z1_.data(), H1z1);

    for (int i=0; i<6; ++i) beta_[i] = -H1z1[i] + v_[i];

    for (int i=0; i<6; ++i) z2_[i] = V_world[i] - beta_[i];

    // Pack WORLD state (physical units) and normalize for RBF
    std::array<double,12> X_world = {
      P_world[0], P_world[1], P_world[2],
      P_world[3], P_world[4], P_world[5],
      V_world[0], V_world[1], V_world[2],
      V_world[3], V_world[4], V_world[5]
    };

    if (rbf_ && rbf_->ready()) {
      float x_f[12];
      x_f[0]  = static_cast<float>(X_world[0]  / scale_pos_);
      x_f[1]  = static_cast<float>(X_world[1]  / scale_pos_);
      x_f[2]  = static_cast<float>(X_world[2]  / scale_pos_);
      x_f[3]  = static_cast<float>(X_world[3]  / scale_ang_);
      x_f[4]  = static_cast<float>(X_world[4]  / scale_ang_);
      x_f[5]  = static_cast<float>(X_world[5]  / scale_ang_);
      x_f[6]  = static_cast<float>(X_world[6]  / scale_lin_);
      x_f[7]  = static_cast<float>(X_world[7]  / scale_lin_);
      x_f[8]  = static_cast<float>(X_world[8]  / scale_lin_);
      x_f[9]  = static_cast<float>(X_world[9]  / scale_angvel_);
      x_f[10] = static_cast<float>(X_world[10] / scale_angvel_);
      x_f[11] = static_cast<float>(X_world[11] / scale_angvel_);

      // 1) compute S(x) on GPU
      rbf_->compute_S(x_f);

      // 2) F = [w1^T S, ..., w6^T S]
      float Ff[6];
      rbf_->dot_F(Ff);
      for (int i = 0; i < 6; ++i) F_[i] = static_cast<double>(Ff[i]);

      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "F=[%.3e, %.3e, %.3e, %.3e, %.3e, %.3e]", Ff[0],Ff[1],Ff[2],Ff[3],Ff[4],Ff[5]);

      // 3) update weights using z2[i]
      float z2f[6];
      for (int i=0;i<6;++i) z2f[i] = static_cast<float>(z2_[i]);
      static const float gamma1 = 1e-3f;
      static const float sigma  = 1e-2f;
      rbf_->update_w(z2f, gamma1, sigma);
    }

    // tau = F - H2*z2 - z1, in world frame
    double H2z2[6]{};
    mat6_mul_vec(H2_, z2_.data(), H2z2);
    for (int i = 0; i < 6; ++i) {
      tau_[i] = F_[i] - H2z2[i] - z1_[i]; 
    }

    // tau in body frame: world -> base is R_bw = R_wb.transpose()
    tf2::Matrix3x3 R_bw = R_wb.transpose();

    tf2::Vector3 F_world(tau_[0], tau_[1], tau_[2]);
    tf2::Vector3 M_world(tau_[3], tau_[4], tau_[5]);

    tf2::Vector3 F_body = R_bw * F_world;
    tf2::Vector3 M_body = R_bw * M_world;

    tau_body_[0]=F_body.x(); tau_body_[1]=F_body.y(); tau_body_[2]=F_body.z();
    tau_body_[3]=M_body.x(); tau_body_[4]=M_body.y(); tau_body_[5]=M_body.z();

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "tau = [%.3f %.3f %.3f | %.3f %.3f %.3f]",
      tau_[0], tau_[1], tau_[2], tau_[3], tau_[4], tau_[5]);

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "tau_body = [%.3f %.3f %.3f | %.3f %.3f %.3f]",
      tau_body_[0], tau_body_[1], tau_body_[2], tau_body_[3], tau_body_[4], tau_body_[5]);
  }

  // ========= Math / utilities =========
  static inline void mat6_mul_vec(const std::array<double,36>& M,
                                  const double v[6], double out[6]) {
    for (int r=0; r<6; ++r) {
      double s=0.0;
      for (int c=0; c<6; ++c) s += M[r*6 + c]*v[c];
      out[r]=s;
    }
  }

  void compute_xdot(double t, const std::array<double,12>& x,
                    std::array<double,12>& xdot) {
    const double* q  = x.data();      // [x,y,z, roll,pitch,yaw]
    const double* qd = x.data()+6;    // [vx,vy,vz, wx,wy,wz]

    double dq[6], A0_qd[6], B0_q[6];
    mat6_mul_vec(A10_, qd, dq);
    mat6_mul_vec(A0_,  qd, A0_qd);
    mat6_mul_vec(B0_,  q,  B0_q);

    // external input u = -[0, 1, 0, 0, 0, 0]^T * forcing_amp_ * cos(t)
    double u[6] = {0.0, -forcing_amp_*std::cos(t), 0.0, 0.0, 0.0, 0.0};

    for (int i=0;i<6;++i) {
      xdot[i]     = dq[i];                     // dq
      xdot[6 + i] = A0_qd[i] + B0_q[i] + u[i]; // dqdot
    }
  }

  void rk4_step(double t, double dt, std::array<double,12>& x) {
    std::array<double,12> k1{}, k2{}, k3{}, k4{}, xt{};

    compute_xdot(t, x, k1);

    for (int i = 0; i < 12; ++i) xt[i] = x[i] + 0.5 * dt * k1[i];
    compute_xdot(t + 0.5 * dt, xt, k2);

    for (int i = 0; i < 12; ++i) xt[i] = x[i] + 0.5 * dt * k2[i];
    compute_xdot(t + 0.5 * dt, xt, k3);

    for (int i = 0; i < 12; ++i) xt[i] = x[i] + dt * k3[i];
    compute_xdot(t + dt, xt, k4);

    for (int i = 0; i < 12; ++i) {
      x[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
  }

  // ========= Members (kept private) =========
  // Config / state
  double sim_time_{0.0};
  double forcing_amp_{80.0};

  int    zetta_ne_;      // samples per dimension (-> ne^12 total points)
  double lambda_;        // Gaussian width parameter

  FullState latest_state_;
  std::mutex state_mutex_;

  // RBF / CUDA
  std::unique_ptr<CudaRBF> rbf_;

  // 6x6 matrices (row-major)
  std::array<double,36> A10_{};
  std::array<double,36> A0_{};
  std::array<double,36> B0_{};

  // Waypoint generator pose/vel
  std::array<double,6>  p_{};
  std::array<double,6>  v_{};
  std::array<double,12> X_waypoint_{};

  // Control variables
  std::array<double,6>  z1_{};
  std::array<double,36> H1_{};
  std::array<double,6>  beta_{};
  std::array<double,6>  z2_{};

  // Gains and outputs
  std::array<double,36> H2_{};
  std::array<double,6>  tau_{};
  std::array<double,6> tau_body_{};
  std::array<double,6>  F_{}; //tau = F - H2*z2 - z1, in world frame

  // Variables related to TF:
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;  // a box that stores all transforms received
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;   // the little worker that listens to /tf and fills the buffer
  std::string global_frame_{"mauv_1/world"};   // 
  std::string odom_frame_{""};   // learned from first odom message

  // Normalization scales (physical units → normalized ≈ [-1,1])
  double scale_pos_{100.0};     // meters
  double scale_ang_{PI};        // radians
  double scale_lin_{2.0};       // m/s
  double scale_angvel_{1.0};    // rad/s

  // ROS I/O
  rclcpp::Time last_step_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  // Worker
  std::thread processing_thread_;
};

// ------------------------------------------------------------------------
int main(int argc , char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DataProcessorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
