#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>
#include <limits>
#include <memory>

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
#include <tf2_ros/static_transform_broadcaster.h>   
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>

#include "rbf_cuda.hpp"

// #include <geographic_msgs/msg/geo_point.hpp>
// #include <mvp_msgs/srv/send_waypoints.hpp>

using std::placeholders::_1;
// using std::placeholders::_2;

// ------------------------------------------------------------------------
// Types
// ------------------------------------------------------------------------
struct FullState {
rclcpp::Time stamp;
std::string frame_id;
std::string child_frame_id;

geometry_msgs::msg::Point position;          // x y z
geometry_msgs::msg::Quaternion orientation;  // qx qy qz qw
geometry_msgs::msg::Vector3 lin_vel;         // vx vy vz
geometry_msgs::msg::Vector3 ang_vel;         // wx wy wz

// // Euler (computed from quaternion)
// double roll{0.0}, pitch{0.0}, yaw{0.0};
};

// DOF indices in [x,y,z, roll,pitch,yaw]
enum { X=0, Y=1, Z=2, ROLL=3, PITCH=4, YAW=5 };
constexpr double PI = 3.14159265358979323846;

// ------------------------------------------------------------------------
// Saving the nodes components value in a CSV file, just for check 
// ------------------------------------------------------------------------
// Writes the 12-D uniform grid (ne samples per dimension) over [-1, 1] to CSV.
// Example: ne=3 -> 3^12 = 531,441 rows.
// inline void write_zetta_grid_csv(const std::string& path,
//                                  int ne,
//                                  double lo = -1.0,
//                                  double hi =  1.0)
// {
//     if (ne <= 0) return;
//     const double step = (ne == 1) ? 0.0 : (hi - lo) / double(ne - 1);

//     std::ofstream f(path, std::ios::out | std::ios::trunc);
//     if (!f.good()) return;

//     // header
//     f << "z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11\n";

//     int  idx[12] = {0};
//     bool done = false;
//     while (!done) {
//         for (int d = 0; d < 12; ++d) {
//             const double val = (ne == 1) ? 0.5*(lo+hi) : (lo + idx[d] * step);
//             f << val << (d < 11 ? ',' : '\n');
//         }
//         // increment odometer
//         for (int d = 0; d < 12; ++d) {
//             if (++idx[d] < ne) break;
//             idx[d] = 0;
//             if (d == 11) done = true;
//         }
//     }
// }




// ------------------------------------------------------------------------
// Node
// ------------------------------------------------------------------------
class DataProcessorNode : public rclcpp::Node
{
public:
    DataProcessorNode() : Node("data_processor_node")
    {
         
        tf_buffer_  = std::make_unique<tf2_ros::Buffer>(this->get_clock());        // TF buffer that stores transforms
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);  // TF listener that fills the buffer asynchronously

        global_frame_ = this->declare_parameter<std::string>("global_frame", "mauv_1/world");
        // odom_is_enu_ = this->declare_parameter<bool>("odom_is_enu", true);
        // static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        
        // --- load param ---
        this->declare_parameter<int>("param_int", 1);
        this->get_parameter("param_int", param_int_);
        RCLCPP_INFO(this->get_logger(), "Test param: param_int=%d", param_int_);

        // --- I/O ---
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "odometry/filtered", 
            rclcpp::QoS(rclcpp::KeepLast(10)).reliable(), //rclcpp::SensorDataQoS(),
            std::bind(&DataProcessorNode::odom_callback, this, _1));  

        wpt_pub_   = this->create_publisher<geometry_msgs::msg::PoseStamped>("waypoint", 10); //("~/waypoint", 10)
        wpt6_pub_  = this->create_publisher<std_msgs::msg::Float64MultiArray>("waypoint_6d", 10);
        odom6_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("odom_6d", 10);
        
        odom_pose_world_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("odom_pose_world", 10);


        // --- ODE matrices ---
        init_matrices_();

        init_H2_();

        // --- initial conditions (q_L0, qDot_L0) ---
        const std::array<double,6> q0   = { 0.0, 80.0, -200.0, 0.0, 0.0, 0.0 };
        const std::array<double,6> qd0  = { 80.0, 0.0, 0.0, 0.0, 0.0, 80.0 };
        for (int i=0;i<6;++i) {
            X_waypoint_[i]    = q0[i];
            X_waypoint_[6+i]  = qd0[i];
            p_[i]             = q0[i];
            v_[i]             = qd0[i];
        }

        // --- CSV logs ---
        open_csv_or_warn_(wpt_csv_,  "waypoint_6d.csv", "stamp,x,y,z,roll,pitch,yaw");
        open_csv_or_warn_(odom_csv_, "odom_6d.csv",     "stamp,x,y,z,roll,pitch,yaw");
        last_step_ = this->get_clock()->now();

        // --- zetta grid CSV (computed ONCE at startup) ---
        auto zetta_csv = this->declare_parameter<std::string>("zetta_csv", "zetta_grid.csv");
        zetta_ne_ = this->declare_parameter<int>("zetta_ne", 3);
        // write_zetta_grid_csv(zetta_csv, zetta_ne_);
        // write_zetta_grid_csv(zetta_csv, ne, -1.0, 1.0); // normalizing

        // RBF/CUDA parameters (allow override via ROS params)
        lambda_   = this->declare_parameter<double>("lambda", 1.5); // we should tune the lambda, 50.0

        // compute N = ne^12 and bytes
        uint64_t N = 1;
        for (int i = 0; i < 12; ++i) N *= (uint64_t) zetta_ne_;
        double mb = (double)N * sizeof(float) / (1024.0*1024.0);
        RCLCPP_INFO(this->get_logger(), "RBF: ne=%d  -> N=ne^12=%llu (%.2f MB buffer)",
                    zetta_ne_, (unsigned long long)N, mb);

        // Create the GPU helper (float)
        // rbf_ = std::make_unique<CudaRBF>(zetta_ne_, -100.0f, 100.0f, static_cast<float>(lambda_));
        rbf_ = std::make_unique<CudaRBF>(zetta_ne_, -1.0f,  1.0f,  static_cast<float>(lambda_));

        if (!rbf_ || !rbf_->ready()) {
        RCLCPP_ERROR(this->get_logger(),
            "CUDA RBF allocation failed: ne=%d N=%llu bytes=%zu cudaError=%s",
            zetta_ne_,
            (unsigned long long)(rbf_ ? rbf_->num_points() : 0ULL),
            (size_t)(rbf_ ? rbf_->bytes_S() : 0),
            cudaGetErrorString(rbf_ ? rbf_->last_status() : cudaErrorUnknown));
        return;
        } else {
        RCLCPP_INFO(this->get_logger(), "CUDA RBF ready: N=%llu (ne=%d)",
            (unsigned long long)rbf_->num_points(), zetta_ne_);

        // S_cpu_.resize(rbf_->num_points()); // only if a host copy is needed
        }

        scale_pos_    = this->declare_parameter<double>("scale_pos",    100.0);
        scale_ang_    = this->declare_parameter<double>("scale_ang",    PI);
        scale_lin_    = this->declare_parameter<double>("scale_lin",    2.0);
        scale_angvel_ = this->declare_parameter<double>("scale_angvel", 1.0);

        // --- spin worker thread at 10 Hz ---
        processing_thread_ = std::thread(&DataProcessorNode::processing_loop, this);
    }

    ~DataProcessorNode() override {
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    if (wpt_csv_.is_open())  wpt_csv_.close();
    if (odom_csv_.is_open()) odom_csv_.close();
    }

private:
    // ---------------------------------------------------------------------------
    // Config / state
    // ---------------------------------------------------------------------------
    int    param_int_{1};
    double sim_time_{0.0}; 
    double forcing_amp_{80.0};
    
    int zetta_ne_{3};              // samples per dimension (4 -> 4^12 points)
    double lambda_{50.0};          // Gaussian width parameter
    
    FullState latest_state_;
    std::mutex state_mutex_;  

    // --- RBF / CUDA ---
    std::unique_ptr<CudaRBF> rbf_;    // GPU helper (prefer unique_ptr over raw pointer)
    std::vector<double> S_cpu_;       // optional: hold S on CPU if you need to read/log it
    
    // 6x6 matrices (row-major)
    std::array<double,36> A10_{}; // 6x6, dq   = A10 * qdot
    std::array<double,36> A0_{};  // 6x6, qddot contribution from qdot  (zero)
    std::array<double,36> B0_{};  // 6x6, qddot contribution from q
    // static double angWrap(double a){ return std::atan2(std::sin(a), std::cos(a)); }
    
    // waypoint generator pose/vel
    std::array<double,6>  p_{};           // [x,y,z, roll,pitch,yaw] from generator
    std::array<double,6>  v_{};           // [vx,vy,vz, wx,wy,wz]    from generator
    std::array<double,12> X_waypoint_{};  // [pose(6); vel(6)] from generator
    
    // odom snapshot
    std::array<double,6>  P_odom_{};    // [x,y,z, roll,pitch,yaw] from ODOM
    std::array<double,6>  V_odom_{};    // [vx,vy,vz, wx,wy,wz]    from ODOM
    std::array<double,12> X_odom_{};    // [pose(6); vel(6)]       from ODOM

    // Virtual Control Variables 
    std::array<double,6> z1_{};     // P_odom_ - P_waypoint_  
    std::array<double,36> H1_{};    // 6x6 diagonal gain
    std::array<double,6>  beta_{};  // beta = -H1*z1 + v
    std::array<double,6> z2_{};     // V_odom_ - beta
    
    // Gains and outputs
    std::array<double,36> H2_{};   // 6x6 row-major
    std::array<double,6>  tau_{};  // taw (torque) = F - H2*z2 - z1
    std::array<double,6>  F_{};    // hold F (copy from CUDA f[6], cast to double)

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    // std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
    std::string global_frame_{"mauv_1/world"};
    std::string odom_frame_{""};          // will be filled from the first odom msg
    
    // bool sent_static_tf_{false};
    // bool odom_is_enu_{true};              // ENU odom → NED world? set via param 

    // Normalization scales (physical units → normalized ≈ [-1,1])
    double scale_pos_{100.0};     // meters
    double scale_ang_{PI};        // radians
    double scale_lin_{2.0};       // m/s
    double scale_angvel_{1.0};    // rad/s

    // ROS I/O
    rclcpp::Time last_step_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr wpt_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr wpt6_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr odom6_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr odom_pose_world_pub_;


    // Logging
    std::ofstream wpt_csv_;   // waypoint_6d logger
    std::ofstream odom_csv_;  // odom_6d logger

    // Worker
    std::deque<nav_msgs::msg::Odometry> odom_buffer_;
    std::mutex odom_mutex_;
    std::thread processing_thread_;

    // ---------------------------------------------------------------------------
    // Helpers: math / ODE
    // ---------------------------------------------------------------------------
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

        for (int i = 0; i < 12; ++i) {
            xt[i] = x[i] + 0.5 * dt * k1[i];
        }
        compute_xdot(t + 0.5 * dt, xt, k2);

        for (int i = 0; i < 12; ++i) {
            xt[i] = x[i] + 0.5 * dt * k2[i];
        }
        compute_xdot(t + 0.5 * dt, xt, k3);

        for (int i = 0; i < 12; ++i) {
            xt[i] = x[i] + dt * k3[i];
        }
        compute_xdot(t + dt, xt, k4);

        for (int i = 0; i < 12; ++i) {
            x[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }

    void init_matrices_() {
        // A10 = I for x,y,yaw  (dq = qdot on those axes)
        A10_[X*6+X]     = 1.0;
        A10_[Y*6+Y]     = 1.0;
        A10_[YAW*6+YAW] = 1.0;

        // A0 already zero-initialized

        // B0 = diag([-1, 0, 0, 0, 0, -1]) on x and yaw
        B0_[X*6+X]      = -1.0;
        B0_[YAW*6+YAW]  = -1.0;

        // H1 = diag([720, 900, 0, 0, 0, 1350])   (X, Y, Z, ROLL, PITCH, YAW)
        H1_[X*6 + X]     = 720.0;
        H1_[Y*6 + Y]     = 900.0;
        H1_[YAW*6 + YAW] = 1350.0;
    }

    static void open_csv_or_warn_(std::ofstream& f,
                                    const std::string& path,
                                    const std::string& header) {
        f.open(path, std::ios::out|std::ios::trunc);
        if (!f.good()) {
        return;
        }
        f << header << '\n';
    }

    inline void write_csv_(std::ofstream& f, double t, const std::array<double,6>& s) {
        if (!f.good()) return;
        f << t << ',' << s[0] << ',' << s[1] << ',' << s[2] << ','
        << s[3] << ',' << s[4] << ',' << s[5] << '\n';
    }

    void init_H2_() {
        // default diagonal gains; change as you like or pass via params
        std::vector<double> h2_diag =
            this->declare_parameter<std::vector<double>>(
                "H2_diag", std::vector<double>{1300.0, 1300.0, 1300.0, 1500.0, 1500.0, 1500.0});

        // zero already by default; set diagonal
        for (size_t i = 0; i < 6 && i < h2_diag.size(); ++i) {
            H2_[i*6 + i] = h2_diag[i];
        }
    }

    // ---------------------------------------------------------------------------
    // ROS callbacks / loop
    // ---------------------------------------------------------------------------
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg){
        // // (Optional) quick stamp print
        // RCLCPP_INFO(this->get_logger(), "odom msg t=%.3f",
        //             rclcpp::Time(msg->header.stamp).seconds());

        // Buffer
        {std::lock_guard<std::mutex> lock(odom_mutex_);
            odom_buffer_.push_back(*msg);}

        // Keep a snapshot for use in process thread
        // s.position, s.orientation (and s.roll/pitch/yaw) = pose of mauv_1/base_link expressed in mauv_1/odom.
        FullState s;
        s.stamp          = rclcpp::Time(msg->header.stamp);
        s.frame_id       = msg->header.frame_id;
        s.child_frame_id = msg->child_frame_id;
        s.position       = msg->pose.pose.position;
        s.orientation    = msg->pose.pose.orientation;
        s.lin_vel        = msg->twist.twist.linear;
        s.ang_vel        = msg->twist.twist.angular;

        // tf2::Quaternion q;
        // tf2::fromMsg(s.orientation, q);
        // tf2::Matrix3x3(q).getRPY(s.roll, s.pitch, s.yaw);

        // store snapshot
        { std::lock_guard<std::mutex> lk(state_mutex_); latest_state_ = s; }
        odom_frame_ = msg->header.frame_id;  // use the exact string from the message
        // if (!sent_static_tf_) {
        

        // geometry_msgs::msg::TransformStamped tf;
        // tf.header.stamp = msg->header.stamp;     // static, time doesn’t really matter
        // tf.header.frame_id = global_frame_;      // parent: shared NED world
        // tf.child_frame_id  = odom_frame_;        // child: this robot’s odom frame

        // tf.transform.translation.x = 0.0;
        // tf.transform.translation.y = 0.0;
        // tf.transform.translation.z = 0.0;

        // // Rotation from NED (world) → ENU (odom) if your odom is ENU.
        // if (odom_is_enu_) {
        //     // 180° about (1,1,0)/√2 → maps NED axes to ENU (x↔y, z flips)
        //     tf.transform.rotation.w = 0.0;
        //     tf.transform.rotation.x = 0.70710678;
        //     tf.transform.rotation.y = 0.70710678;
        //     tf.transform.rotation.z = 0.0;
        // } else {
        //     // odom already NED → identity
        //     tf.transform.rotation.w = 1.0;
        //     tf.transform.rotation.x = 0.0;
        //     tf.transform.rotation.y = 0.0;
        //     tf.transform.rotation.z = 0.0;
        // }

        // static_broadcaster_->sendTransform(tf);
        // sent_static_tf_ = true;

        // RCLCPP_INFO(get_logger(),
        //     "Published static TF: %s -> %s (odom_is_enu=%s)",
        //     global_frame_.c_str(), odom_frame_.c_str(), odom_is_enu_ ? "true":"false");
        // }
    }

    void processing_loop(){
        rclcpp::Rate rate(10); // 10 Hz
        while (rclcpp::ok()){
            process_data();
            rate.sleep();
        }
    }    

    void process_data()
    {
        // Drain odometry buffer 
        {std::lock_guard<std::mutex> lock(odom_mutex_);
            while (!odom_buffer_.empty()) {odom_buffer_.pop_front();}
        }

        // Time Step
        rclcpp::Time now = this->get_clock()->now();
        double dt = (last_step_.nanoseconds() == 0) ? 0.1 : (now - last_step_).seconds();
        if (dt <= 0.0 || dt > 1.0) dt = 0.1;
        last_step_ = now;

        // Integrate ODE state (X_waypoint_) with RK4 and unpack to p_, v_
        rk4_step(sim_time_, dt, X_waypoint_);
        sim_time_ += dt;
        for (int i = 0; i < 6; ++i) {p_[i] = X_waypoint_[i];v_[i] = X_waypoint_[6 + i];}
        // p_[5] = angWrap(p_[5]);  // keep yaw in (-pi, pi]

        // Publish waypoint pose
        geometry_msgs::msg::PoseStamped wp;
        wp.header.stamp = now;
        wp.header.frame_id = global_frame_;
        wp.pose.position.x = p_[X];
        wp.pose.position.y = p_[Y];
        wp.pose.position.z = p_[Z];
        tf2::Quaternion q; q.setRPY(p_[ROLL], p_[PITCH], p_[YAW]);
        wp.pose.orientation = tf2::toMsg(q);
        wpt_pub_->publish(wp);

        // Publish 6D array and log CSV (use sim_time_ so time starts at 0)
        std_msgs::msg::Float64MultiArray wp6;
        wp6.data = { p_[X], p_[Y], p_[Z], p_[ROLL], p_[PITCH], p_[YAW] };
        wpt6_pub_->publish(wp6);
        write_csv_(wpt_csv_, sim_time_, p_);

        FullState s_copy;
        { std::lock_guard<std::mutex> lk(state_mutex_); s_copy = latest_state_; }

        // Require that we have published the static TF already
        // if (!sent_static_tf_) {
        //     RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        //         "Static TF %s -> <odom> not published yet. Waiting for first odom message...",
        //         global_frame_.c_str());
        //     return;
        // }

        if (odom_frame_.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Waiting for first odom message to learn odom frame...");
            return;
        }


        geometry_msgs::msg::PoseStamped ps_odom, ps_world;
        ps_odom.header.stamp    = s_copy.stamp;
        ps_odom.header.frame_id = odom_frame_;               // use the exact frame we latched
        ps_odom.pose.position   = s_copy.position;
        ps_odom.pose.orientation= s_copy.orientation;

        try {
            auto T_g_o = tf_buffer_->lookupTransform(
                global_frame_, odom_frame_, s_copy.stamp, tf2::durationFromSec(0.2));
            tf2::doTransform(ps_odom, ps_world, T_g_o);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "TF %s<-%s unavailable (%s). Skipping control step.",
                global_frame_.c_str(), odom_frame_.c_str(), ex.what());
            return;
        }

        // --- Pose in WORLD as [x,y,z,r,p,y] ---
        tf2::Quaternion q_wb;
        tf2::fromMsg(ps_world.pose.orientation, q_wb);
        double roll_w=0, pitch_w=0, yaw_w=0;
        tf2::Matrix3x3(q_wb).getRPY(roll_w, pitch_w, yaw_w);

        std::array<double,6> P_world = {
        ps_world.pose.position.x,
        ps_world.pose.position.y,
        ps_world.pose.position.z,
        roll_w, pitch_w, yaw_w
        };

        geometry_msgs::msg::PoseStamped od_world = ps_world;   // pose of base_link expressed in global_frame_
        od_world.header.frame_id = global_frame_;              // e.g., "mauv_1/world"
        odom_pose_world_pub_->publish(od_world);


        // --- Rotate body-frame twist to WORLD using the same orientation ---
        // (Odometry twist is in child_frame_id = base_link)
        tf2::Matrix3x3 R_wb(q_wb);  // rotation: WORLD <- base_link
        tf2::Vector3 v_b(s_copy.lin_vel.x, s_copy.lin_vel.y, s_copy.lin_vel.z);
        tf2::Vector3 w_b(s_copy.ang_vel.x, s_copy.ang_vel.y, s_copy.ang_vel.z);
        tf2::Vector3 v_w = R_wb * v_b;
        tf2::Vector3 w_w = R_wb * w_b;

        std::array<double,6> V_world = {
        v_w.x(), v_w.y(), v_w.z(),
        w_w.x(), w_w.y(), w_w.z()
        };

        // (Optional) publish/log WORLD pose instead of ODOM for visibility
        std_msgs::msg::Float64MultiArray od6;
        od6.data = { P_world[X], P_world[Y], P_world[Z],
                    P_world[ROLL], P_world[PITCH], P_world[YAW] };
        odom6_pub_->publish(od6);
        write_csv_(odom_csv_, sim_time_, P_world);

        // --- Errors in WORLD (matches waypoint frame) ---
        z1_[X]     = P_world[X]     - p_[X];
        z1_[Y]     = P_world[Y]     - p_[Y];
        z1_[Z]     = P_world[Z]     - p_[Z];
        z1_[ROLL]  = P_world[ROLL]  - p_[ROLL];
        z1_[PITCH] = P_world[PITCH] - p_[PITCH];
        z1_[YAW]   = P_world[YAW]   - p_[YAW];

        // beta = -H1*z1 + v   (v_ is already a waypoint velocity in WORLD)
        double H1z1[6]{};
        mat6_mul_vec(H1_, z1_.data(), H1z1);
        for (int i=0; i<6; ++i) beta_[i] = -H1z1[i] + v_[i];

        // z2 = measured WORLD velocity - beta
        for (int i=0; i<6; ++i) z2_[i] = V_world[i] - beta_[i];


        // Pack WORLD state (physical units)
        std::array<double,12> X_world = {
        P_world[0], P_world[1], P_world[2],
        P_world[3], P_world[4], P_world[5],
        V_world[0], V_world[1], V_world[2],
        V_world[3], V_world[4], V_world[5]
        };

        // --- Normalize ONLY for the RBF input ---
        if (rbf_ && rbf_->ready()) {
            float x_f[12];
            x_f[0] = static_cast<float>(X_world[0] / scale_pos_);
            x_f[1] = static_cast<float>(X_world[1] / scale_pos_);
            x_f[2] = static_cast<float>(X_world[2] / scale_pos_);
            x_f[3] = static_cast<float>(X_world[3] / scale_ang_);
            x_f[4] = static_cast<float>(X_world[4] / scale_ang_);
            x_f[5] = static_cast<float>(X_world[5] / scale_ang_);
            x_f[6] = static_cast<float>(X_world[6] / scale_lin_);
            x_f[7] = static_cast<float>(X_world[7] / scale_lin_);
            x_f[8] = static_cast<float>(X_world[8] / scale_lin_);
            x_f[9] = static_cast<float>(X_world[9] / scale_angvel_);
            x_f[10]= static_cast<float>(X_world[10]/ scale_angvel_);
            x_f[11]= static_cast<float>(X_world[11]/ scale_angvel_);

        // If you want to feed WORLD to CUDA, copy X_world into your buffer instead of X_odom_.
        // for (int i=0;i<12;++i) X_odom_[i] = X_world[i]; // (optional reuse)


        // ***** --- Compute S for current X_odom_ on the GPU --- *****
        // S is being computed in rbf_cuda.cu: S[i] = exp(-d2 * inv_lambda). 
        // Adjust to your actual rbf_ API (e.g., eval_to_host vs eval_device).
        // rbf_->eval_to_host(P_odom_.data(), S_cpu_.data());   // fills S_cpu_ with size rbf_->num_points()
        // if (rbf_ && rbf_->ready()) {
        //     rbf_->compute(X_odom_.data());   // launch kernel, assumes API takes const double[12]

        // // rbf_->download(S_cpu_);  // copies S from GPU to host vector (Avoid this large copy unless necessary)
        // // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        // //   "S: first=%.6e mid=%.6e last=%.6e (N=%zu)",
        // //   S_cpu_.front(), S_cpu_[S_cpu_.size()/2], S_cpu_.back(), S_cpu_.size());
        // }

        // if (rbf_ && rbf_->ready()) {
        // // Build 12D query as float
        // float x_f[12];
        // for (int i=0;i<12;++i) 
        //     x_f[i] = static_cast<float>(X_world_[i]);

        // 1) compute S(x) on GPU
        rbf_->compute_S(x_f);

        // 2) F = [w1^T S, ..., w6^T S]
        float F[6];
        rbf_->dot_F(F);
        for (int i = 0; i < 6; ++i)
            F_[i] = static_cast<double>(F[i]);

        // Check F
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "F=[%.3e, %.3e, %.3e, %.3e, %.3e, %.3e]", F[0],F[1],F[2],F[3],F[4],F[5]);

        // 3) update weights using z2[i] (cast to float), gamma1, sigma
        float z2f[6];
        for (int i=0;i<6;++i) z2f[i] = static_cast<float>(z2_[i]); // z2 in physical units (OK)

        // choose the gains (or params)
        static const float gamma1 = 1e-3f;
        static const float sigma  = 1e-2f;

        rbf_->update_w(z2f, gamma1, sigma);
        }

        double H2z2[6]{};
        mat6_mul_vec(H2_, z2_.data(), H2z2);   // H2 * z2

        for (int i = 0; i < 6; ++i) {
            tau_[i] = F_[i] - H2z2[i] - z1_[i]; // tau = F - H2*z2 - z1
        }
        
        // Check tau_
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "tau = [%.3f %.3f %.3f | %.3f %.3f %.3f]",
            tau_[0], tau_[1], tau_[2], tau_[3], tau_[4], tau_[5]);
            
        // quick check
        // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        //     "P_odom[xyz rpy]=[%.2f %.2f %.2f | %.2f %.2f %.2f]"
        //     "V_odom[xyz rpy]=[%.2f %.2f %.2f | %.2f %.2f %.2f]"
        //     "P_waypoint[xyz rpy]=[%.2f %.2f %.2f | %.2f %.2f %.2f]"
        //     "V_waypoint[xyz rpy]=[%.2f %.2f %.2f | %.2f %.2f %.2f]"
        //     "z1[xyz rpy]=[%.2f %.2f %.2f | %.2f %.2f %.2f]"
        //     "beta = [%.2f %.2f %.2f | %.2f %.2f %.2f]"
        //     "z2 = [%.2f %.2f %.2f | %.2f %.2f %.2f]",
        //     P_odom_[X],P_odom_[Y],P_odom_[Z],P_odom_[ROLL],P_odom_[PITCH],P_odom_[YAW],
        //     V_odom_[X],V_odom_[Y],V_odom_[Z],V_odom_[ROLL],V_odom_[PITCH],V_odom_[YAW],
        //     p_[X],p_[Y],p_[Z],p_[ROLL],p_[PITCH],p_[YAW],
        //     v_[X],v_[Y],v_[Z],v_[ROLL],v_[PITCH],v_[YAW],
        //     z1_[X],z1_[Y],z1_[Z],z1_[ROLL],z1_[PITCH],z1_[YAW],
        //     beta_[X], beta_[Y], beta_[Z], beta_[ROLL], beta_[PITCH], beta_[YAW],
        //     z2_[X], z2_[Y], z2_[Z], z2_[ROLL], z2_[PITCH], z2_[YAW]);

    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DataProcessorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
