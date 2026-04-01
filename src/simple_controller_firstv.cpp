#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
// #include <geographic_msgs/msg/geo_point.hpp>
#include <std_msgs/msg/header.hpp>
// #include <mvp_msgs/srv/send_waypoints.hpp>

#include <mutex>
#include <deque>
#include <thread>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>  // fromMsg/toMsg

#include <array>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cmath>
#include <geometry_msgs/msg/vector3.hpp>

using std::placeholders::_1;
// using std::placeholders::_2;




// Add a small state struct and a mutex
struct FullState {
  rclcpp::Time stamp;
  std::string frame_id;
  std::string child_frame_id;

  geometry_msgs::msg::Point position;       // x y z
  geometry_msgs::msg::Quaternion orientation;// qx qy qz qw

  geometry_msgs::msg::Vector3 lin_vel;      // vx vy vz
  geometry_msgs::msg::Vector3 ang_vel;      // wx wy wz

// Euler (computed from quaternion)
  double roll{0.0}, pitch{0.0}, yaw{0.0};
};


constexpr double PI = 3.14159265358979323846;

class DataProcessorNode : public rclcpp::Node
{
public:
    DataProcessorNode() : Node("data_processor_node")
    {
        // load param
        this->declare_parameter<int>("param_int", 1);
        this->get_parameter("param_int", param_int_);
        RCLCPP_INFO(this->get_logger(), "Test param: param_int=%d", param_int_);


        // Odometry subscriber
        // odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        //     "~/odom", 10, std::bind(&DataProcessorNode::odom_callback, this, _1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "odometry/filtered", rclcpp::QoS(rclcpp::KeepLast(10)).reliable(), //rclcpp::SensorDataQoS(),
            std::bind(&DataProcessorNode::odom_callback, this, _1));

        // // SendWaypoints service
        // waypoint_srv_ = this->create_service<mvp_msgs::srv::SendWaypoints>(
        //     "~/set_waypoint", std::bind(&DataProcessorNode::waypoint_callback, this, _1, _2));
        
        
        
        // --- init trajectory + publisher ---
        p_[0] = radius_;  // x
        p_[1] = 0.0;      // y
        p_[2] = z0_;      // z
        p_[3] = 0.0; p_[4] = 0.0; p_[5] = PI / 2.0;  // roll,pitch,yaw

        v_[0] = 0.0;                    // vx
        v_[1] = omega_ * radius_;       // vy
        v_[2] = 0.0;                    // vz
        v_[3] = 0.0; v_[4] = 0.0; v_[5] = omega_;  // wx,wy,wz

        wpt_pub_   = this->create_publisher<geometry_msgs::msg::PoseStamped>("waypoint", 10); //("~/waypoint", 10)
        last_step_ = this->get_clock()->now();


        // Start the processing thread
        processing_thread_ = std::thread(&DataProcessorNode::processing_loop, this);
    }

    ~DataProcessorNode()
    {
        processing_thread_.join(); // Ensure clean shutdown
    }

private:

    // param
    int param_int_;

    FullState latest_state_;
    std::mutex state_mutex_;


    // --- trajectory state & publisher ---
    double omega_{0.2};             // rad/s
    double radius_{5.0};            // m
    double z0_{0.0};                // m

    std::array<double,6> p_{};      // [x,y,z, roll,pitch,yaw]
    std::array<double,6> v_{};      // [vx,vy,vz, wx,wy,wz]

    rclcpp::Time last_step_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr wpt_pub_;

    // Odometry buffer
    std::deque<nav_msgs::msg::Odometry> odom_buffer_;
    std::mutex odom_mutex_;

    // // SendWaypoints buffer
    // std::deque<mvp_msgs::srv::SendWaypoints::Request> wpt_buffer_;
    // std::mutex wpt_mutex_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    // rclcpp::Service<mvp_msgs::srv::SendWaypoints>::SharedPtr waypoint_srv_;

    std::thread processing_thread_;

    // void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    // {
    //     // Quick sanity print: proves the callback is being called
    //     RCLCPP_INFO(this->get_logger(), "odom msg t=%.3f",
    //                 rclcpp::Time(msg->header.stamp).seconds());

    //     {
    //         std::lock_guard<std::mutex> lock(odom_mutex_);
    //         odom_buffer_.push_back(*msg);
    //     }

    //     {
    //         std::lock_guard<std::mutex> lock(odom_mutex_);
    //         odom_buffer_.push_back(*msg);
    //         // RCLCPP_INFO(this->get_logger(), "Received odom");
    //     }
        
    //     FullState s;
    //     s.stamp = rclcpp::Time(msg->header.stamp);
    //     s.frame_id = msg->header.frame_id;
    //     s.child_frame_id = msg->child_frame_id;

    //     s.position    = msg->pose.pose.position;
    //     s.orientation = msg->pose.pose.orientation;
    //     s.lin_vel     = msg->twist.twist.linear;
    //     s.ang_vel     = msg->twist.twist.angular;

    //     // Quaternion -> RPY
    //     tf2::Quaternion q;
    //     tf2::fromMsg(s.orientation, q);
    //     tf2::Matrix3x3(q).getRPY(s.roll, s.pitch, s.yaw);

    //     {
    //         std::lock_guard<std::mutex> lk(state_mutex_);
    //         latest_state_ = std::move(s);
    //     }

    //     // Example print with everything
    //     RCLCPP_INFO_THROTTLE(
    //         this->get_logger(), *this->get_clock(), 1000,
    //         "t=%.3f frame=%s child=%s | pos(%.2f,%.2f,%.2f) | q(%.2f,%.2f,%.2f,%.2f) | "
    //         "rpy(%.2f,%.2f,%.2f) | v(%.2f,%.2f,%.2f) | w(%.2f,%.2f,%.2f)",
    //         rclcpp::Time(msg->header.stamp).seconds(),
    //         msg->header.frame_id.c_str(), msg->child_frame_id.c_str(),
    //         msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z,
    //         msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
    //         msg->pose.pose.orientation.z, msg->pose.pose.orientation.w,
    //         latest_state_.roll, latest_state_.pitch, latest_state_.yaw,
    //         msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z,
    //         msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z);
        
    // }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // sanity print
        RCLCPP_INFO(this->get_logger(), "odom msg t=%.3f",
                    rclcpp::Time(msg->header.stamp).seconds());

        // push once
        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            odom_buffer_.push_back(*msg);
        }

        // fill latest_state_
        FullState s;
        s.stamp = rclcpp::Time(msg->header.stamp);
        s.frame_id = msg->header.frame_id;
        s.child_frame_id = msg->child_frame_id;
        s.position    = msg->pose.pose.position;
        s.orientation = msg->pose.pose.orientation;
        s.lin_vel     = msg->twist.twist.linear;
        s.ang_vel     = msg->twist.twist.angular;

        tf2::Quaternion q;
        tf2::fromMsg(s.orientation, q);
        tf2::Matrix3x3(q).getRPY(s.roll, s.pitch, s.yaw);

        // print from the local snapshot
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "ODOM pos(%.2f, %.2f, %.2f) ODOM rpy(%.2f, %.2f, %.2f) ODOM linear velocity(%.2f, %.2f, %.2f) ODOM angular velocity(%.2f, %.2f, %.2f)",
            s.position.x, s.position.y, s.position.z, s.roll, s.pitch, s.yaw, 
            s.lin_vel.x, s.lin_vel.y, s.lin_vel.z, s.ang_vel.x, s.ang_vel.y, s.ang_vel.z);

        // store snapshot
        { std::lock_guard<std::mutex> lk(state_mutex_); latest_state_ = s; }

        // {
        //     std::lock_guard<std::mutex> lk(state_mutex_);
        //     latest_state_ = std::move(s);
        // }

        // RCLCPP_INFO(this->get_logger(),
        //     "ODOM pos(%.2f, %.2f, %.2f) rpy(%.2f, %.2f, %.2f)",
        //     latest_state_.position.x, latest_state_.position.y, latest_state_.position.z,
        //     latest_state_.roll, latest_state_.pitch, latest_state_.yaw);


        // RCLCPP_INFO_THROTTLE(
        //     this->get_logger(), *this->get_clock(), 1000,
        //     "t=%.3f frame=%s child=%s | pos(%.2f,%.2f,%.2f) | q(%.2f,%.2f,%.2f,%.2f) | "
        //     "rpy(%.2f,%.2f,%.2f) | v(%.2f,%.2f,%.2f) | w(%.2f,%.2f,%.2f)",
        //     rclcpp::Time(msg->header.stamp).seconds(),
        //     msg->header.frame_id.c_str(), msg->child_frame_id.c_str(),
        //     msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z,
        //     msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
        //     msg->pose.pose.orientation.z, msg->pose.pose.orientation.w,
        //     latest_state_.roll, latest_state_.pitch, latest_state_.yaw,
        //     msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z,
        //     msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z);
    }





    // void waypoint_callback(
    //     const std::shared_ptr<mvp_msgs::srv::SendWaypoints::Request> request,
    //     std::shared_ptr<mvp_msgs::srv::SendWaypoints::Response> response)
    // {
    //     std::lock_guard<std::mutex> lock(wpt_mutex_);
    //     wpt_buffer_.push_back(*request);
    //     RCLCPP_INFO(this->get_logger(), "Received waypoint");

    //     response->success = true;
    // }


    void processing_loop()
    {
        rclcpp::Rate rate(10); // 10 Hz
        while (rclcpp::ok())
        {
            process_data();
            rate.sleep();
        }
    }

    // void process_data()
    // {
    // // 1) Drain odometry buffer quickly (under lock)
    // {
    //     std::lock_guard<std::mutex> lock(odom_mutex_);
    //     while (!odom_buffer_.empty()) {
    //     odom_buffer_.pop_front();
    //     }
    // }

    void process_data()
    {
        // 1) Drain odometry buffer quickly (under lock)
        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            while (!odom_buffer_.empty()) {
            odom_buffer_.pop_front();
            }
        }

        // 2) Integrate Xdot = [v; A X]  (circle in XY)  — NO locks here
        rclcpp::Time now = this->get_clock()->now();
        double dt = (last_step_.nanoseconds() == 0) ? 0.1 : (now - last_step_).seconds();
        if (dt <= 0.0 || dt > 1.0) dt = 0.1;
        last_step_ = now;

        const double ax = -omega_ * omega_ * p_[0];  // -ω² x
        const double ay = -omega_ * omega_ * p_[1];  // -ω² y
        const double az = 0.0;

        // velocities
        v_[0] += ax * dt;
        v_[1] += ay * dt;
        v_[2] += az * dt;
        // v_[3], v_[4] stay 0; v_[5] set in constructor

        // states
        p_[0] += v_[0] * dt;   // x
        p_[1] += v_[1] * dt;   // y
        p_[2] += v_[2] * dt;   // z
        p_[3] += v_[3] * dt;   // roll
        p_[4] += v_[4] * dt;   // pitch
        p_[5] += v_[5] * dt;   // yaw

        // 3) Publish waypoint
        geometry_msgs::msg::PoseStamped wp;
        wp.header.stamp = now;
        wp.header.frame_id = "world";
        wp.pose.position.x = p_[0];
        wp.pose.position.y = p_[1];
        wp.pose.position.z = p_[2];

        tf2::Quaternion q;
        q.setRPY(p_[3], p_[4], p_[5]);
        wp.pose.orientation = tf2::toMsg(q);

        wpt_pub_->publish(wp);

        // 4) Unthrottled print (for now)
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "WP pos(%.2f, %.2f, %.2f) orientation=(%.2f, %.2f, %.2f) linear velocity=(%.2f, %.2f, %.2f) angular velocity=(%.2f, %.2f, %.2f)", 
            p_[0], p_[1], p_[2], p_[3], p_[4],p_[5], v_[0], v_[1], v_[2], v_[3], v_[4], v_[5]);
    }




    // // 2) Drain waypoint buffer quickly (under lock)
    // {
    //     std::lock_guard<std::mutex> lock(wpt_mutex_);
    //     while (!wpt_buffer_.empty()) {
    //     auto wp = wpt_buffer_.front();
    //     wpt_buffer_.pop_front();
    //     // ... process incoming waypoints if you want ...
    //     }
    // }

    // // 3) Integrate Xdot = [v; A X] and publish waypoint (NO locks held here)
    // rclcpp::Time now = this->get_clock()->now();
    // double dt = (last_step_.nanoseconds() == 0) ? 0.1 : (now - last_step_).seconds();
    // if (dt <= 0.0 || dt > 1.0) dt = 0.1;
    // last_step_ = now;

    // // central accel for circle in XY: ax = -ω² x, ay = -ω² y
    // const double ax = -omega_ * omega_ * p_[0];
    // const double ay = -omega_ * omega_ * p_[1];
    // const double az = 0.0;

    // // integrate velocities
    // v_[0] += ax * dt;     // vx
    // v_[1] += ay * dt;     // vy
    // v_[2] += az * dt;     // vz
    // // v_[3], v_[4] stay 0; v_[5] (yaw rate) set in constructor

    // // integrate position & orientation
    // p_[0] += v_[0] * dt;  // x
    // p_[1] += v_[1] * dt;  // y
    // p_[2] += v_[2] * dt;  // z
    // p_[3] += v_[3] * dt;  // roll
    // p_[4] += v_[4] * dt;  // pitch
    // p_[5] += v_[5] * dt;  // yaw

    // // publish waypoint
    // geometry_msgs::msg::PoseStamped wp;
    // wp.header.stamp = now;
    // wp.header.frame_id = "world";
    // wp.pose.position.x = p_[0];
    // wp.pose.position.y = p_[1];
    // wp.pose.position.z = p_[2];

    // tf2::Quaternion q;
    // q.setRPY(p_[3], p_[4], p_[5]);
    // wp.pose.orientation = tf2::toMsg(q);

    // wpt_pub_->publish(wp);

    // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
    //     "WP pos(%.2f, %.2f, %.2f) yaw=%.2f", p_[0], p_[1], p_[2], p_[5]);
    

        // // get waypoints from the buffer
        // {
        //     std::lock_guard<std::mutex> lock(wpt_mutex_);
        //     while (!wpt_buffer_.empty())
        //     {
        //         auto wp = wpt_buffer_.front();
        //         wpt_buffer_.pop_front();

        //         // loop all the wp
        //         for(const auto& p : wp.wpt) {
        //             RCLCPP_INFO(this->get_logger(), "wp time:%.9f, [%.2f, %.2f, %.2f]",
        //                 rclcpp::Time(p.header.stamp).seconds(),
        //                 p.wpt.x, p.wpt.y, p.wpt.z);
        //         }

        //         // Process waypoint

        //     }
        // }


        // grab data
        // do your algorithm
        // detemrine the thruster [-1, 1] for thrusters in mauv1_1: surge, sway, ... in 10hz
        // publish the thruster cmd for topic: /mwav_1/thruster

        // publish to those topics:
            // /mauv_1/control/thruster/heave_bow [-0.2]
            // /mauv_1/control/thruster/heave_stern [ 0.4]
            // /mauv_1/control/thruster/surge [0.8]
            // /mauv_1/control/thruster/sway_bow [0.7]
        
    // }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DataProcessorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
