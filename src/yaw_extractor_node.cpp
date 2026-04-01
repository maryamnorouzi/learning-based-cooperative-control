#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class YawExtractorNode : public rclcpp::Node
{
public:
  YawExtractorNode() : Node("yaw_extractor_node")
  {
    // Parameters to let you choose which odom topics to listen to
    std::string world_topic =
      this->declare_parameter<std::string>("world_odom_topic", "world_odom");
    std::string wp_topic =
      this->declare_parameter<std::string>("wp_odom_topic", "waypoint_odom");

    // Publishers: actual and desired yaw [rad]
    pub_yaw_world_ = this->create_publisher<std_msgs::msg::Float64>(
        "yaw_world", 10);
    pub_yaw_wp_ = this->create_publisher<std_msgs::msg::Float64>(
        "yaw_desired", 10);

    // Subscriptions
    sub_world_ = this->create_subscription<nav_msgs::msg::Odometry>(
        world_topic, 10,
        std::bind(&YawExtractorNode::world_cb_, this, std::placeholders::_1));

    sub_wp_ = this->create_subscription<nav_msgs::msg::Odometry>(
        wp_topic, 10,
        std::bind(&YawExtractorNode::wp_cb_, this, std::placeholders::_1));
  }

private:
  static double quat_to_yaw(const geometry_msgs::msg::Quaternion & q_msg)
  {
    tf2::Quaternion q;
    tf2::fromMsg(q_msg, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    return yaw;   // [rad]
  }

  void world_cb_(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std_msgs::msg::Float64 out;
    out.data = quat_to_yaw(msg->pose.pose.orientation);
    pub_yaw_world_->publish(out);
  }

  void wp_cb_(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std_msgs::msg::Float64 out;
    out.data = quat_to_yaw(msg->pose.pose.orientation);
    pub_yaw_wp_->publish(out);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_world_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_wp_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_yaw_world_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_yaw_wp_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<YawExtractorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
