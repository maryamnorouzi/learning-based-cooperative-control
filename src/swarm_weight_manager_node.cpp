#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

using std::placeholders::_1;

constexpr int kRbfDim = 6;

class SwarmWeightManagerNode : public rclcpp::Node
{
public:
  SwarmWeightManagerNode() : Node("swarm_weight_manager_node")
  {
    agent_names_ = this->declare_parameter<std::vector<std::string>>(
      "agent_names",
      std::vector<std::string>{"mauv_1", "mauv_2", "mauv_3"});
    zetta_ne_ = this->declare_parameter<int>("zetta_ne", 8);
    lambda_ = this->declare_parameter<double>("lambda", 0.2);

    const auto lo = this->declare_parameter<std::vector<double>>(
      "rbf_lo6",
      std::vector<double>(kRbfDim, -1.0));
    const auto hi = this->declare_parameter<std::vector<double>>(
      "rbf_hi6",
      std::vector<double>(kRbfDim, 1.0));

    if (lo.size() != static_cast<std::size_t>(kRbfDim) ||
        hi.size() != static_cast<std::size_t>(kRbfDim)) {
      RCLCPP_WARN(
        get_logger(),
        "rbf_lo6 and rbf_hi6 must each contain %d values. "
        "Falling back to [-1, 1] defaults for metadata validation.",
        kRbfDim);
      rbf_lo_.assign(kRbfDim, -1.0);
      rbf_hi_.assign(kRbfDim, 1.0);
    } else {
      rbf_lo_ = lo;
      rbf_hi_ = hi;
    }

    const std::filesystem::path default_storage_dir = default_storage_dir_();
    shared_wbar_save_path_ = this->declare_parameter<std::string>(
      "shared_wbar_save_path",
      (default_storage_dir / "shared_wbar.bin").string());
    shared_wbar_meta_path_ = this->declare_parameter<std::string>(
      "shared_wbar_meta_path",
      (default_storage_dir / "shared_wbar.meta").string());
    auto_save_shared_wbar_ =
      this->declare_parameter<bool>("auto_save_shared_wbar", true);
    auto_load_shared_wbar_ =
      this->declare_parameter<bool>("auto_load_shared_wbar", true);

    const auto latched_qos =
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    pub_shared_wbar_ =
      this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "/swarm/shared_wbar",
        latched_qos);

    for (const auto& agent_name : agent_names_) {
      const std::string topic_name = "/" + agent_name + "/local_frozen_wbar";
      auto subscription =
        this->create_subscription<std_msgs::msg::Float32MultiArray>(
          topic_name,
          latched_qos,
          [this, agent_name](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
            this->local_frozen_wbar_callback_(agent_name, msg);
          });

      local_wbar_subs_.push_back(subscription);
      RCLCPP_INFO(
        get_logger(),
        "Subscribed to local frozen w_bar topic: %s",
        topic_name.c_str());
    }

    if (auto_load_shared_wbar_) {
      std::vector<float> loaded_shared_wbar;
      if (load_shared_wbar_from_file_(loaded_shared_wbar)) {
        publish_shared_wbar_(loaded_shared_wbar, "Loaded saved shared swarm w_bar");
      }
    }
  }

private:
  // --------------------------------------------------------------------------
  // Storage helpers
  // --------------------------------------------------------------------------
  static std::filesystem::path default_storage_dir_()
  {
    const char* home = std::getenv("HOME");
    if (home && *home) {
      return std::filesystem::path(home) / ".ros" / "simple_controller";
    }
    return std::filesystem::path("/tmp") / "simple_controller";
  }

  std::size_t expected_weight_count_() const
  {
    uint64_t basis_count = 1;
    for (int idx = 0; idx < kRbfDim; ++idx) {
      basis_count *= static_cast<uint64_t>(zetta_ne_);
    }
    return static_cast<std::size_t>(4 * basis_count);
  }

  static std::string join_vector_(const std::vector<double>& values)
  {
    std::ostringstream stream;
    stream << std::setprecision(17);
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
      if (idx > 0) {
        stream << ",";
      }
      stream << values[idx];
    }
    return stream.str();
  }

  static bool parse_csv_doubles_(
      const std::string& text,
      std::vector<double>& values)
  {
    values.clear();
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
      if (item.empty()) {
        return false;
      }
      values.push_back(std::stod(item));
    }
    return !values.empty();
  }

  static bool nearly_equal_(double lhs, double rhs, double tolerance = 1e-9)
  {
    return std::fabs(lhs - rhs) <= tolerance;
  }

  static bool nearly_equal_vectors_(
      const std::vector<double>& lhs,
      const std::vector<double>& rhs,
      double tolerance = 1e-9)
  {
    if (lhs.size() != rhs.size()) {
      return false;
    }

    for (std::size_t idx = 0; idx < lhs.size(); ++idx) {
      if (!nearly_equal_(lhs[idx], rhs[idx], tolerance)) {
        return false;
      }
    }
    return true;
  }

  bool save_shared_wbar_to_file_(const std::vector<float>& shared_weights)
  {
    try {
      const std::filesystem::path binary_path(shared_wbar_save_path_);
      const std::filesystem::path meta_path(shared_wbar_meta_path_);

      if (!binary_path.parent_path().empty()) {
        std::filesystem::create_directories(binary_path.parent_path());
      }
      if (!meta_path.parent_path().empty()) {
        std::filesystem::create_directories(meta_path.parent_path());
      }

      std::ofstream binary_stream(binary_path, std::ios::binary | std::ios::trunc);
      if (!binary_stream) {
        RCLCPP_ERROR(
          get_logger(),
          "Failed to open shared weight file for writing: %s",
          binary_path.c_str());
        return false;
      }

      binary_stream.write(
        reinterpret_cast<const char*>(shared_weights.data()),
        static_cast<std::streamsize>(shared_weights.size() * sizeof(float)));
      if (!binary_stream) {
        RCLCPP_ERROR(
          get_logger(),
          "Failed while writing shared weight file: %s",
          binary_path.c_str());
        return false;
      }

      std::ofstream meta_stream(meta_path, std::ios::trunc);
      if (!meta_stream) {
        RCLCPP_ERROR(
          get_logger(),
          "Failed to open shared weight metadata file for writing: %s",
          meta_path.c_str());
        return false;
      }

      meta_stream << "version=2\n";
      meta_stream << "weight_count=" << shared_weights.size() << "\n";
      meta_stream << "zetta_ne=" << zetta_ne_ << "\n";
      meta_stream << "rbf_dim=" << kRbfDim << "\n";
      meta_stream << std::setprecision(17)
                  << "lambda=" << lambda_ << "\n";
      meta_stream << "rbf_lo6=" << join_vector_(rbf_lo_) << "\n";
      meta_stream << "rbf_hi6=" << join_vector_(rbf_hi_) << "\n";

      if (!meta_stream) {
        RCLCPP_ERROR(
          get_logger(),
          "Failed while writing shared weight metadata file: %s",
          meta_path.c_str());
        return false;
      }

      RCLCPP_INFO(
        get_logger(),
        "Saved shared swarm w_bar to '%s' with metadata '%s'.",
        binary_path.c_str(),
        meta_path.c_str());
      return true;
    } catch (const std::exception& error) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to save shared swarm w_bar: %s",
        error.what());
      return false;
    }
  }

  bool load_shared_wbar_from_file_(std::vector<float>& shared_weights)
  {
    shared_weights.clear();

    try {
      const std::filesystem::path binary_path(shared_wbar_save_path_);
      const std::filesystem::path meta_path(shared_wbar_meta_path_);
      if (!std::filesystem::exists(binary_path) ||
          !std::filesystem::exists(meta_path)) {
        RCLCPP_INFO(
          get_logger(),
          "No saved shared swarm w_bar found at '%s' and '%s'.",
          binary_path.c_str(),
          meta_path.c_str());
        return false;
      }

      std::unordered_map<std::string, std::string> metadata;
      std::ifstream meta_stream(meta_path);
      std::string line;
      while (std::getline(meta_stream, line)) {
        const auto separator = line.find('=');
        if (separator == std::string::npos) {
          continue;
        }
        metadata[line.substr(0, separator)] = line.substr(separator + 1);
      }

      if (metadata.empty()) {
        RCLCPP_WARN(
          get_logger(),
          "Shared weight metadata file '%s' is empty or invalid.",
          meta_path.c_str());
        return false;
      }

      const std::size_t stored_weight_count =
        static_cast<std::size_t>(std::stoull(metadata.at("weight_count")));
      const int stored_zetta_ne = std::stoi(metadata.at("zetta_ne"));
      const double stored_lambda = std::stod(metadata.at("lambda"));
      const auto stored_rbf_dim_it = metadata.find("rbf_dim");
      if (stored_rbf_dim_it != metadata.end() &&
          std::stoi(stored_rbf_dim_it->second) != kRbfDim) {
        RCLCPP_WARN(
          get_logger(),
          "Shared weight metadata was saved for a different RBF dimension. "
          "Skipping auto-load.");
        return false;
      }

      const auto lo_metadata_it = metadata.find("rbf_lo6");
      const auto hi_metadata_it = metadata.find("rbf_hi6");
      if (lo_metadata_it == metadata.end() || hi_metadata_it == metadata.end()) {
        RCLCPP_WARN(
          get_logger(),
          "Shared weight metadata file '%s' does not contain 6-D RBF bounds. "
          "Skipping auto-load.",
          meta_path.c_str());
        return false;
      }

      std::vector<double> stored_lo;
      std::vector<double> stored_hi;
      if (!parse_csv_doubles_(lo_metadata_it->second, stored_lo) ||
          !parse_csv_doubles_(hi_metadata_it->second, stored_hi)) {
        RCLCPP_WARN(
          get_logger(),
          "Shared weight metadata file '%s' contains invalid RBF bounds.",
          meta_path.c_str());
        return false;
      }

      const std::size_t expected_weight_count = expected_weight_count_();
      if (stored_weight_count != expected_weight_count ||
          stored_zetta_ne != zetta_ne_ ||
          !nearly_equal_(stored_lambda, lambda_) ||
          !nearly_equal_vectors_(stored_lo, rbf_lo_) ||
          !nearly_equal_vectors_(stored_hi, rbf_hi_)) {
        RCLCPP_WARN(
          get_logger(),
          "Saved shared swarm w_bar metadata does not match the active controller "
          "configuration. Skipping auto-load.");
        return false;
      }

      const auto file_size = std::filesystem::file_size(binary_path);
      if (file_size != stored_weight_count * sizeof(float)) {
        RCLCPP_WARN(
          get_logger(),
          "Shared weight file '%s' has unexpected size: got=%zu expected=%zu.",
          binary_path.c_str(),
          static_cast<std::size_t>(file_size),
          stored_weight_count * sizeof(float));
        return false;
      }

      shared_weights.resize(stored_weight_count);
      std::ifstream binary_stream(binary_path, std::ios::binary);
      binary_stream.read(
        reinterpret_cast<char*>(shared_weights.data()),
        static_cast<std::streamsize>(stored_weight_count * sizeof(float)));
      if (!binary_stream) {
        RCLCPP_WARN(
          get_logger(),
          "Failed while reading shared weight file '%s'.",
          binary_path.c_str());
        shared_weights.clear();
        return false;
      }

      RCLCPP_INFO(
        get_logger(),
        "Loaded shared swarm w_bar from '%s' (%zu weights).",
        binary_path.c_str(),
        shared_weights.size());
      return true;
    } catch (const std::exception& error) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to load saved shared swarm w_bar: %s",
        error.what());
      shared_weights.clear();
      return false;
    }
  }

  // --------------------------------------------------------------------------
  // Shared swarm-weight publication
  // --------------------------------------------------------------------------
  void publish_shared_wbar_(
      const std::vector<float>& shared_weights,
      const std::string& source_label)
  {
    if (shared_weights.empty()) {
      return;
    }

    std_msgs::msg::Float32MultiArray msg;
    msg.layout.dim.resize(2);
    msg.layout.dim[0].label = "output";
    msg.layout.dim[0].size = 4;
    msg.layout.dim[0].stride = static_cast<uint32_t>(shared_weights.size());
    msg.layout.dim[1].label = "rbf";
    msg.layout.dim[1].size = static_cast<uint32_t>(shared_weights.size() / 4);
    msg.layout.dim[1].stride = static_cast<uint32_t>(shared_weights.size() / 4);
    msg.data = shared_weights;
    pub_shared_wbar_->publish(msg);

    RCLCPP_INFO(
      get_logger(),
      "%s (size=%zu).",
      source_label.c_str(),
      shared_weights.size());
  }

  // --------------------------------------------------------------------------
  // Local frozen w_bar input
  // --------------------------------------------------------------------------
  void local_frozen_wbar_callback_(
      const std::string& agent_name,
      const std_msgs::msg::Float32MultiArray::SharedPtr msg)
  {
    {
      std::lock_guard<std::mutex> lk(local_wbar_mutex_);
      local_frozen_wbars_[agent_name] = msg->data;
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 10000,
      "Received local frozen w_bar from '%s' (size=%zu).",
      agent_name.c_str(),
      msg->data.size());

    publish_shared_wbar_if_ready_();
  }

  void publish_shared_wbar_if_ready_()
  {
    std::vector<float> shared_average;
    std::size_t weight_count = 0;

    {
      std::lock_guard<std::mutex> lk(local_wbar_mutex_);
      if (local_frozen_wbars_.size() != agent_names_.size()) {
        return;
      }

      for (const auto& agent_name : agent_names_) {
        const auto it = local_frozen_wbars_.find(agent_name);
        if (it == local_frozen_wbars_.end()) {
          return;
        }

        if (weight_count == 0) {
          weight_count = it->second.size();
          if (weight_count == 0) {
            return;
          }
          shared_average.assign(weight_count, 0.0f);
        } else if (it->second.size() != weight_count) {
          RCLCPP_WARN(
            get_logger(),
            "Local frozen w_bar size mismatch across agents. "
            "Cannot compute shared average yet.");
          return;
        }

        for (std::size_t idx = 0; idx < weight_count; ++idx) {
          shared_average[idx] += it->second[idx];
        }
      }
    }

    const float inverse_agent_count =
      1.0f / static_cast<float>(agent_names_.size());
    for (float& value : shared_average) {
      value *= inverse_agent_count;
    }

    publish_shared_wbar_(
      shared_average,
      "Published shared swarm w_bar using all local frozen weights");

    if (auto_save_shared_wbar_) {
      save_shared_wbar_to_file_(shared_average);
    }
  }

  std::vector<std::string> agent_names_;
  int zetta_ne_{8};
  double lambda_{0.2};
  std::vector<double> rbf_lo_{std::vector<double>(kRbfDim, -1.0)};
  std::vector<double> rbf_hi_{std::vector<double>(kRbfDim, 1.0)};
  std::string shared_wbar_save_path_;
  std::string shared_wbar_meta_path_;
  bool auto_save_shared_wbar_{true};
  bool auto_load_shared_wbar_{true};
  std::vector<rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr>
    local_wbar_subs_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr
    pub_shared_wbar_;

  std::unordered_map<std::string, std::vector<float>> local_frozen_wbars_;
  std::mutex local_wbar_mutex_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SwarmWeightManagerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
