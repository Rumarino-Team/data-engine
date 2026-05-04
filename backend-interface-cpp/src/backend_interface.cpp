#include "backend_interface.hpp"

#include <curl/curl.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>

namespace {
// Truncate large bodies to keep logs readable while remaining informative.
inline std::string truncateForLog(const std::string& s,
                                  std::size_t max = 8192) {
  if (s.size() <= max) return s;
  std::string out = s.substr(0, max);
  out += "... [truncated ";
  out += std::to_string(s.size() - max);
  out += " bytes]";
  return out;
}
}  // namespace

// Constructor
BackendInterface::BackendInterface(const std::string& base_url)
    : base_url_(base_url), backend_pid_(-1), backend_running_(false) {}

// Destructor
BackendInterface::~BackendInterface() { stopBackend(); }

// Lifecycle management
// Launch the FastAPI backend
bool BackendInterface::launchBackend(const std::string& python_executable,
                                     const std::string& api_script_path) {
  if (backend_running_) {
    return true;
  }

  // Build command to launch FastAPI with uvicorn from within api_script_path's
  // folder Do it literally like: cd <dir> && python <module>
  std::string dir;
  std::string script;
  {
    // naive split: everything before last '/' is dir, after is filename
    auto pos = api_script_path.find_last_of('/');
    if (pos == std::string::npos) {
      dir = ".";
      script = api_script_path;
    } else {
      dir = api_script_path.substr(0, pos);
      script = api_script_path.substr(pos + 1);
    }
  }

  // derive module name from filename (strip trailing .py if present)
  std::string module = script;
  const std::string py_ext = ".py";
  if (module.size() >= py_ext.size() &&
      module.compare(module.size() - py_ext.size(), py_ext.size(), py_ext) ==
          0) {
    module = module.substr(0, module.size() - py_ext.size());
  }

  // cd into dir and run uvicorn using module:app
  std::string command = "cd \"" + dir + "\" && " + python_executable +
                        " -m uvicorn " + module +
                        ":app --host 0.0.0.0 --port 8000 --no-access-log";

  // On POSIX systems, use a subshell to capture the PID of the uvicorn
  // process. This keeps things simple while still allowing us to send a
  // SIGTERM in stopBackend.
  // Redirect backend stdout and stderr to log files in the script directory.
  // Overwrite existing logs on each launch to avoid unbounded growth.
  std::string full_command =
      command + " >backend_stdout.log 2>backend_stderr.log & echo $!";

  FILE* pipe = popen(full_command.c_str(), "r");
  if (!pipe) {
    std::cerr << "Failed to start backend process" << std::endl;
    backend_running_ = false;
    backend_pid_ = -1;
    return false;
  }

  char buffer[64];
  std::string pid_str;
  if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    pid_str = buffer;
  }
  int status = pclose(pipe);
  if (!pid_str.empty()) {
    try {
      backend_pid_ = std::stoi(pid_str);
    } catch (...) {
      backend_pid_ = -1;
    }
  } else {
    backend_pid_ = -1;
  }

  if (status == -1) {
    std::cerr << "Failed to obtain backend process status" << std::endl;
  }

  if (backend_pid_ <= 0) {
    std::cerr << "Failed to determine backend PID" << std::endl;
  }

  // Poll the /health endpoint until the backend reports ready or timeout.
  const int max_retries = 50;  // up to ~10 seconds
  const auto delay = std::chrono::milliseconds(200);
  bool ready = false;

  for (int i = 0; i < max_retries; ++i) {
    std::string resp = httpGet("/health", false);
    if (!resp.empty()) {
      ready = true;
      break;
    }
    std::this_thread::sleep_for(delay);
  }

  backend_running_ = ready;
  if (!backend_running_) {
    std::cerr << "Backend did not become healthy within timeout" << std::endl;
  }

  return backend_running_;
}

bool BackendInterface::stopBackend() {
  if (!backend_running_) {
    return true;
  }

  if (backend_pid_ > 0) {
    // Send SIGTERM to allow graceful shutdown; ignore errors for now.
    ::kill(backend_pid_, SIGTERM);
  }

  backend_running_ = false;
  backend_pid_ = -1;
  return true;
}

// Status endpoint
BackendInterface::StatusResponse BackendInterface::getStatus() {
  using nlohmann::json;

  StatusResponse result;
  std::string response = httpGet("/status");
  if (response.empty()) {
    result.status = "unreachable";
    result.device = std::nullopt;
    return result;
  }

  try {
    auto j = json::parse(response);
    result.status = j.value("status", "unknown");
    if (j.contains("device") && !j["device"].is_null()) {
      result.device = j["device"].get<std::string>();
    } else {
      result.device = std::nullopt;
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /status response: " << e.what() << std::endl;
    result.status = "error";
    result.device = std::nullopt;
  }

  return result;
}

// Video masking endpoints
bool BackendInterface::videoInitState(const std::string& video_frames_dir) {
  nlohmann::json j;
  j["video_frames_dir"] = video_frames_dir;
  std::string resp = httpPost("/video/init_state", j.dump());
  return !resp.empty();
}

bool BackendInterface::videoResetState() {
  std::string resp = httpPost("/video/reset_state", "{}");
  return !resp.empty();
}

BackendInterface::VideoAddPointsOrBoxResponse
BackendInterface::videoAddNewPointsOrBox(
    const VideoAddPointsOrBoxRequest& request) {
  auto body = serializeVideoAddPointsOrBoxRequest(request);
  std::string resp = httpPost("/video/add_new_points_or_box", body);

  VideoAddPointsOrBoxResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);

    // Check if response contains an error
    if (j.contains("error")) {
      std::cerr << "Backend error: " << j["error"].get<std::string>()
                << std::endl;
      return result;
    }

    result.out_obj_ids = j.at("out_obj_ids").get<std::vector<int>>();
    result.out_masks =
        j.at("out_masks").get<std::vector<std::vector<std::vector<bool>>>>();
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /video/add_new_points_or_box response: "
              << e.what() << std::endl;
    std::cerr << "Response was: " << resp << std::endl;
  }

  return result;
}

BackendInterface::VideoAddMaskResponse BackendInterface::videoAddNewMask(
    const VideoAddMaskRequest& request) {
  auto body = serializeVideoAddMaskRequest(request);
  std::string resp = httpPost("/video/add_new_mask", body);

  VideoAddMaskResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);
    result.frame_idx = j.at("frame_idx").get<int>();
    result.out_obj_ids = j.at("out_obj_ids").get<std::vector<int>>();
    result.out_masks =
        j.at("out_masks").get<std::vector<std::vector<std::vector<bool>>>>();
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /video/add_new_mask response: " << e.what()
              << std::endl;
  }

  return result;
}

BackendInterface::VideoPropagateResponse
BackendInterface::videoPropagateInVideo(const VideoPropagateRequest& request) {
  auto body = serializeVideoPropagateRequest(request);
  std::string resp = httpPost("/video/propagate_in_video", body);

  VideoPropagateResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);

    // video_segments: map<int, map<int, vector<vector<vector<bool>>>>>
    // (preserve 3D masks)
    if (j.contains("video_segments")) {
      for (auto& [frame_str, obj_dict] : j["video_segments"].items()) {
        int frame_idx = std::stoi(frame_str);
        std::map<int, std::vector<std::vector<std::vector<bool>>>> inner_map;
        for (auto& [obj_str, mask_val] : obj_dict.items()) {
          int obj_id = std::stoi(obj_str);
          if (mask_val.is_array()) {
            inner_map[obj_id] =
                mask_val.get<std::vector<std::vector<std::vector<bool>>>>();
            std::cout << "is else if" << std::endl;
          }
        }
        result.video_segments[frame_idx] = std::move(inner_map);
      }
    }

    if (j.contains("saved_mask_paths")) {
      for (auto& [frame_str, paths_json] : j["saved_mask_paths"].items()) {
        int frame_idx = -1;
        try {
          frame_idx = std::stoi(frame_str);
        } catch (...) {
        }
        if (frame_idx < 0) {
          continue;
        }
        std::vector<std::string> paths;
        paths = paths_json.get<std::vector<std::string>>();

        result.saved_mask_paths[frame_idx] = std::move(paths);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /video/propagate_in_video response: "
              << e.what() << std::endl;
    std::clog
        << "[videoPropagateInVideo] Exception during parse. Response snippet: "
        << truncateForLog(resp, 512) << std::endl;
  }

  return result;
}

bool BackendInterface::videoClearAllPromptsInFrame(int frame_idx, int obj_id) {
  nlohmann::json j;
  j["frame_idx"] = frame_idx;
  j["obj_id"] = obj_id;
  std::string resp = httpPost("/video/clear_all_prompts_in_frame", j.dump());
  return !resp.empty();
}

bool BackendInterface::videoRemoveObject(int obj_id) {
  nlohmann::json j;
  j["obj_id"] = obj_id;
  std::string resp = httpPost("/video/remove_object", j.dump());
  return !resp.empty();
}

// Tracking endpoints
BackendInterface::TrackingLoadVideoResponse BackendInterface::trackingLoadVideo(
    const TrackingLoadVideoRequest& request) {
  auto body = serializeTrackingLoadVideoRequest(request);
  std::string resp = httpPost("/tracking/load_video", body);

  TrackingLoadVideoResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);
    result.message = j.value("message", "");
    result.shape = j.at("shape").get<std::vector<int>>();
    result.num_frames = j.at("num_frames").get<int>();
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /tracking/load_video response: " << e.what()
              << std::endl;
  }

  return result;
}

BackendInterface::TrackingGridResponse BackendInterface::trackingTrackGrid(
    const TrackingGridRequest& request) {
  auto body = serializeTrackingGridRequest(request);
  std::string resp = httpPost("/tracking/track_grid", body);

  TrackingGridResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);
    result.message = j.value("message", "");
    result.tracks =
        j.at("tracks").get<std::vector<std::vector<std::vector<float>>>>();
    // visibility can arrive as booleans or ints (0/1). Try bools first, then
    // fallback.
    try {
      result.visibility =
          j.at("visibility").get<std::vector<std::vector<bool>>>();
    } catch (const std::exception&) {
      try {
        auto vis_ints = j.at("visibility").get<std::vector<std::vector<int>>>();
        result.visibility.clear();
        result.visibility.reserve(vis_ints.size());
        for (const auto& row : vis_ints) {
          std::vector<bool> brow;
          brow.reserve(row.size());
          for (int v : row) brow.push_back(v != 0);
          result.visibility.push_back(std::move(brow));
        }
      } catch (...) {
        // Leave visibility empty if parsing fails.
      }
    }
    result.num_points = j.at("num_points").get<int>();
    result.num_frames = j.at("num_frames").get<int>();
    result.output_video_path = j.value("output_video_path", "");
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /tracking/track_grid response: " << e.what()
              << std::endl;
  }

  return result;
}

BackendInterface::TrackingPointsResponse BackendInterface::trackingTrackPoints(
    const TrackingPointsRequest& request) {
  auto body = serializeTrackingPointsRequest(request);
  std::string resp = httpPost("/tracking/track_points", body);

  TrackingPointsResponse result;
  if (resp.empty()) {
    return result;
  }

  try {
    auto j = nlohmann::json::parse(resp);
    result.message = j.value("message", "");
    result.tracks =
        j.at("tracks").get<std::vector<std::vector<std::vector<float>>>>();
    // visibility can arrive as booleans or ints (0/1). Try bools first, then
    // fallback.
    try {
      result.visibility =
          j.at("visibility").get<std::vector<std::vector<bool>>>();
    } catch (const std::exception&) {
      try {
        auto vis_ints = j.at("visibility").get<std::vector<std::vector<int>>>();
        result.visibility.clear();
        result.visibility.reserve(vis_ints.size());
        for (const auto& row : vis_ints) {
          std::vector<bool> brow;
          brow.reserve(row.size());
          for (int v : row) brow.push_back(v != 0);
          result.visibility.push_back(std::move(brow));
        }
      } catch (...) {
        // Leave visibility empty if parsing fails.
      }
    }
    result.num_points = j.at("num_points").get<int>();
    result.num_frames = j.at("num_frames").get<int>();
    result.output_video_path = j.value("output_video_path", "");
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse /tracking/track_points response: " << e.what()
              << std::endl;
  }

  return result;
}

// HTTP helper methods
std::string BackendInterface::httpGet(const std::string& endpoint,
                                      bool logFailures) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    std::cerr << "Failed to initialize CURL for GET" << std::endl;
    return "";
  }

  std::string readBuffer;
  std::string url = base_url_ + endpoint;

  if (logFailures) {
    std::clog << "[BackendInterface] HTTP GET  " << url << std::endl;
  }

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(
      curl, CURLOPT_WRITEFUNCTION,
      +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
        auto* buffer = static_cast<std::string*>(userdata);
        buffer->append(ptr, size * nmemb);
        return size * nmemb;
      });
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    if (logFailures) {
      std::cerr << "CURL GET failed: " << curl_easy_strerror(res) << std::endl;
    }
    readBuffer.clear();
  }
  long status_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);

  curl_easy_cleanup(curl);
  if (logFailures) {
    std::clog << "[BackendInterface] HTTP GET  " << url << " -> " << status_code
              << " body: " << truncateForLog(readBuffer) << std::endl;
  }
  return readBuffer;
}

std::string BackendInterface::httpPost(const std::string& endpoint,
                                       const std::string& json_body) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    std::cerr << "Failed to initialize CURL for POST" << std::endl;
    return "";
  }

  std::string readBuffer;
  std::string url = base_url_ + endpoint;

  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "Accept: application/json");

  std::clog << "[BackendInterface] HTTP POST " << url
            << " body: " << truncateForLog(json_body) << std::endl;

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
  curl_easy_setopt(
      curl, CURLOPT_WRITEFUNCTION,
      +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
        auto* buffer = static_cast<std::string*>(userdata);
        buffer->append(ptr, size * nmemb);
        return size * nmemb;
      });
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    std::cerr << "CURL POST failed: " << curl_easy_strerror(res) << std::endl;
    readBuffer.clear();
  }
  long status_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  std::clog << "[BackendInterface] HTTP POST " << url << " -> " << status_code
            << " body: " << truncateForLog(readBuffer) << std::endl;
  return readBuffer;
}

// JSON conversion helpers
std::string BackendInterface::serializeVideoAddPointsOrBoxRequest(
    const VideoAddPointsOrBoxRequest& request) {
  nlohmann::json j;
  j["frame_idx"] = request.frame_idx;
  j["obj_id"] = request.obj_id;
  if (request.points.has_value()) {
    j["points"] = *request.points;
  } else {
    j["points"] = nullptr;
  }
  if (request.labels.has_value()) {
    j["labels"] = *request.labels;
  } else {
    j["labels"] = nullptr;
  }
  j["clear_old_points"] = request.clear_old_points;
  if (request.box.has_value()) {
    j["box"] = *request.box;
  } else {
    j["box"] = nullptr;
  }
  return j.dump();
}

std::string BackendInterface::serializeVideoAddMaskRequest(
    const VideoAddMaskRequest& request) {
  nlohmann::json j;
  j["frame_idx"] = request.frame_idx;
  j["obj_id"] = request.obj_id;
  j["mask"] = request.mask;
  return j.dump();
}

std::string BackendInterface::serializeVideoPropagateRequest(
    const VideoPropagateRequest& request) {
  nlohmann::json j;
  if (request.start_frame_idx.has_value()) {
    j["start_frame_idx"] = *request.start_frame_idx;
  } else {
    j["start_frame_idx"] = nullptr;
  }
  if (request.max_frame_num_to_track.has_value()) {
    j["max_frame_num_to_track"] = *request.max_frame_num_to_track;
  } else {
    j["max_frame_num_to_track"] = nullptr;
  }
  j["reverse"] = request.reverse;
  return j.dump();
}

std::string BackendInterface::serializeTrackingLoadVideoRequest(
    const TrackingLoadVideoRequest& request) {
  nlohmann::json j;
  std::filesystem::path input_path(request.video_path);
  std::error_code ec;
  std::filesystem::path abs_path = std::filesystem::absolute(input_path, ec);
  if (!ec) {
    // Normalize to remove redundant ./ and ../ segments.
    j["video_path"] = abs_path.lexically_normal().string();
  } else {
    // Fallback: use the original value if absolute resolution failed.
    j["video_path"] = input_path.string();
  }
  return j.dump();
}

std::string BackendInterface::serializeTrackingGridRequest(
    const TrackingGridRequest& request) {
  nlohmann::json j;
  j["grid_size"] = request.grid_size;
  j["add_support_grid"] = request.add_support_grid;
  return j.dump();
}

std::string BackendInterface::serializeTrackingPointsRequest(
    const TrackingPointsRequest& request) {
  nlohmann::json j;
  j["queries"] = request.queries;
  j["add_support_grid"] = request.add_support_grid;
  return j.dump();
}
