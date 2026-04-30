#ifndef BACKEND_INTERFACE_HPP
#define BACKEND_INTERFACE_HPP

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Forward declarations
struct VideoSegments;
struct TrackingResult;

class BackendInterface {
 public:
  BackendInterface(const std::string& base_url = "http://localhost:8000");
  ~BackendInterface();

  // Lifecycle management
  // Launch the FastAPI backend
  bool launchBackend(const std::string& python_executable = "python3",
                     const std::string& api_script_path = "./backend/api.py");
  bool stopBackend();

  // Status endpoint
  struct StatusResponse {
    std::string status;
    std::optional<std::string> device;
  };
  StatusResponse getStatus();

  // Video masking endpoints
  bool videoInitState(const std::string& video_frames_dir);
  bool videoResetState();

  struct VideoAddPointsOrBoxRequest {
    int frame_idx;
    int obj_id;
    std::optional<std::vector<std::vector<float>>> points;
    std::optional<std::vector<int>> labels;
    bool clear_old_points = true;
    std::optional<std::vector<float>> box;
  };

  struct VideoAddPointsOrBoxResponse {
    std::vector<int> out_obj_ids;
    std::vector<std::vector<std::vector<bool>>> out_masks;
  };

  VideoAddPointsOrBoxResponse videoAddNewPointsOrBox(
      const VideoAddPointsOrBoxRequest& request);

  struct VideoAddMaskRequest {
    int frame_idx;
    int obj_id;
    std::vector<std::vector<bool>> mask;
  };

  struct VideoAddMaskResponse {
    int frame_idx;
    std::vector<int> out_obj_ids;
    std::vector<std::vector<std::vector<bool>>> out_masks;
  };

  VideoAddMaskResponse videoAddNewMask(const VideoAddMaskRequest& request);

  struct VideoPropagateRequest {
    std::optional<int> start_frame_idx;
    std::optional<int> max_frame_num_to_track;
    bool reverse = false;
  };

  struct VideoPropagateResponse {
    std::map<int, std::map<int, std::vector<std::vector<std::vector<bool>>>>>
        video_segments;
    std::map<int, std::vector<std::string>> saved_mask_paths;
  };

  VideoPropagateResponse videoPropagateInVideo(
      const VideoPropagateRequest& request);

  bool videoClearAllPromptsInFrame(int frame_idx, int obj_id);
  bool videoRemoveObject(int obj_id);

  // Tracking endpoints
  struct TrackingLoadVideoRequest {
    std::string video_path;
  };

  struct TrackingLoadVideoResponse {
    std::string message;
    std::vector<int> shape;
    int num_frames;
  };

  TrackingLoadVideoResponse trackingLoadVideo(
      const TrackingLoadVideoRequest& request);

  struct TrackingGridRequest {
    int grid_size = 15;
    bool add_support_grid = true;
  };

  struct TrackingGridResponse {
    std::string message;
    std::vector<std::vector<std::vector<float>>>
        tracks;                                 // [num_points, num_frames, 2]
    std::vector<std::vector<bool>> visibility;  // [num_points, num_frames]
    int num_points;
    int num_frames;
    std::string output_video_path;
  };

  TrackingGridResponse trackingTrackGrid(const TrackingGridRequest& request);

  struct TrackingPointsRequest {
    std::vector<std::vector<float>> queries;  // List of [t, x, y] coordinates
    bool add_support_grid = true;
  };

  struct TrackingPointsResponse {
    std::string message;
    std::vector<std::vector<std::vector<float>>>
        tracks;                                 // [num_points, num_frames, 2]
    std::vector<std::vector<bool>> visibility;  // [num_points, num_frames]
    int num_points;
    int num_frames;
    std::string output_video_path;
  };

  TrackingPointsResponse trackingTrackPoints(
      const TrackingPointsRequest& request);

 private:
  std::string base_url_;
  int backend_pid_;
  bool backend_running_;

  // HTTP helper methods
  std::string httpGet(const std::string& endpoint, bool logFailures = true);
  std::string httpPost(const std::string& endpoint,
                       const std::string& json_body);

  // JSON conversion helpers
  std::string serializeVideoAddPointsOrBoxRequest(
      const VideoAddPointsOrBoxRequest& request);
  std::string serializeVideoAddMaskRequest(const VideoAddMaskRequest& request);
  std::string serializeVideoPropagateRequest(
      const VideoPropagateRequest& request);
  std::string serializeTrackingLoadVideoRequest(
      const TrackingLoadVideoRequest& request);
  std::string serializeTrackingGridRequest(const TrackingGridRequest& request);
  std::string serializeTrackingPointsRequest(
      const TrackingPointsRequest& request);
};

#endif  // BACKEND_INTERFACE_HPP