#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

#include "backend_interface.hpp"

namespace {

const std::string BASE_URL = "http://localhost:8000";
const std::string BEDROOM_ZIP_URL =
    "https://dl.fbaipublicfiles.com/segment_anything_2/assets/bedroom.zip";
const std::string BEDROOM_DIR = "tests/bedroom";
const std::string TRACKING_VIDEO_PATH = "backend/tests/apple/apple.mp4";  // relative to repo root

void runVideoTests(BackendInterface& backend) {
  std::cout
      << "\n============================================================\n";
  std::cout << "RUNNING VIDEO MASKING TESTS (C++)" << std::endl;
  std::cout << "============================================================\n";

  if (!backend.videoInitState(BEDROOM_DIR)) {
    std::cerr << "Failed to initialize video state. Skipping video tests."
              << std::endl;
    return;
  }

  int ann_frame_idx = 0;
  int ann_obj_id = 1;

  // First point (210, 350)
  std::cout << "Adding first point at (210, 350) on frame " << ann_frame_idx
            << "..." << std::endl;

  BackendInterface::VideoAddPointsOrBoxRequest req1;
  req1.frame_idx = ann_frame_idx;
  req1.obj_id = ann_obj_id;
  req1.points = std::vector<std::vector<float>>{{210.0f, 350.0f}};
  req1.labels = std::vector<int>{1};
  req1.clear_old_points = true;

  auto resp1 = backend.videoAddNewPointsOrBox(req1);
  if (resp1.out_obj_ids.empty()) {
    std::cerr << "Failed to add first point. Aborting video tests."
              << std::endl;
    backend.videoResetState();
    return;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Second point (210, 350) and (250, 220)
  std::cout << "Adding second point at (250, 220) on frame " << ann_frame_idx
            << "..." << std::endl;

  BackendInterface::VideoAddPointsOrBoxRequest req2;
  req2.frame_idx = ann_frame_idx;
  req2.obj_id = ann_obj_id;
  req2.points =
      std::vector<std::vector<float>>{{210.0f, 350.0f}, {250.0f, 220.0f}};
  req2.labels = std::vector<int>{1, 1};
  req2.clear_old_points = true;

  auto resp2 = backend.videoAddNewPointsOrBox(req2);
  if (resp2.out_obj_ids.empty()) {
    std::cerr << "Failed to add second point. Aborting video tests."
              << std::endl;
    backend.videoResetState();
    return;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Propagate through the video
  std::cout << "Propagating masks through video..." << std::endl;
  BackendInterface::VideoPropagateRequest propReq;
  propReq.start_frame_idx = std::nullopt;
  propReq.max_frame_num_to_track = std::nullopt;
  propReq.reverse = false;

  auto propResp = backend.videoPropagateInVideo(propReq);
  std::cout << "Propagation processed " << propResp.video_segments.size()
            << " frames." << std::endl;
  std::cout << "Saved masks for " << propResp.saved_mask_paths.size()
            << " frames." << std::endl;

  if (!propResp.video_segments.empty()) {
    std::cout << "\nVideo masking test completed successfully!" << std::endl;
  } else {
    std::cout << "\nVideo masking test may have failed (no segments)."
              << std::endl;
  }

  backend.videoResetState();
}

void runTrackingTests(BackendInterface& backend) {
  std::cout
      << "\n============================================================\n";
  std::cout << "RUNNING TRACKING TESTS (C++)" << std::endl;
  std::cout << "============================================================\n";

  if (!std::filesystem::exists(TRACKING_VIDEO_PATH)) {
    std::cerr << "Tracking video '" << TRACKING_VIDEO_PATH
              << "' not found. Skipping tracking tests." << std::endl;
    return;
  }

  // Test 1: Load video
  std::cout << "\n--- Test 1: Load tracking video ---" << std::endl;
  BackendInterface::TrackingLoadVideoRequest loadReq;
  loadReq.video_path = TRACKING_VIDEO_PATH;
  auto loadResp = backend.trackingLoadVideo(loadReq);

  if (loadResp.shape.empty() || loadResp.num_frames <= 0) {
    std::cerr << "Failed to load tracking video. Skipping remaining tests."
              << std::endl;
    return;
  }

  std::cout << "Video loaded. Frames: " << loadResp.num_frames << " Shape: [";
  for (size_t i = 0; i < loadResp.shape.size(); ++i) {
    std::cout << loadResp.shape[i];
    if (i + 1 < loadResp.shape.size()) std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Test 2: Track grid
  std::cout << "\n--- Test 2: Track grid of points ---" << std::endl;
  BackendInterface::TrackingGridRequest gridReq;
  gridReq.grid_size = 10;
  gridReq.add_support_grid = true;
  auto gridResp = backend.trackingTrackGrid(gridReq);

  if (gridResp.num_points > 0 && gridResp.num_frames > 0) {
    std::cout << "Grid tracking completed. Points: " << gridResp.num_points
              << " Frames: " << gridResp.num_frames << std::endl;
  } else {
    std::cerr << "Grid tracking test may have failed." << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Test 3: Track specific points
  std::cout << "\n--- Test 3: Track specific query points ---" << std::endl;
  BackendInterface::TrackingPointsRequest ptsReq1;
  ptsReq1.add_support_grid = false;
  ptsReq1.queries = {{0.f, 400.f, 350.f},
                     {10.f, 600.f, 500.f},
                     {20.f, 750.f, 600.f},
                     {30.f, 900.f, 200.f}};

  auto ptsResp1 = backend.trackingTrackPoints(ptsReq1);
  if (ptsResp1.num_points > 0 && ptsResp1.num_frames > 0) {
    std::cout << "Point tracking completed. Points: " << ptsResp1.num_points
              << " Frames: " << ptsResp1.num_frames << std::endl;
  } else {
    std::cerr << "Point tracking test may have failed." << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Test 4: Track points with support grid
  std::cout << "\n--- Test 4: Track points with support grid ---" << std::endl;
  BackendInterface::TrackingPointsRequest ptsReq2;
  ptsReq2.add_support_grid = true;
  ptsReq2.queries = {{0.f, 320.f, 240.f}};

  auto ptsResp2 = backend.trackingTrackPoints(ptsReq2);
  if (ptsResp2.num_points > 0 && ptsResp2.num_frames > 0) {
    std::cout << "Point tracking with support grid completed. Points: "
              << ptsResp2.num_points << " Frames: " << ptsResp2.num_frames
              << std::endl;
  } else {
    std::cerr << "Point tracking with support grid may have failed."
              << std::endl;
  }

  std::cout
      << "\n============================================================\n";
  std::cout << "TRACKING TESTS COMPLETED (C++)" << std::endl;
  std::cout << "============================================================\n";
}

}  // namespace

int main() {
  std::cout << "Starting C++ backend interface tests" << std::endl;

  BackendInterface backend(BASE_URL);
  if (!backend.launchBackend()) {
    std::cerr << "Failed to launch backend." << std::endl;
    return 1;
  }

  runVideoTests(backend);
  runTrackingTests(backend);

  backend.stopBackend();

  std::cout << "All C++ backend interface tests completed." << std::endl;
  return 0;
}
