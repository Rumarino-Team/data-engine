export interface VideoInitStateRequest {
  video_frames_dir: string;
  online_mode?: boolean;
  batch_size?: number;
  offload_video_to_cpu?: boolean;
  offload_state_to_cpu?: boolean;
  async_loading_frames?: boolean;
}

export interface VideoAddPointsOrBoxRequest {
  frame_idx: number;
  obj_id: number;
  points?: number[][];
  labels?: number[];
  clear_old_points?: boolean;
  box?: number[];
}

export interface VideoPropagateRequest {
  start_frame_idx?: number;
  max_frame_num_to_track?: number;
  reverse?: boolean;
  batch_size?: number;
  online_mode?: boolean;
  use_tracked_points?: boolean;
  tracked_point_keyframe_interval?: number;
  max_tracked_points_per_object_per_frame?: number;
  include_masks_in_response?: boolean;
  include_saved_mask_paths?: boolean;
  max_frames_in_response?: number;
  max_mask_values_in_response?: number;
}

export interface VideoAddMaskRequest {
  frame_idx: number;
  obj_id: number;
  mask: boolean[][];
}

export interface VideoSaveRequest {
  name: string;
  interactive_state?: VideoSaveInteractiveState;
}

export interface VideoSaveResponse {
  message: string;
  name: string;
  saved_path: string;
  state_epoch: number;
}

export interface VideoAddPointsResponse {
  request_frame_idx: number;
  frame_idx: number;
  frame_file: string;
  out_obj_ids: number[];
  out_masks: boolean[][][];
  mask_pixel_counts: Record<number, number>;
  mask_shapes: Record<number, [number, number]>;
  single_frame_fallback_used?: boolean;
  state_epoch: number;
}

export interface VideoInitStateResponse {
  message: string;
  num_frames: number;
  resolved_video_frames_dir: string;
  source_video_path?: string | null;
  online_mode: boolean;
  batch_size: number;
  offload_video_to_cpu: boolean;
  offload_state_to_cpu: boolean;
  state_epoch: number;
  source_type?: 'frames_dir' | 'video_file' | 'saved_session';
  restored_session?: RestoredSessionPayload;
}

export interface InteractiveObject {
  id: number;
  name: string;
  color: string;
}

export interface InteractivePoint {
  frame_idx: number;
  obj_id: number;
  x: number;
  y: number;
  label: 0 | 1;
}

export interface InteractiveMaskRle {
  frame_idx: number;
  obj_id: number;
  height: number;
  width: number;
  counts: number[];
}

export interface VideoSaveInteractiveState {
  version: number;
  objects: InteractiveObject[];
  selected_object_id?: number | null;
  interaction_mode?: 'positive' | 'negative' | null;
  current_frame_idx?: number | null;
  points: InteractivePoint[];
  live_masks: InteractiveMaskRle[];
}

export interface RestoredSessionPayload {
  session_meta: Record<string, unknown>;
  interactive_state?: VideoSaveInteractiveState | null;
  has_mask_manifest: boolean;
  interactive_state_warnings?: string[];
  tracking_result?: {
    result_id: string;
    summary?: Record<string, unknown>;
  };
}

export interface VideoPropagateResponse {
  video_segments: { [frame_idx: string]: { [obj_id: string]: boolean[][] } };
  saved_mask_paths: { [frame_idx: string]: string[] };
  saved_mask_frame_count?: number;
  video_segments_total_frames?: number;
  video_segments_returned_frames?: number;
  video_segments_returned_mask_values?: number;
  video_segments_truncated?: boolean;
  mask_manifest_path?: string;
  'state.mask_manifest_path'?: string;
  tracked_points_used?: boolean;
  tracked_points_skipped_reason?: string | null;
  tracked_points_seeded_count?: number;
  tracked_points_seeded_frames?: number;
  state_epoch?: number;
}

export interface VideoMaskObjectData {
  size: [number, number];
  rle: number[][];
  bbox: [number, number, number, number];
}

export interface VideoMaskDataResponse {
  frame_idx: number;
  objects: { [obj_id: string]: VideoMaskObjectData };
}

export interface VideoMaskDataWindowResponse {
  start_frame_idx: number;
  end_frame_idx: number;
  frames: {
    [frame_idx: string]: {
      objects: { [obj_id: string]: VideoMaskObjectData };
    };
  };
}

export interface TrackPromptPointsRequest {
  add_support_grid?: boolean;
}

export interface TrackPromptPointMetadata {
  point_id: string;
  obj_id: number;
  source_frame_idx: number;
  source_x: number;
  source_y: number;
}

export interface TrackPromptPointsJobResponse {
  message: string;
  model_name: string;
  num_points: number;
  num_frames: number;
  add_support_grid_used?: boolean;
  tracking_mode?: 'streaming' | 'in_memory';
  streaming_frame_threshold?: number;
  tracking_result_id: string;
  state_epoch?: number;
}

export interface TrackPromptPointsResult {
  version: number;
  result_id: string;
  model_name: string;
  num_points: number;
  num_frames: number;
  add_support_grid_used?: boolean;
  tracking_mode?: 'streaming' | 'in_memory';
  streaming_frame_threshold?: number;
  tracks: number[][][];
  visibility: boolean[][];
  points: TrackPromptPointMetadata[];
}

export type ApiHealthStatus = 'checking' | 'online' | 'offline';
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed';
export type JobOperation = 'video_init' | 'mask_propagation' | 'prompt_tracking';

export interface HealthResponse {
  status: string;
}

export interface JobStartResponse {
  job_id: string;
  status: JobStatus;
  operation: JobOperation;
  message: string;
}

export interface JobError {
  code: string;
  message: string;
  detail?: string | null;
}

export interface JobStageHistoryEntry {
  stage: string;
  stage_label: string;
  message: string;
  progress: number | null;
  updated_at: string;
}

export interface BackendJob<T = unknown> {
  job_id: string;
  operation: JobOperation;
  status: JobStatus;
  stage: string;
  stage_label: string;
  progress: number | null;
  current: number | null;
  total: number | null;
  window_index: number | null;
  window_count: number | null;
  frame_idx: number | null;
  batch_current: number | null;
  batch_total: number | null;
  batch_index: number | null;
  batch_count: number | null;
  stage_history?: JobStageHistoryEntry[];
  message: string;
  result: T | null;
  error: JobError | null;
  started_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface JobResponse<T = unknown> {
  job: BackendJob<T>;
}

export interface CurrentJobResponse {
  job: BackendJob | null;
}
