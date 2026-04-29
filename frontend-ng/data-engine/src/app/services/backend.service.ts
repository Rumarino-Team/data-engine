import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, timeout } from 'rxjs';

const API_URL_STORAGE_KEY = 'dataEngineApiUrl';
const DEFAULT_API_URL = 'http://127.0.0.1:8000';

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
	out_masks: boolean[][][]; // List of masks (which are 2D boolean arrays)
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

export interface TrackPromptPointsRequest {
	model_name: 'cotracker3_online' | 'cotracker3_offline';
	add_support_grid?: boolean;
}

export interface TrackPromptPointMetadata {
	point_id: string;
	obj_id: number;
	source_frame_idx: number;
	source_x: number;
	source_y: number;
}

export interface TrackPromptPointsResponse {
	message: string;
	model_name: string;
	num_points: number;
	num_frames: number;
	tracks: number[][][];
	visibility: boolean[][];
	points: TrackPromptPointMetadata[];
	state_epoch?: number;
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

@Injectable({
	providedIn: 'root'
})
export class BackendService {
	private apiUrl = this.resolveApiUrl();

	constructor(private http: HttpClient) { }

	private resolveApiUrl(): string {
		const persistedConfig = this.readStoredApiUrl();
		const globalConfig = (globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__;
		return this.normalizeApiUrl(persistedConfig || globalConfig || DEFAULT_API_URL);
	}

	private readStoredApiUrl(): string | null {
		if (typeof localStorage === 'undefined') {
			return null;
		}
		return localStorage.getItem(API_URL_STORAGE_KEY);
	}

	private writeStoredApiUrl(value: string | null): void {
		if (typeof localStorage === 'undefined') {
			return;
		}
		if (value === null) {
			localStorage.removeItem(API_URL_STORAGE_KEY);
			return;
		}
		localStorage.setItem(API_URL_STORAGE_KEY, value);
	}

	private normalizeApiUrl(value: string): string {
		const normalized = value.trim().replace(/^['"]|['"]$/g, '');
		if (!normalized) {
			return DEFAULT_API_URL;
		}
		return normalized.replace(/\/+$/, '');
	}

	private endpoint(path: string): string {
		return `${this.apiUrl}${path}`;
	}

	getApiUrl(): string {
		return this.apiUrl;
	}

	setApiUrl(value: string): string {
		this.apiUrl = this.normalizeApiUrl(value);
		this.writeStoredApiUrl(this.apiUrl);
		return this.apiUrl;
	}

	resetApiUrl(): string {
		this.apiUrl = DEFAULT_API_URL;
		this.writeStoredApiUrl(null);
		return this.apiUrl;
	}

	health(): Observable<HealthResponse> {
		return this.http.get<HealthResponse>(this.endpoint('/health')).pipe(timeout(2000));
	}

	getCurrentJob(): Observable<CurrentJobResponse> {
		return this.http.get<CurrentJobResponse>(this.endpoint('/jobs/current'));
	}

	getJob<T = unknown>(jobId: string): Observable<JobResponse<T>> {
		return this.http.get<JobResponse<T>>(this.endpoint(`/jobs/${jobId}`));
	}

	private safeDecodeURIComponent(value: string): string {
		try {
			return decodeURIComponent(value);
		} catch {
			return value;
		}
	}

	private normalizePath(path: string): string {
		let normalized = path.trim().replace(/^['\"]|['\"]$/g, '');
		if (!normalized) {
			return normalized;
		}

		if (normalized.startsWith('file://')) {
			try {
				const parsed = new URL(normalized);
				normalized = parsed.pathname || '';
				if (parsed.host && parsed.host !== 'localhost') {
					normalized = `//${parsed.host}${normalized}`;
				}
				if (/^\/[A-Za-z]:\//.test(normalized)) {
					normalized = normalized.slice(1);
				}
			} catch {
				// keep the original input if URL parsing fails
			}
		}

		normalized = this.safeDecodeURIComponent(normalized);
		normalized = normalized.replace(/\\ /g, ' ');
		return normalized;
	}

	initVideoState(
		dir: string,
		options?: Omit<VideoInitStateRequest, 'video_frames_dir'>
	): Observable<JobStartResponse> {
		const payload: VideoInitStateRequest = {
			video_frames_dir: this.normalizePath(dir),
			...options
		};
		return this.http.post<JobStartResponse>(this.endpoint('/video/init_state'), payload);
	}

	resetVideoState(): Observable<any> {
		return this.http.post(this.endpoint('/video/reset_state'), {});
	}

	addNewPointsOrBox(request: VideoAddPointsOrBoxRequest): Observable<VideoAddPointsResponse> {
		return this.http.post<VideoAddPointsResponse>(this.endpoint('/video/add_new_points_or_box'), request);
	}

	propagateInVideo(request: VideoPropagateRequest): Observable<JobStartResponse> {
		return this.http.post<JobStartResponse>(this.endpoint('/video/propagate_in_video'), request);
	}

	saveVideoSession(name: string, interactiveState?: VideoSaveInteractiveState): Observable<VideoSaveResponse> {
		const payload: VideoSaveRequest = {
			name,
			...(interactiveState ? { interactive_state: interactiveState } : {}),
		};
		return this.http.post<VideoSaveResponse>(this.endpoint('/video/save'), payload);
	}

	clearAllPromptsInFrame(frameIdx: number, objId: number): Observable<any> {
		return this.http.post(this.endpoint('/video/clear_all_prompts_in_frame'), null, {
			params: { frame_idx: frameIdx.toString(), obj_id: objId.toString() }
		});
	}

	removeObject(objId: number): Observable<any> {
		return this.http.post(this.endpoint('/video/remove_object'), null, {
			params: { obj_id: objId.toString() }
		});
	}

	getVideoInfo(): Observable<{ num_frames: number, frame_files: string[] }> {
		return this.http.get<{ num_frames: number, frame_files: string[] }>(this.endpoint('/video/info'));
	}

	getVideoFrameUrl(frameIdx: number): string {
		return this.endpoint(`/video/frame/${frameIdx}`);
	}

	getVideoMaskFrameUrl(frameIdx: number): string {
		return this.endpoint(`/video/mask_frame/${frameIdx}`);
	}

	getVideoMaskData(frameIdx: number): Observable<VideoMaskDataResponse> {
		return this.http.get<VideoMaskDataResponse>(this.endpoint(`/video/mask_data/${frameIdx}`));
	}

	trackPromptPoints(request: TrackPromptPointsRequest): Observable<JobStartResponse> {
		return this.http.post<JobStartResponse>(this.endpoint('/tracking/track_prompt_points'), request);
	}
}
