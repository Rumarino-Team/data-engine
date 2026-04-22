import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

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

@Injectable({
	providedIn: 'root'
})
export class BackendService {
	private readonly apiUrl = this.resolveApiUrl();

	constructor(private http: HttpClient) { }

	private resolveApiUrl(): string {
		const globalConfig = (globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__;
		const localStorageConfig = typeof localStorage !== 'undefined'
			? localStorage.getItem('dataEngineApiUrl')
			: null;

		const browserHost = typeof window !== 'undefined' && window.location.hostname
			? window.location.hostname
			: '127.0.0.1';
		const fallback = `http://${browserHost}:8000`;
		return (globalConfig || localStorageConfig || fallback).replace(/\/+$/, '');
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
	): Observable<VideoInitStateResponse> {
		const payload: VideoInitStateRequest = {
			video_frames_dir: this.normalizePath(dir),
			...options
		};
		return this.http.post<VideoInitStateResponse>(`${this.apiUrl}/video/init_state`, payload);
	}

	resetVideoState(): Observable<any> {
		return this.http.post(`${this.apiUrl}/video/reset_state`, {});
	}

	addNewPointsOrBox(request: VideoAddPointsOrBoxRequest): Observable<VideoAddPointsResponse> {
		return this.http.post<VideoAddPointsResponse>(`${this.apiUrl}/video/add_new_points_or_box`, request);
	}

	propagateInVideo(request: VideoPropagateRequest): Observable<VideoPropagateResponse> {
		return this.http.post<VideoPropagateResponse>(`${this.apiUrl}/video/propagate_in_video`, request);
	}

	clearAllPromptsInFrame(frameIdx: number, objId: number): Observable<any> {
		return this.http.post(`${this.apiUrl}/video/clear_all_prompts_in_frame`, null, {
			params: { frame_idx: frameIdx.toString(), obj_id: objId.toString() }
		});
	}

	removeObject(objId: number): Observable<any> {
		return this.http.post(`${this.apiUrl}/video/remove_object`, null, {
			params: { obj_id: objId.toString() }
		});
	}

	getVideoInfo(): Observable<{ num_frames: number, frame_files: string[] }> {
		return this.http.get<{ num_frames: number, frame_files: string[] }>(`${this.apiUrl}/video/info`);
	}

	getVideoFrameUrl(frameIdx: number): string {
		return `${this.apiUrl}/video/frame/${frameIdx}`;
	}

	getVideoMaskFrameUrl(frameIdx: number): string {
		return `${this.apiUrl}/video/mask_frame/${frameIdx}`;
	}

	getVideoMaskData(frameIdx: number): Observable<VideoMaskDataResponse> {
		return this.http.get<VideoMaskDataResponse>(`${this.apiUrl}/video/mask_data/${frameIdx}`);
	}

	trackPromptPoints(request: TrackPromptPointsRequest): Observable<TrackPromptPointsResponse> {
		return this.http.post<TrackPromptPointsResponse>(`${this.apiUrl}/tracking/track_prompt_points`, request);
	}
}
