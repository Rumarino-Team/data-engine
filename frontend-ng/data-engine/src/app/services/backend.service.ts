import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface VideoInitStateRequest {
	video_frames_dir: string;
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
}

export interface VideoAddMaskRequest {
	frame_idx: number;
	obj_id: number;
	mask: boolean[][];
}

export interface VideoAddPointsResponse {
	out_obj_ids: number[];
	out_masks: boolean[][][]; // List of masks (which are 2D boolean arrays)
}

export interface VideoPropagateResponse {
	video_segments: { [frame_idx: string]: { [obj_id: string]: boolean[][] } };
	saved_mask_paths: { [frame_idx: string]: { [obj_id: string]: string } };
}

@Injectable({
	providedIn: 'root'
})
export class BackendService {
	private apiUrl = 'http://localhost:8000';

	constructor(private http: HttpClient) { }

	private normalizePath(path: string): string {
		return path.trim().replace(/^['\"]|['\"]$/g, '');
	}

	initVideoState(dir: string): Observable<any> {
		return this.http.post(`${this.apiUrl}/video/init_state`, { video_frames_dir: this.normalizePath(dir) });
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
}
