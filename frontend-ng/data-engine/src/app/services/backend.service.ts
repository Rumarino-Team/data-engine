import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, timeout } from 'rxjs';

import { DEFAULT_API_URL, normalizeApiUrl, readStoredApiUrl, writeStoredApiUrl } from './backend-api-url';
import { normalizeBackendPath } from './backend-path';
import type {
	CurrentJobResponse,
	HealthResponse,
	JobResponse,
	JobStartResponse,
	TrackPromptPointsRequest,
	TrackPromptPointsResult,
	VideoAddPointsOrBoxRequest,
	VideoAddPointsResponse,
	VideoInitStateRequest,
	VideoMaskDataResponse,
	VideoPropagateRequest,
	VideoSaveInteractiveState,
	VideoSaveRequest,
	VideoSaveResponse,
} from './backend-api.types';
export * from './backend-api.types';

@Injectable({
	providedIn: 'root'
})
export class BackendService {
	private apiUrl = this.resolveApiUrl();

	constructor(private http: HttpClient) { }

	private resolveApiUrl(): string {
		const persistedConfig = readStoredApiUrl();
		const globalConfig = (globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__;
		return normalizeApiUrl(persistedConfig || globalConfig || DEFAULT_API_URL);
	}

	private endpoint(path: string): string {
		return `${this.apiUrl}${path}`;
	}

	getApiUrl(): string {
		return this.apiUrl;
	}

	setApiUrl(value: string): string {
		this.apiUrl = normalizeApiUrl(value);
		writeStoredApiUrl(this.apiUrl);
		return this.apiUrl;
	}

	resetApiUrl(): string {
		this.apiUrl = DEFAULT_API_URL;
		writeStoredApiUrl(null);
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

	clearJobResult(jobId: string): Observable<{ cleared: boolean }> {
		return this.http.post<{ cleared: boolean }>(this.endpoint(`/jobs/${jobId}/clear_result`), {});
	}

	initVideoState(
		dir: string,
		options?: Omit<VideoInitStateRequest, 'video_frames_dir'>
	): Observable<JobStartResponse> {
		const payload: VideoInitStateRequest = {
			video_frames_dir: normalizeBackendPath(dir),
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

	getTrackingResult(resultId: string): Observable<{ result: TrackPromptPointsResult }> {
		return this.http.get<{ result: TrackPromptPointsResult }>(this.endpoint(`/tracking/results/${resultId}`));
	}
}
