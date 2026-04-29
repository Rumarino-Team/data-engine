import { HttpClient } from '@angular/common/http';
import { of } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BackendService } from './backend.service';

describe('BackendService', () => {
	const originalGlobalApiUrl = (globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__;

	beforeEach(() => {
		localStorage.clear();
		delete (globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__;
	});

	afterEach(() => {
		localStorage.clear();
		if (originalGlobalApiUrl === undefined) {
			delete (globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__;
			return;
		}
		(globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__ = originalGlobalApiUrl;
	});

	it('defaults to localhost when no override exists', () => {
		const http = { post: vi.fn(), get: vi.fn() } as unknown as HttpClient;

		const service = new BackendService(http);

		expect(service.getApiUrl()).toBe('http://127.0.0.1:8000');
	});

	it('prefers stored API URL over global config', () => {
		localStorage.setItem('dataEngineApiUrl', 'http://192.168.0.50:9000/');
		(globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__ = 'http://10.0.0.1:8000';
		const http = { post: vi.fn(), get: vi.fn() } as unknown as HttpClient;

		const service = new BackendService(http);

		expect(service.getApiUrl()).toBe('http://192.168.0.50:9000');
	});

	it('updates future requests after applying a new API URL', () => {
		const http = {
			post: vi.fn(() => of({})),
			get: vi.fn(() => of({})),
		} as unknown as HttpClient;
		const service = new BackendService(http);

		service.setApiUrl('http://localhost:9001/');
		service.getVideoMaskData(3).subscribe();

		expect(service.getApiUrl()).toBe('http://localhost:9001');
		expect(localStorage.getItem('dataEngineApiUrl')).toBe('http://localhost:9001');
		expect((http.get as any).mock.calls[0][0]).toBe('http://localhost:9001/video/mask_data/3');
	});

	it('resets the API URL back to localhost and clears persisted override', () => {
		const http = { post: vi.fn(), get: vi.fn() } as unknown as HttpClient;
		const service = new BackendService(http);

		service.setApiUrl('http://localhost:9001');
		service.resetApiUrl();

		expect(service.getApiUrl()).toBe('http://127.0.0.1:8000');
		expect(localStorage.getItem('dataEngineApiUrl')).toBeNull();
	});

	it('calls the health endpoint', () => {
		const http = {
			post: vi.fn(),
			get: vi.fn(() => of({ status: 'ok' })),
		} as unknown as HttpClient;
		const service = new BackendService(http);

		service.health().subscribe();

		expect((http.get as any).mock.calls[0][0]).toBe('http://127.0.0.1:8000/health');
	});

	it('fetches job status by id', () => {
		const http = {
			post: vi.fn(),
			get: vi.fn(() => of({ job: null })),
		} as unknown as HttpClient;
		const service = new BackendService(http);

		service.getJob('abc123').subscribe();

		expect((http.get as any).mock.calls[0][0]).toBe('http://127.0.0.1:8000/jobs/abc123');
	});

	it('saves a video session by name', () => {
		const http = {
			post: vi.fn(() => of({})),
			get: vi.fn(),
		} as unknown as HttpClient;
		const service = new BackendService(http);

		service.saveVideoSession('review-run').subscribe();

		expect((http.post as any).mock.calls[0][0]).toBe('http://127.0.0.1:8000/video/save');
		expect((http.post as any).mock.calls[0][1]).toEqual({ name: 'review-run' });
	});

	it('sends interactive state when saving a session snapshot', () => {
		const http = {
			post: vi.fn(() => of({})),
			get: vi.fn(),
		} as unknown as HttpClient;
		const service = new BackendService(http);

		service.saveVideoSession('review-run', {
			version: 1,
			objects: [{ id: 1, name: 'Object 1', color: '#ff6600' }],
			selected_object_id: 1,
			interaction_mode: 'positive',
			current_frame_idx: 3,
			points: [{ frame_idx: 3, obj_id: 1, x: 10, y: 20, label: 1 }],
			live_masks: [{ frame_idx: 3, obj_id: 1, height: 2, width: 2, counts: [0, 1, 3] }],
		}).subscribe();

		expect((http.post as any).mock.calls[0][0]).toBe('http://127.0.0.1:8000/video/save');
		expect((http.post as any).mock.calls[0][1]).toEqual({
			name: 'review-run',
			interactive_state: {
				version: 1,
				objects: [{ id: 1, name: 'Object 1', color: '#ff6600' }],
				selected_object_id: 1,
				interaction_mode: 'positive',
				current_frame_idx: 3,
				points: [{ frame_idx: 3, obj_id: 1, x: 10, y: 20, label: 1 }],
				live_masks: [{ frame_idx: 3, obj_id: 1, height: 2, width: 2, counts: [0, 1, 3] }],
			},
		});
	});
});
