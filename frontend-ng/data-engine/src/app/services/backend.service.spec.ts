import { HttpClient } from '@angular/common/http';
import { of } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BackendService } from './backend.service';

describe('BackendService', () => {
  const originalGlobalApiUrl = (globalThis as { __DATA_ENGINE_API_URL__?: string })
    .__DATA_ENGINE_API_URL__;
  const originalLocalStorage = globalThis.localStorage;
  let storage: Record<string, string>;

  beforeEach(() => {
    storage = {};
    Object.defineProperty(globalThis, 'localStorage', {
      value: {
        getItem: vi.fn((key: string) => storage[key] ?? null),
        setItem: vi.fn((key: string, value: string) => {
          storage[key] = String(value);
        }),
        removeItem: vi.fn((key: string) => {
          delete storage[key];
        }),
        clear: vi.fn(() => {
          storage = {};
        }),
      },
      configurable: true,
    });
    localStorage.clear();
    delete (globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__;
  });

  afterEach(() => {
    localStorage.clear();
    Object.defineProperty(globalThis, 'localStorage', {
      value: originalLocalStorage,
      configurable: true,
    });
    if (originalGlobalApiUrl === undefined) {
      delete (globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__;
      return;
    }
    (globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__ =
      originalGlobalApiUrl;
  });

  it('defaults to localhost when no override exists', () => {
    const http = { post: vi.fn(), get: vi.fn() } as unknown as HttpClient;

    const service = new BackendService(http);

    expect(service.getApiUrl()).toBe('http://127.0.0.1:8000');
  });

  it('prefers stored API URL over global config', () => {
    localStorage.setItem('dataEngineApiUrl', 'http://192.168.0.50:9000/');
    (globalThis as { __DATA_ENGINE_API_URL__?: string }).__DATA_ENGINE_API_URL__ =
      'http://10.0.0.1:8000';
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

  it('fetches mask data windows with object filters', () => {
    const http = {
      post: vi.fn(),
      get: vi.fn(() => of({})),
    } as unknown as HttpClient;
    const service = new BackendService(http);

    service.getVideoMaskDataWindow(2, 8, [1, 5]).subscribe();

    expect((http.get as any).mock.calls[0][0]).toBe(
      'http://127.0.0.1:8000/video/mask_data_window',
    );
    expect((http.get as any).mock.calls[0][1]).toEqual({
      params: {
        start_frame_idx: '2',
        end_frame_idx: '8',
        object_ids: '1,5',
      },
    });
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

  it('starts prompt tracking without a model selector payload', () => {
    const http = {
      post: vi.fn(() => of({})),
      get: vi.fn(),
    } as unknown as HttpClient;
    const service = new BackendService(http);

    service.trackPromptPoints({ add_support_grid: true }).subscribe();

    expect((http.post as any).mock.calls[0][0]).toBe(
      'http://127.0.0.1:8000/tracking/track_prompt_points',
    );
    expect((http.post as any).mock.calls[0][1]).toEqual({ add_support_grid: true });
    expect((http.post as any).mock.calls[0][1]).not.toHaveProperty('model_name');
  });

  it('fetches a persisted tracking result by id', () => {
    const http = {
      post: vi.fn(),
      get: vi.fn(() => of({ result: {} })),
    } as unknown as HttpClient;
    const service = new BackendService(http);

    service.getTrackingResult('abc123').subscribe();

    expect((http.get as any).mock.calls[0][0]).toBe(
      'http://127.0.0.1:8000/tracking/results/abc123',
    );
  });

  it('clears a completed job result by id', () => {
    const http = {
      post: vi.fn(() => of({ cleared: true })),
      get: vi.fn(),
    } as unknown as HttpClient;
    const service = new BackendService(http);

    service.clearJobResult('job-1').subscribe();

    expect((http.post as any).mock.calls[0][0]).toBe(
      'http://127.0.0.1:8000/jobs/job-1/clear_result',
    );
    expect((http.post as any).mock.calls[0][1]).toEqual({});
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

    service
      .saveVideoSession('review-run', {
        version: 1,
        objects: [{ id: 1, name: 'Object 1', color: '#ff6600' }],
        selected_object_id: 1,
        interaction_mode: 'positive',
        current_frame_idx: 3,
        points: [{ frame_idx: 3, obj_id: 1, x: 10, y: 20, label: 1 }],
        live_masks: [{ frame_idx: 3, obj_id: 1, height: 2, width: 2, counts: [0, 1, 3] }],
      })
      .subscribe();

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
