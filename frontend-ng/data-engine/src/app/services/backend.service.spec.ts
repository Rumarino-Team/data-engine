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

  it('sends required identity and revision headers on every object mutation', () => {
    const http = { post: vi.fn(), put: vi.fn() };
    const service = new BackendService(http as unknown as HttpClient);
    const precondition = { session_id: 'session-a', expected_revision: 7 };
    const headers = { 'X-Session-ID': 'session-a', 'X-Annotation-Revision': '7' };
    service.createObject(precondition);
    service.updateObject({ id: 2, label: 'Hand', color: '#123456' }, precondition);
    service.removeObject(2, precondition);
    service.resetVideoState(precondition);
    service.clearAllPromptsInFrame(3, 2, precondition);
    const prompt = { frame_idx: 3, obj_id: 2, points: [[1, 2]], labels: [1] };
    service.addNewPointsOrBox(prompt, precondition);
    expect(http.post).toHaveBeenCalledWith('http://127.0.0.1:8000/video/objects', null, {
      headers,
    });
    expect(http.post).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/video/reset_state',
      {},
      { headers },
    );
    expect(http.post).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/video/clear_all_prompts_in_frame',
      null,
      { headers, params: { frame_idx: '3', obj_id: '2' } },
    );
    expect(http.post).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/video/add_new_points_or_box',
      prompt,
      { headers },
    );
    expect(http.put).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/video/objects/2',
      { label: 'Hand', color: '#123456' },
      { headers },
    );
    expect(http.post).toHaveBeenCalledWith('http://127.0.0.1:8000/video/remove_object', null, {
      headers,
      params: { obj_id: '2' },
    });
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

  it('resets the API URL back to localhost and clears persisted override', () => {
    const http = { post: vi.fn(), get: vi.fn() } as unknown as HttpClient;
    const service = new BackendService(http);

    service.setApiUrl('http://localhost:9001');
    service.resetApiUrl();

    expect(service.getApiUrl()).toBe('http://127.0.0.1:8000');
    expect(localStorage.getItem('dataEngineApiUrl')).toBeNull();
  });

  describe('request shapes', () => {
    const precondition = { session_id: 'session-1', expected_revision: 0 };
    const preferences = {
      version: 1,
      selected_object_id: 1,
      interaction_mode: 'positive' as const,
      current_frame_idx: 3,
    };
    type Call = (service: BackendService) => { subscribe: () => unknown };
    const cases: [string, Call, 'get' | 'post', string, unknown?][] = [
      ['health', (s) => s.health(), 'get', '/health'],
      ['job status', (s) => s.getJob('abc123'), 'get', '/jobs/abc123'],
      ['tracking result', (s) => s.getTrackingResult('abc123'), 'get', '/tracking/results/abc123'],
      [
        'clear job result',
        (s) => s.clearJobResult('job-1'),
        'post',
        '/jobs/job-1/clear_result',
        {},
      ],
      [
        'prompt tracking without a model selector',
        (s) => s.trackPromptPoints({ add_support_grid: true }, precondition),
        'post',
        '/tracking/track_prompt_points',
        { add_support_grid: true },
      ],
      [
        'first save by name',
        (s) => s.saveVideoSession({ name: 'review-run' }),
        'post',
        '/video/save',
        { name: 'review-run' },
      ],
      ['re-save without a name', (s) => s.saveVideoSession({}), 'post', '/video/save', {}],
      [
        'Save As',
        (s) => s.saveVideoSession({ name: 'copy', saveAs: true }),
        'post',
        '/video/save',
        { name: 'copy', save_as: true },
      ],
      [
        'save with editor preferences',
        (s) => s.saveVideoSession({ name: 'review-run' }, preferences),
        'post',
        '/video/save',
        { name: 'review-run', interactive_state: preferences },
      ],
    ];

    it.each(cases)('%s', (_name, call, method, path, body) => {
      const http = { get: vi.fn(() => of({})), post: vi.fn(() => of({})) };
      call(new BackendService(http as unknown as HttpClient)).subscribe();
      const request = http[method].mock.calls[0] as unknown[];
      expect(request[0]).toBe(`http://127.0.0.1:8000${path}`);
      if (body !== undefined) expect(request[1]).toEqual(body);
    });
  });
});
