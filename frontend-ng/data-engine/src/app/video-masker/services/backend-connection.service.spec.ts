import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BackendService } from '../../services/backend.service';
import { BackendConnectionService } from './backend-connection.service';
import { FramePipelineService } from './frame-pipeline.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

describe('BackendConnectionService health monitoring', () => {
  const backend = {
    health: vi.fn(),
    getApiUrl: vi.fn(() => 'http://old'),
    setApiUrl: vi.fn((url: string) => url),
    resetApiUrl: vi.fn(() => 'http://default'),
  };
  beforeEach(() => {
    vi.useFakeTimers();
    backend.health.mockReset();
    backend.setApiUrl.mockClear();
    backend.resetApiUrl.mockClear();
    TestBed.configureTestingModule({
      providers: [
        BackendConnectionService,
        { provide: FramePipelineService, useValue: { clearFrameCaches: vi.fn() } },
        VideoMaskerStateStore,
        { provide: BackendService, useValue: backend },
      ],
    });
  });
  afterEach(() => {
    TestBed.resetTestingModule();
    vi.useRealTimers();
  });

  it('skips overlapping checks and stops requests when destroyed', async () => {
    const pending = new Subject<unknown>();
    backend.health.mockReturnValue(pending);
    const service = TestBed.inject(BackendConnectionService);
    service.checkApiHealth(true);
    await vi.advanceTimersByTimeAsync(9000);
    expect(backend.health).toHaveBeenCalledTimes(1);
    expect(pending.observed).toBe(true);
    TestBed.resetTestingModule();
    expect(pending.observed).toBe(false);
    await vi.advanceTimersByTimeAsync(6000);
    expect(backend.health).toHaveBeenCalledTimes(1);
  });

  it('continues monitoring after a failed check', async () => {
    backend.health
      .mockReturnValueOnce(throwError(() => new Error('offline')))
      .mockReturnValue(of({}));
    const service = TestBed.inject(BackendConnectionService);
    const store = TestBed.inject(VideoMaskerStateStore);
    service.checkApiHealth(true);
    await vi.advanceTimersByTimeAsync(0);
    expect(store.apiHealthStatus()).toBe('offline');
    await vi.advanceTimersByTimeAsync(3000);
    expect(store.apiHealthStatus()).toBe('online');
  });

  it('cancels the old URL check when changing URLs', async () => {
    const oldRequest = new Subject<unknown>();
    backend.health.mockReturnValueOnce(oldRequest).mockReturnValue(of({}));
    const service = TestBed.inject(BackendConnectionService);
    const store = TestBed.inject(VideoMaskerStateStore);
    service.checkApiHealth(true);
    await vi.advanceTimersByTimeAsync(0);
    store.apiUrlInput.set('http://new');
    service.applyApiUrl();
    expect(oldRequest.observed).toBe(false);
    await vi.advanceTimersByTimeAsync(0);
    expect(backend.setApiUrl).toHaveBeenCalledWith('http://new');
    expect(store.apiHealthStatus()).toBe('online');
  });
  it('clears accepted state, tracking, navigation and frame caches on backend switch', () => {
    backend.health.mockReturnValue(of({}));
    const service = TestBed.inject(BackendConnectionService);
    const store = TestBed.inject(VideoMaskerStateStore);
    store.isInitialized.set(true);
    store.sessionId.set('old-session');
    store.annotationsBackendUrl.set('http://old');
    store.objects.set([{ id: 1, label: 'Old', color: '#ff0000' }]);
    store.selectedObjectId.set(1);
    store.numFrames.set(12);
    store.displayedFrameIdx.set(7);
    store.hasManifestMasks.set(true);
    store.pendingPointEdit.set({ frameIdx: 7, objId: 1, points: [] });
    store.trackingResultId = 'old-tracks';
    store.apiUrlInput.set('http://new');
    service.applyApiUrl();
    expect(store.isInitialized()).toBe(false);
    expect(store.sessionId()).toBeNull();
    expect(store.annotationsBackendUrl()).toBeNull();
    expect(store.objects()).toEqual([]);
    expect(store.numFrames()).toBe(0);
    expect(store.displayedFrameIdx()).toBe(-1);
    expect(store.pendingPointEdit()).toBeNull();
    expect(store.trackingResultId).toBeNull();
    expect(store.hasManifestMasks()).toBe(false);
    expect(TestBed.inject(FramePipelineService).clearFrameCaches).toHaveBeenCalledOnce();
  });

  it('preserves the session for an unchanged URL and refuses switching during an edit', () => {
    backend.health.mockReturnValue(of({}));
    const service = TestBed.inject(BackendConnectionService);
    const store = TestBed.inject(VideoMaskerStateStore);
    store.sessionId.set('old-session');
    store.apiUrlInput.set('http://old');
    service.applyApiUrl();
    expect(store.sessionId()).toBe('old-session');
    expect(TestBed.inject(FramePipelineService).clearFrameCaches).not.toHaveBeenCalled();
    backend.setApiUrl.mockClear();
    store.isPointRequestInFlight.set(true);
    store.apiUrlInput.set('http://new');
    service.applyApiUrl();
    service.resetApiUrl();
    expect(backend.setApiUrl).not.toHaveBeenCalled();
    expect(backend.resetApiUrl).not.toHaveBeenCalled();
    expect(store.sessionId()).toBe('old-session');
  });

  it('clears the old session when resetting the backend URL', () => {
    backend.health.mockReturnValue(of({}));
    const service = TestBed.inject(BackendConnectionService);
    const store = TestBed.inject(VideoMaskerStateStore);
    store.sessionId.set('old-session');
    service.resetApiUrl();
    expect(store.sessionId()).toBeNull();
    expect(store.apiUrlInput()).toBe('http://default');
    expect(TestBed.inject(FramePipelineService).clearFrameCaches).toHaveBeenCalledOnce();
  });
});
