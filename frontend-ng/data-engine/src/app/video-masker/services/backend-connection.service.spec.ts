import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BackendService } from '../../services/backend.service';
import { BackendConnectionService } from './backend-connection.service';
import { FramePipelineService } from './frame-pipeline.service';
import { SessionLeaveService } from './session-leave.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

describe('BackendConnectionService health monitoring', () => {
  const backend = {
    health: vi.fn(),
    getApiUrl: vi.fn(() => 'http://old'),
    setApiUrl: vi.fn((url: string) => url),
    resetApiUrl: vi.fn(() => 'http://default'),
  };
  const leave = { confirmLeave: vi.fn(async (reason: string) => reason !== '') };
  beforeEach(() => {
    vi.useFakeTimers();
    backend.health.mockReset();
    backend.setApiUrl.mockClear();
    backend.resetApiUrl.mockClear();
    leave.confirmLeave.mockReset();
    leave.confirmLeave.mockResolvedValue(true);
    TestBed.configureTestingModule({
      providers: [
        BackendConnectionService,
        { provide: FramePipelineService, useValue: { clearFrameCaches: vi.fn() } },
        VideoMaskerStateStore,
        { provide: BackendService, useValue: backend },
        { provide: SessionLeaveService, useValue: leave },
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

  describe('unsaved changes', () => {
    const openDirtySession = () => {
      backend.health.mockReturnValue(of({}));
      const service = TestBed.inject(BackendConnectionService);
      const store = TestBed.inject(VideoMaskerStateStore);
      store.isInitialized.set(true);
      store.sessionId.set('old-session');
      store.savedState.set({
        checkpoint_id: 'cp-1',
        display_name: 'Bedroom',
        saved_at: 'then',
        dirty: true,
      });
      return { service, store };
    };

    it('asks before a backend switch drops unsaved changes, then switches once confirmed', async () => {
      const { service, store } = openDirtySession();
      store.apiUrlInput.set('http://new');
      service.applyApiUrl();
      expect(leave.confirmLeave).toHaveBeenCalledWith('switch backends');
      expect(backend.setApiUrl).not.toHaveBeenCalled();
      expect(store.sessionId()).toBe('old-session');
      await vi.advanceTimersByTimeAsync(0);
      expect(backend.setApiUrl).toHaveBeenCalledWith('http://new');
      expect(store.sessionId()).toBeNull();
    });

    it('keeps the backend and the session when the switch is declined', async () => {
      const { service, store } = openDirtySession();
      leave.confirmLeave.mockResolvedValue(false);
      store.apiUrlInput.set('http://new');
      service.applyApiUrl();
      service.resetApiUrl();
      await vi.advanceTimersByTimeAsync(0);
      expect(backend.setApiUrl).not.toHaveBeenCalled();
      expect(backend.resetApiUrl).not.toHaveBeenCalled();
      expect(store.sessionId()).toBe('old-session');
    });

    it('asks before a reset to the default URL as well', async () => {
      const { service, store } = openDirtySession();
      service.resetApiUrl();
      expect(leave.confirmLeave).toHaveBeenCalledWith('switch backends');
      await vi.advanceTimersByTimeAsync(0);
      expect(backend.resetApiUrl).toHaveBeenCalledOnce();
      expect(store.sessionId()).toBeNull();
    });

    it('does not ask when the URL would not change or nothing is unsaved', () => {
      const { service, store } = openDirtySession();
      store.apiUrlInput.set('http://old');
      service.applyApiUrl();
      expect(leave.confirmLeave).not.toHaveBeenCalled();
      store.savedState.update((saved) => (saved ? { ...saved, dirty: false } : saved));
      store.apiUrlInput.set('http://new');
      service.applyApiUrl();
      expect(leave.confirmLeave).not.toHaveBeenCalled();
      expect(backend.setApiUrl).toHaveBeenCalledWith('http://new');
    });

    it('does not switch if work starts while the question is open', async () => {
      const { service, store } = openDirtySession();
      store.apiUrlInput.set('http://new');
      service.applyApiUrl();
      store.isLoading.set(true);
      await vi.advanceTimersByTimeAsync(0);
      expect(backend.setApiUrl).not.toHaveBeenCalled();
    });
  });
});
