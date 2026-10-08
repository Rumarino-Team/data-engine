import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BackendService } from '../../services/backend.service';
import { BackendConnectionService } from './backend-connection.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

describe('BackendConnectionService health monitoring', () => {
  const backend = {
    health: vi.fn(),
    getApiUrl: vi.fn(() => 'http://old'),
    setApiUrl: vi.fn((url: string) => url),
  };
  beforeEach(() => {
    vi.useFakeTimers();
    backend.health.mockReset();
    TestBed.configureTestingModule({
      providers: [
        BackendConnectionService,
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
});
