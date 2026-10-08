import { DestroyRef } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { Subject, of } from 'rxjs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { BackendService } from '../../services/backend.service';
import { VideoJobsService } from './video-jobs.service';

describe('VideoJobsService', () => {
  afterEach(() => {
    TestBed.resetTestingModule();
    vi.useRealTimers();
  });

  it('waits for a response and the delay before polling again', async () => {
    vi.useFakeTimers();
    const pending = new Subject<any>();
    const backend = {
      getJob: vi
        .fn()
        .mockReturnValueOnce(pending)
        .mockReturnValue(of({ job: { status: 'completed', result: 42 } })),
    };
    const onStatus = vi.fn();
    const onFinish = vi.fn();
    const service = new VideoJobsService(backend as unknown as BackendService);
    const run = service.run({
      title: 'Test',
      startJob: async () => ({ job_id: 'job-1' }),
      onStatus,
      onFinish,
    });
    await vi.advanceTimersByTimeAsync(2000);
    expect(backend.getJob).toHaveBeenCalledTimes(1);
    pending.next({ job: { status: 'running' } });
    pending.complete();
    await vi.advanceTimersByTimeAsync(499);
    expect(backend.getJob).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(await run).toEqual({ result: 42, completedJobId: 'job-1' });
    await vi.advanceTimersByTimeAsync(2000);
    expect(backend.getJob).toHaveBeenCalledTimes(2);
    expect(onStatus).toHaveBeenCalledTimes(2);
    expect(onFinish).toHaveBeenCalledTimes(1);
  });

  it('cancels an in-flight poll when the owner is destroyed', async () => {
    const pending = new Subject<any>();
    const backend = { getJob: vi.fn(() => pending) };
    const onFailure = vi.fn();
    const onFinish = vi.fn();
    const destroyRef = TestBed.inject(DestroyRef);
    const service = new VideoJobsService(backend as unknown as BackendService);
    const run = service.run({
      title: 'Test',
      startJob: async () => ({ job_id: 'job-1' }),
      destroyRef,
      onFailure,
      onFinish,
    });
    await Promise.resolve();
    expect(pending.observed).toBe(true);
    TestBed.resetTestingModule();
    expect(await run).toEqual({ result: null, completedJobId: null });
    expect(pending.observed).toBe(false);
    expect(onFailure).not.toHaveBeenCalled();
    expect(onFinish).toHaveBeenCalledTimes(1);
  });
  it('returns completed result and job id', async () => {
    const backend = {
      getJob: vi.fn(() => of({ job: { status: 'completed', result: { ok: true } } })),
    } as unknown as BackendService;
    const service = new VideoJobsService(backend);

    const out = await service.run<{ ok: boolean }>({
      title: 'Test job',
      startJob: async () => ({ job_id: 'job-1' }),
    });

    expect(out.completedJobId).toBe('job-1');
    expect(out.result).toEqual({ ok: true });
  });

  it('returns null result on failed status', async () => {
    const backend = {
      getJob: vi.fn(() => of({ job: { status: 'failed', message: 'bad run', error: null } })),
    } as unknown as BackendService;
    const service = new VideoJobsService(backend);
    const onFailure = vi.fn();

    const out = await service.run({
      title: 'Test job',
      startJob: async () => ({ job_id: 'job-1' }),
      onFailure,
    });

    expect(out.result).toBeNull();
    expect(onFailure).toHaveBeenCalledWith('bad run');
  });
});
