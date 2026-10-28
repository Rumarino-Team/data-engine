import { DestroyRef } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
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

  it('retries a transient progress read without submitting the job again', async () => {
    vi.useFakeTimers();
    const startJob = vi.fn(async () => ({ job_id: 'job-1' }));
    const backend = {
      getJob: vi
        .fn()
        .mockReturnValueOnce(throwError(() => ({ status: 0 })))
        .mockReturnValueOnce(of({ job: { status: 'running' } }))
        .mockReturnValue(of({ job: { status: 'completed', result: 42 } })),
    };
    const onFailure = vi.fn();
    const run = new VideoJobsService(backend as unknown as BackendService).run({
      title: 'Test',
      startJob,
      onFailure,
    });
    await vi.advanceTimersByTimeAsync(499);
    expect(backend.getJob).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(backend.getJob).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(500);
    expect(await run).toEqual({ result: 42, completedJobId: 'job-1' });
    expect(startJob).toHaveBeenCalledOnce();
    expect(onFailure).not.toHaveBeenCalled();
  });

  it('times out and unsubscribes a hung progress read before retrying', async () => {
    vi.useFakeTimers();
    const pending = new Subject<any>();
    const backend = {
      getJob: vi
        .fn()
        .mockReturnValueOnce(pending)
        .mockReturnValue(of({ job: { status: 'completed', result: 42 } })),
    };
    const run = new VideoJobsService(backend as unknown as BackendService).run({
      title: 'Test',
      startJob: async () => ({ job_id: 'job-1' }),
    });
    await vi.advanceTimersByTimeAsync(10000);
    expect(pending.observed).toBe(false);
    expect(backend.getJob).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(500);
    expect(await run).toEqual({ result: 42, completedJobId: 'job-1' });
  });

  it('bounds progress retries and reports an unknown outcome instead of a failed computation', async () => {
    vi.useFakeTimers();
    const backend = { getJob: vi.fn(() => throwError(() => ({ status: 503 }))) };
    const startJob = vi.fn(async () => ({ job_id: 'job-1' }));
    const onFailure = vi.fn();
    const onFinish = vi.fn();
    const run = new VideoJobsService(backend as unknown as BackendService).run({
      title: 'Test',
      startJob,
      onFailure,
      onFinish,
    });
    await vi.advanceTimersByTimeAsync(1500);
    expect(await run).toEqual({ result: null, completedJobId: null });
    expect(backend.getJob).toHaveBeenCalledTimes(3);
    expect(startJob).toHaveBeenCalledOnce();
    expect(onFailure).toHaveBeenCalledWith(
      'Unable to check job progress. The operation may still be running.',
    );
    expect(onFinish).toHaveBeenCalledOnce();
  });

  it('does not retry job submission or non-transient progress errors', async () => {
    const startJob = vi.fn(async () => {
      throw { status: 0 };
    });
    const backend = {
      getJob: vi.fn(() => throwError(() => ({ status: 404, error: { detail: 'Job not found.' } }))),
    };
    const onFailure = vi.fn();
    const service = new VideoJobsService(backend as unknown as BackendService);
    await service.run({ title: 'Test', startJob, onFailure });
    expect(startJob).toHaveBeenCalledOnce();
    expect(backend.getJob).not.toHaveBeenCalled();
    await service.run({ title: 'Test', startJob: async () => ({ job_id: 'job-1' }), onFailure });
    expect(backend.getJob).toHaveBeenCalledOnce();
    expect(onFailure).toHaveBeenLastCalledWith('Job not found.');
  });

  it('cancels a scheduled progress retry when the owner is destroyed', async () => {
    vi.useFakeTimers();
    const backend = { getJob: vi.fn(() => throwError(() => ({ status: 0 }))) };
    const onFailure = vi.fn();
    const run = new VideoJobsService(backend as unknown as BackendService).run({
      title: 'Test',
      startJob: async () => ({ job_id: 'job-1' }),
      destroyRef: TestBed.inject(DestroyRef),
      onFailure,
    });
    await vi.advanceTimersByTimeAsync(0);
    TestBed.resetTestingModule();
    expect(await run).toEqual({ result: null, completedJobId: null });
    await vi.advanceTimersByTimeAsync(1500);
    expect(backend.getJob).toHaveBeenCalledOnce();
    expect(onFailure).not.toHaveBeenCalled();
  });
});
