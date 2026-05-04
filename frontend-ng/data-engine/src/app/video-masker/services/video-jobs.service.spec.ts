import { of } from 'rxjs';
import { describe, expect, it, vi } from 'vitest';
import { BackendService } from '../../services/backend.service';
import { VideoJobsService } from './video-jobs.service';

describe('VideoJobsService', () => {
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
