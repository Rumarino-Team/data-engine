import { Injectable } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { BackendJob, BackendService } from '../../services/backend.service';

export interface JobRunnerOptions {
  title: string;
  startJob: () => Promise<{ job_id: string }>;
  onStatus?: (job: BackendJob) => void;
  onStart?: () => void;
  onFinish?: () => void;
  onFailure?: (message: string) => void;
  pollIntervalMs?: number;
  fallbackErrorMessage?: string;
}

@Injectable({ providedIn: 'root' })
export class VideoJobsService {
  constructor(private backend: BackendService) {}

  async run<T>(
    options: JobRunnerOptions,
  ): Promise<{ result: T | null; completedJobId: string | null }> {
    const pollIntervalMs = options.pollIntervalMs ?? 500;
    let completedJobId: string | null = null;

    options.onStart?.();
    try {
      const started = await options.startJob();
      while (true) {
        const response = await firstValueFrom(this.backend.getJob<T>(started.job_id));
        options.onStatus?.(response.job);
        if (response.job.status === 'completed') {
          completedJobId = started.job_id;
          return { result: response.job.result as T, completedJobId };
        }
        if (response.job.status === 'failed') {
          const message =
            response.job.error?.message || response.job.message || `${options.title} failed`;
          options.onFailure?.(message);
          return { result: null, completedJobId };
        }
        await this.delay(pollIntervalMs);
      }
    } catch {
      options.onFailure?.(options.fallbackErrorMessage || `${options.title} failed`);
      return { result: null, completedJobId };
    } finally {
      options.onFinish?.();
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
