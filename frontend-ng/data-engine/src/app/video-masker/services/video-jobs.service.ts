import { DestroyRef, Injectable } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  defer,
  lastValueFrom,
  repeat,
  retry,
  switchMap,
  takeWhile,
  tap,
  throwError,
  timeout,
  TimeoutError,
  timer,
} from 'rxjs';
import { BackendJob, BackendService } from '../../services/backend.service';
import { getErrorMessage } from '../video-masker.util';

export interface JobRunnerOptions {
  title: string;
  destroyRef?: DestroyRef;
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
    let pollingStarted = false;
    const transientPollFailure = (error: unknown): boolean => {
      const status = (error as { status?: number } | null)?.status;
      return (
        error instanceof TimeoutError || status === 0 || (status !== undefined && status >= 500)
      );
    };

    if (options.destroyRef?.destroyed) {
      return { result: null, completedJobId };
    }
    options.onStart?.();
    try {
      // Repeat only after a response completes, preserving the delay between requests.
      let statuses = defer(options.startJob).pipe(
        switchMap((started) =>
          defer(() => {
            pollingStarted = true;
            return this.backend.getJob<T>(started.job_id);
          }).pipe(
            // Retry only this read, never the job submission. Each retry gets
            // its own timeout and polling remains sequential.
            timeout(10000),
            retry({
              count: 2,
              delay: (error, attempt) =>
                transientPollFailure(error)
                  ? timer(pollIntervalMs * attempt)
                  : throwError(() => error),
            }),
            repeat({ delay: pollIntervalMs }),
            tap((response) => {
              options.onStatus?.(response.job);
              if (response.job.status === 'completed') completedJobId = started.job_id;
            }),
            takeWhile((response) => !['completed', 'failed'].includes(response.job.status), true),
          ),
        ),
      );
      if (options.destroyRef) {
        statuses = statuses.pipe(takeUntilDestroyed(options.destroyRef));
      }
      const response = await lastValueFrom(statuses, { defaultValue: null });
      if (options.destroyRef?.destroyed || !response) {
        return { result: null, completedJobId: null };
      }
      if (response.job.status === 'completed') {
        return { result: response.job.result as T, completedJobId };
      }
      const message =
        response.job.error?.message || response.job.message || `${options.title} failed`;
      options.onFailure?.(message);
      return { result: null, completedJobId };
    } catch (error) {
      options.onFailure?.(
        pollingStarted && transientPollFailure(error)
          ? 'Unable to check job progress. The operation may still be running.'
          : getErrorMessage(error, options.fallbackErrorMessage || `${options.title} failed`),
      );
      return { result: null, completedJobId };
    } finally {
      options.onFinish?.();
    }
  }
}
