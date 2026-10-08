import { Injectable, inject } from '@angular/core';
import { VideoJobsService } from './video-jobs.service';
import { VideoMaskerStateStore } from './video-masker-state.store';
import { ToastService } from './toast.service';

/** Keeps the editor busy and displays progress for one backend job. */
@Injectable()
export class EditorJobService {
  private readonly jobs = inject(VideoJobsService);
  private readonly store = inject(VideoMaskerStateStore);
  private readonly toast = inject(ToastService);

  async run<T>(
    title: string,
    startJob: () => Promise<{ job_id: string }>,
  ): Promise<{ result: T | null; completedJobId: string | null }> {
    return this.jobs.run<T>({
      title,
      startJob,
      onStart: () => {
        this.store.isLoading.set(true);
        this.store.activeJobTitle.set(title);
        this.store.activeJob.set(null);
      },
      onStatus: (job) => this.store.activeJob.set(job),
      onFailure: (message) => this.toast.show('error', title, message),
      onFinish: () => {
        this.store.activeJob.set(null);
        this.store.activeJobTitle.set('');
        this.store.isLoading.set(false);
      },
      fallbackErrorMessage: `${title} failed`,
    });
  }
}
