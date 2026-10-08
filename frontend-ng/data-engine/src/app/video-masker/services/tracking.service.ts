import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import {
  BackendService,
  TrackPromptPointsJobResponse,
  TrackPromptPointsResult,
} from '../../services/backend.service';
import { VideoMaskerStateStore, TrackedPointSeries } from './video-masker-state.store';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import { EditorJobService } from './editor-job.service';
import { getErrorMessage } from '../video-masker.util';

/** Runs point tracking and rejects results invalidated by later edits. */
@Injectable()
export class TrackingService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly framePipeline = inject(FramePipelineService);
  private readonly toast = inject(ToastService);
  private readonly jobs = inject(EditorJobService);
  private get isBusy(): boolean {
    return this.store.isInteractionBusy();
  }

  async runTracking(): Promise<void> {
    if (this.isBusy) return;
    const { result: response, completedJobId } = await this.jobs.run<TrackPromptPointsJobResponse>(
      'Tracking prompt points',
      () =>
        firstValueFrom(
          this.backend.trackPromptPoints({ add_support_grid: this.store.trackingUseSupportGrid() }),
        ),
    );
    if (!response) {
      return;
    }
    this.store.updateStateEpoch(response.state_epoch, 'tracking restore');
    const loaded = await this.loadTrackingResult(response.tracking_result_id, 'Tracking');
    if (loaded && completedJobId) {
      try {
        await firstValueFrom(this.backend.clearJobResult(completedJobId));
      } catch (error) {
        console.error(error);
      }
    }
  }

  async loadTrackingResult(resultId: string, sourceLabel: string): Promise<boolean> {
    const revision = this.store.trackingRevision;
    const epoch = this.store.stateEpoch();
    try {
      const response = await firstValueFrom(this.backend.getTrackingResult(resultId));
      if (revision !== this.store.trackingRevision || epoch !== this.store.stateEpoch())
        return false;
      const result: TrackPromptPointsResult = response.result;
      const trackedSeries: TrackedPointSeries[] = result.points.map((point, index) => ({
        ...point,
        tracks: result.tracks[index] || [],
        visibility: result.visibility[index] || [],
      }));
      this.store.trackedPoints.set(trackedSeries);
      this.framePipeline.redraw();
      return true;
    } catch (error) {
      console.error(error);
      this.toast.show(
        'warning',
        'Tracking result unavailable',
        getErrorMessage(error, `${sourceLabel} result could not be loaded.`),
      );
      return false;
    }
  }
}
