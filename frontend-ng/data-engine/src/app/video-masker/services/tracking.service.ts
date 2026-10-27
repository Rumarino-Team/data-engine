import { DestroyRef, Injectable, inject } from '@angular/core';
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

/** Displays backend-selected tracks while rejecting stale network responses. */
@Injectable()
export class TrackingService {
  private readonly destroyRef = inject(DestroyRef);
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
    const backendUrl = this.backend.getApiUrl();
    const sessionId = this.store.sessionId();
    const revision = this.store.annotationRevision();
    if (!sessionId || revision === null || backendUrl !== this.store.annotationsBackendUrl())
      return;
    const precondition = { session_id: sessionId, expected_revision: revision };
    const { result: response, completedJobId } = await this.jobs.run<TrackPromptPointsJobResponse>(
      'Tracking prompt points',
      () =>
        firstValueFrom(
          this.backend.trackPromptPoints(
            { add_support_grid: this.store.trackingUseSupportGrid() },
            precondition,
          ),
        ),
    );
    if (
      this.destroyRef.destroyed ||
      backendUrl !== this.backend.getApiUrl() ||
      sessionId !== this.store.sessionId()
    )
      return;
    if (!response) {
      this.store.annotationRecoveryRequested.set(true);
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
    const revision = this.store.annotationRevision();
    const sessionId = this.store.sessionId();
    const backendUrl = this.backend.getApiUrl();
    const epoch = this.store.stateEpoch();
    try {
      const response = await firstValueFrom(this.backend.getTrackingResult(resultId));
      const result: TrackPromptPointsResult = response.result;
      if (
        this.destroyRef.destroyed ||
        revision !== this.store.annotationRevision() ||
        epoch !== this.store.stateEpoch() ||
        sessionId !== this.store.sessionId() ||
        backendUrl !== this.backend.getApiUrl() ||
        result.session_id !== sessionId ||
        result.annotation_revision !== revision ||
        result.state_epoch !== epoch ||
        result.result_id !== resultId
      )
        return false;
      const trackedSeries: TrackedPointSeries[] = result.points.map((point, index) => ({
        ...point,
        tracks: result.tracks[index] || [],
        visibility: result.visibility[index] || [],
      }));
      this.store.trackingResultId = resultId;
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
