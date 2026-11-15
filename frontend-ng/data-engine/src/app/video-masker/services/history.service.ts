import { DestroyRef, Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { BackendService } from '../../services/backend.service';
import { getErrorMessage } from '../video-masker.util';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import { TrackingService } from './tracking.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

type Direction = 'undo' | 'redo';

/** Undo and redo of accepted changes. The backend holds the history; this applies what it answers. */
@Injectable()
export class HistoryService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly framePipeline = inject(FramePipelineService);
  private readonly tracking = inject(TrackingService);
  private readonly toast = inject(ToastService);
  private readonly destroyRef = inject(DestroyRef);

  canUndo(): boolean {
    return this.available() && this.store.history().undo > 0;
  }

  canRedo(): boolean {
    return this.available() && this.store.history().redo > 0;
  }

  undo(): Promise<boolean> {
    return this.step('undo');
  }

  redo(): Promise<boolean> {
    return this.step('redo');
  }

  private available(): boolean {
    return this.store.isInitialized() && !this.store.isInteractionBusy();
  }

  private async step(direction: Direction): Promise<boolean> {
    if (!(direction === 'undo' ? this.canUndo() : this.canRedo())) return false;
    const sessionId = this.store.sessionId();
    const revision = this.store.annotationRevision();
    if (!sessionId || revision === null) return false;
    const precondition = { session_id: sessionId, expected_revision: revision };
    const title = direction === 'undo' ? 'Undo' : 'Redo';
    this.store.isLoading.set(true);
    try {
      const response = await firstValueFrom(
        direction === 'undo'
          ? this.backend.undoVideoChange(precondition)
          : this.backend.redoVideoChange(precondition),
      );
      if (this.destroyRef.destroyed) return false;
      if (!this.store.applyAnnotationUpdate(response)) {
        this.store.annotationRecoveryRequested.set(true);
        return false;
      }
      // The restored points and results can differ on any frame, so nothing cached for them is trusted.
      this.framePipeline.clearMaskDataCache();
      const resultId = response.tracking_validity?.result_id;
      if (resultId && resultId !== this.store.trackingResultId) {
        await this.tracking.loadTrackingResult(resultId, 'Restored tracking');
      }
      this.framePipeline.redraw();
      this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
      this.toast.show(
        'info',
        direction === 'undo' ? 'Undone' : 'Redone',
        response.message.replace(/^(Undo|Redo): /, ''),
      );
      return true;
    } catch (error: unknown) {
      console.error(error);
      const status = (error as { status?: number })?.status;
      // The session changed elsewhere, or the backend could not answer: reload what it holds.
      if (status === undefined || status === 0 || status === 409 || status >= 500) {
        this.store.annotationRecoveryRequested.set(true);
      }
      this.toast.show('error', `${title} failed`, getErrorMessage(error, `${title} failed.`));
      return false;
    } finally {
      if (!this.destroyRef.destroyed) this.store.isLoading.set(false);
    }
  }
}
