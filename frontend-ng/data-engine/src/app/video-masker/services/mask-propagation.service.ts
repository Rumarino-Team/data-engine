import { DestroyRef, Injectable, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { firstValueFrom, finalize } from 'rxjs';
import { BackendService, VideoPropagateResponse } from '../../services/backend.service';
import { VideoMaskerStateStore } from './video-masker-state.store';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import { EditorJobService } from './editor-job.service';
import { getErrorMessage } from '../video-masker.util';

/** Runs propagation and clears generated masks while preserving unaffected edits. */
@Injectable()
export class MaskPropagationService {
  private readonly destroyRef = inject(DestroyRef);
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly framePipeline = inject(FramePipelineService);
  private readonly toast = inject(ToastService);
  private readonly jobs = inject(EditorJobService);
  private get isBusy(): boolean {
    return this.store.isInteractionBusy();
  }

  async propagate(): Promise<void> {
    if (this.isBusy) return;
    const startFrameIdx = this.store.propagationStartFrameIdx();
    const endFrameIdx = this.store.propagationEndFrameIdx();
    if (startFrameIdx > endFrameIdx) {
      this.toast.show(
        'warning',
        'Invalid frame range',
        'The start frame must not be after the end frame.',
      );
      return;
    }
    const { result: response } = await this.jobs.run<VideoPropagateResponse>(
      'Propagating masks',
      () =>
        firstValueFrom(
          this.backend.propagateInVideo({
            start_frame_idx: startFrameIdx,
            max_frame_num_to_track: endFrameIdx - startFrameIdx + 1,
            include_masks_in_response: false,
            include_saved_mask_paths: false,
          }),
        ),
    );
    if (!response) {
      return;
    }
    this.store.updateStateEpoch(response.state_epoch, 'propagation');
    const maskManifestPath = response.mask_manifest_path || response['state.mask_manifest_path'];
    this.store.hasManifestMasks.set(Boolean(maskManifestPath));
    if (response.tracked_points_skipped_reason) {
      this.toast.show(
        'warning',
        'Tracking guidance skipped',
        response.tracked_points_skipped_reason,
      );
    }
    this.framePipeline.clearMaskDataCache();
    this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
  }

  clearMasks(): void {
    if (this.isBusy) return;
    const backendUrl = this.backend.getApiUrl();
    const sessionId = this.store.sessionId();
    const revision = this.store.annotationRevision();
    if (!sessionId || revision === null || backendUrl !== this.store.annotationsBackendUrl())
      return;
    const ownsContext = () =>
      !this.destroyRef.destroyed &&
      backendUrl === this.backend.getApiUrl() &&
      sessionId === this.store.sessionId() &&
      backendUrl === this.store.annotationsBackendUrl();
    this.store.isLoading.set(true);
    this.backend
      .resetVideoState({ session_id: sessionId, expected_revision: revision })
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        finalize(() => this.store.isLoading.set(false)),
      )
      .subscribe({
        next: (response) => {
          if (!ownsContext()) return;
          if (!this.store.applyAnnotationUpdate(response, backendUrl)) {
            this.store.annotationRecoveryRequested.set(true);
            return;
          }
          this.store.hasManifestMasks.set(false);
          this.store.invalidateTracking();
          this.store.resetInteractiveMaps();
          this.framePipeline.clearMaskDataCache();
          this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
        },
        error: (error) => {
          if (!ownsContext()) return;
          const status = (error as { status?: number })?.status;
          if (status === undefined || status === 0 || status === 409 || status >= 500)
            this.store.annotationRecoveryRequested.set(true);
          this.toast.show(
            'error',
            'Clear failed',
            getErrorMessage(error, 'Failed to clear masks.'),
          );
        },
      });
  }
}
