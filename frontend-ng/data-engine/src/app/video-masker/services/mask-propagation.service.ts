import { Injectable, inject } from '@angular/core';
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
    const previousMasks = this.store.masks();
    const previousLiveEdits = this.store.liveEditedObjectFrames();
    this.store.updateStateEpoch(response.state_epoch, 'propagation');
    // The restored predictor has a new epoch, but propagation only replaces this range.
    this.store.masks.set(
      new Map([...previousMasks].filter(([frame]) => frame < startFrameIdx || frame > endFrameIdx)),
    );
    this.store.liveEditedObjectFrames.set(
      new Map(
        [...previousLiveEdits].filter(([frame]) => frame < startFrameIdx || frame > endFrameIdx),
      ),
    );
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
    this.store.isLoading.set(true);
    this.backend
      .resetVideoState()
      .pipe(finalize(() => this.store.isLoading.set(false)))
      .subscribe({
        next: (response) => {
          this.store.updateStateEpoch(response?.state_epoch, 'reset');
          this.store.hasManifestMasks.set(false);
          this.store.invalidateTracking();
          this.store.resetInteractiveMaps();
          this.framePipeline.clearMaskDataCache();
          this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
        },
        error: (error) => {
          console.error(error);
          this.toast.show(
            'error',
            'Clear failed',
            getErrorMessage(error, 'Failed to clear masks.'),
          );
        },
      });
  }
}
