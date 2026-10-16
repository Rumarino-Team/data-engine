import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import {
  BackendService,
  RestoredSessionPayload,
  VideoInitStateResponse,
  VideoSaveInteractiveState,
  VideoSaveResponse,
} from '../../services/backend.service';
import { VideoMaskerStateStore } from './video-masker-state.store';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import { EditorJobService } from './editor-job.service';
import { TrackingService } from './tracking.service';
import { ObjectOperationsService } from './object-operations.service';
import {
  buildInteractiveStateSnapshot,
  deserializeLiveMasks,
  deserializePoints,
  getErrorMessage,
} from '../video-masker.util';

/** Loads, restores, and saves annotation sessions. */
@Injectable()
export class SessionService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly framePipeline = inject(FramePipelineService);
  private readonly toast = inject(ToastService);
  private readonly jobs = inject(EditorJobService);
  private readonly tracking = inject(TrackingService);
  private readonly objectOperations = inject(ObjectOperationsService);
  private get isBusy(): boolean {
    return this.store.isInteractionBusy();
  }

  async initVideo(): Promise<boolean> {
    if (this.isBusy) return false;
    const enteredPath = this.store
      .videoDir()
      .trim()
      .replace(/^['"]|['"]$/g, '');
    if (!enteredPath) {
      if (this.store.loadSourceMode() === 'saved_session_dir') {
        this.toast.show(
          'warning',
          'Missing session path',
          'Enter a saved session directory path or browse for one.',
        );
        return false;
      }
      if (this.store.loadSourceMode() === 'video_file') {
        this.toast.show(
          'warning',
          'Missing video path',
          'Enter a video file path or browse for one.',
        );
        return false;
      }
      this.toast.show(
        'warning',
        'Missing frames path',
        'Enter a frames directory path or browse for one.',
      );
      return false;
    }

    this.store.videoDir.set(enteredPath);
    const { result: res } = await this.jobs.run<VideoInitStateResponse>('Loading video', () =>
      firstValueFrom(this.backend.initVideoState(enteredPath)),
    );
    if (!res) {
      return false;
    }

    this.store.numFrames.set(res.num_frames);
    this.store.targetFrameIdx.set(0);
    this.store.displayedFrameIdx.set(-1);
    this.store.propagationStartFrameIdx.set(0);
    this.store.propagationEndFrameIdx.set(Math.max(0, res.num_frames - 1));
    this.store.saveName.set('');
    this.store.invalidateTracking();
    this.store.nextObjectId = res.restored_session?.annotations.next_object_id ?? 1;
    this.store.resetInteractiveMaps();
    this.framePipeline.clearFrameCaches();
    this.store.objects.set(
      Object.values(res.restored_session?.annotations.objects ?? {}).sort((a, b) => a.id - b.id),
    );
    this.store.selectedObjectId.set(null);
    this.store.hasManifestMasks.set(Boolean(res.restored_session?.has_mask_manifest));
    this.store.updateStateEpoch(res.state_epoch, 'video init');
    this.restoreInteractiveSessionState(res.restored_session);
    if (res.restored_session) {
      // Canonical sessions may intentionally have no objects after deletion.
      if (this.store.selectedObjectId() === null) {
        this.store.selectedObjectId.set(this.store.objects()[0]?.id ?? null);
      }
    } else {
      this.objectOperations.addObject();
    }

    const restoredTrackingResultId = res.restored_session?.tracking_result?.result_id;
    if (restoredTrackingResultId) {
      await this.tracking.loadTrackingResult(restoredTrackingResultId, 'Restored tracking');
    }

    this.store.resetDebugState();
    this.store.isInitialized.set(true);
    return true;
  }

  private restoreInteractiveSessionState(
    restored: RestoredSessionPayload | null | undefined,
  ): void {
    if (!restored?.interactive_state) {
      return;
    }
    const interactive = restored.interactive_state;
    const restoredPoints = deserializePoints(interactive.points || []);
    if (restoredPoints.size > 0) {
      this.store.points.set(restoredPoints);
    }

    const { masks, liveEditedFrames } = deserializeLiveMasks(interactive.live_masks || []);
    if (masks.size > 0) {
      this.store.masks.set(masks);
      this.store.liveEditedObjectFrames.set(liveEditedFrames);
    }
    this.store.selectedObjectId.set(interactive.selected_object_id ?? null);

    if (
      interactive.interaction_mode === 'positive' ||
      interactive.interaction_mode === 'negative'
    ) {
      this.store.interactionMode.set(interactive.interaction_mode);
    }
    if (
      typeof interactive.current_frame_idx === 'number' &&
      Number.isFinite(interactive.current_frame_idx) &&
      interactive.current_frame_idx >= 0 &&
      interactive.current_frame_idx < this.store.numFrames()
    ) {
      this.store.targetFrameIdx.set(Math.trunc(interactive.current_frame_idx));
    }
  }

  private buildSnapshot(): VideoSaveInteractiveState {
    return buildInteractiveStateSnapshot({
      selectedObjectId: this.store.selectedObjectId(),
      interactionMode: this.store.interactionMode(),
      currentFrameIdx: this.store.targetFrameIdx(),
      pointsByFrame: this.store.points(),
      masksByFrame: this.store.masks(),
    });
  }

  save(): void {
    if (this.isBusy) return;
    const name = this.store.saveName().trim();
    if (!name) {
      this.toast.show('warning', 'Missing save name', 'Enter a name for this saved session.');
      return;
    }
    const interactiveState = this.buildSnapshot();
    this.store.isLoading.set(true);
    firstValueFrom(this.backend.saveVideoSession(name, interactiveState))
      .then((response: VideoSaveResponse) => {
        this.store.updateStateEpoch(response.state_epoch, 'save');
        this.store.saveName.set(response.name);
        this.toast.show('success', 'Session saved', `Saved to ${response.saved_path}`);
      })
      .catch((error: unknown) => {
        console.error(error);
        this.toast.show('error', 'Save failed', getErrorMessage(error, 'Session save failed'));
      })
      .finally(() => {
        this.store.isLoading.set(false);
      });
  }
}
