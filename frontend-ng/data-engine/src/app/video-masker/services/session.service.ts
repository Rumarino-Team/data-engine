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
import {
  buildInteractiveStateSnapshot,
  deserializeLiveMasks,
  deserializePoints,
  getErrorMessage,
  normalizeInteractiveObjects,
  randomColor,
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
    this.store.nextObjectId = 1;
    this.store.resetInteractiveMaps();
    this.framePipeline.clearFrameCaches();
    this.store.objects.set([{ id: 1, name: 'Object 1', color: randomColor() }]);
    this.store.selectedObjectId.set(1);
    this.store.hasManifestMasks.set(Boolean(res.restored_session?.has_mask_manifest));
    this.store.updateStateEpoch(res.state_epoch, 'video init');
    this.restoreInteractiveSessionState(res.restored_session);

    const restoredTrackingResultId = res.restored_session?.tracking_result?.result_id;
    if (restoredTrackingResultId) {
      await this.tracking.loadTrackingResult(restoredTrackingResultId, 'Restored tracking');
    }

    const restoredWarnings = res.restored_session?.interactive_state_warnings || [];
    if (restoredWarnings.length > 0) {
      this.toast.show(
        'warning',
        'Session partially restored',
        restoredWarnings.slice(0, 2).join(' '),
      );
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
    const restoredObjects = normalizeInteractiveObjects(interactive.objects || []);
    if (restoredObjects.length > 0) {
      this.store.objects.set(restoredObjects);
    }

    const restoredPoints = deserializePoints(interactive.points || []);
    if (restoredPoints.size > 0) {
      this.store.points.set(restoredPoints);
    }

    const { masks, liveEditedFrames } = deserializeLiveMasks(interactive.live_masks || []);
    if (masks.size > 0) {
      this.store.masks.set(masks);
      this.store.liveEditedObjectFrames.set(liveEditedFrames);
    }
    this.ensureObjectsForRestoredState();

    const availableObjectIds = new Set(this.store.objects().map((objectEntry) => objectEntry.id));
    const requestedObjectId = interactive.selected_object_id ?? null;
    if (requestedObjectId !== null && availableObjectIds.has(requestedObjectId)) {
      this.store.selectedObjectId.set(requestedObjectId);
    } else if (this.store.objects().length > 0) {
      this.store.selectedObjectId.set(this.store.objects()[0].id);
    }

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

  private ensureObjectsForRestoredState(): void {
    const existing = new Map(this.store.objects().map((entry) => [entry.id, entry]));
    const requiredObjectIds = new Set<number>();
    this.store.points().forEach((framePoints) => {
      framePoints.forEach((_objPoints, objId) => requiredObjectIds.add(objId));
    });
    this.store.masks().forEach((frameMasks) => {
      frameMasks.forEach((_mask, objId) => requiredObjectIds.add(objId));
    });
    let changed = false;
    for (const objId of requiredObjectIds) {
      if (existing.has(objId)) {
        continue;
      }
      existing.set(objId, { id: objId, name: `Object ${objId}`, color: randomColor() });
      changed = true;
    }
    if (changed) {
      this.store.objects.set(Array.from(existing.values()).sort((a, b) => a.id - b.id));
    }
  }

  private buildSnapshot(): VideoSaveInteractiveState {
    return buildInteractiveStateSnapshot({
      objects: this.store.objects(),
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
