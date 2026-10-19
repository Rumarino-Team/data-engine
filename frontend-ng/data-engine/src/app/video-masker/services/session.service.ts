import { DestroyRef, Injectable, effect, inject, untracked } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  BackendService,
  RestoredSessionPayload,
  VideoInitStateResponse,
  VideoSaveInteractiveState,
  VideoSaveResponse,
} from '../../services/backend.service';
import { Point, VideoMaskerStateStore } from './video-masker-state.store';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import { EditorJobService } from './editor-job.service';
import { TrackingService } from './tracking.service';
import {
  buildInteractiveStateSnapshot,
  deserializeLiveMasks,
  getErrorMessage,
} from '../video-masker.util';

/** Loads, restores, and saves annotation sessions. */
@Injectable()
export class SessionService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly destroyRef = inject(DestroyRef);
  private snapshotRequest = 0;
  private readonly backend = inject(BackendService);
  private readonly framePipeline = inject(FramePipelineService);
  private readonly toast = inject(ToastService);
  private readonly jobs = inject(EditorJobService);
  private readonly tracking = inject(TrackingService);
  private get isBusy(): boolean {
    return this.store.isInteractionBusy();
  }

  constructor() {
    let recoveryPending = false;
    effect(() => {
      const status = this.store.apiHealthStatus();
      const initialized = this.store.isInitialized();
      const busy = this.isBusy || this.store.pendingObjectUpdates() > 0;
      if (status === 'offline' && initialized) recoveryPending = true;
      if (status === 'online' && recoveryPending && initialized && !busy) {
        recoveryPending = false;
        untracked(() => void this.refreshAnnotations());
      }
    });
  }

  /** Recover accepted inputs without reopening the video or resetting the viewport. */
  async refreshAnnotations(): Promise<boolean> {
    if (!this.store.isInitialized() || this.isBusy || this.store.pendingObjectUpdates() > 0)
      return false;
    const backendUrl = this.backend.getApiUrl();
    if (backendUrl !== this.store.annotationsBackendUrl()) return false;
    const sessionId = this.store.sessionId();
    const epoch = this.store.stateEpoch();
    const objects = this.store.objects();
    const points = this.store.points();
    const boxes = this.store.annotations()?.prompts.filter((prompt) => prompt.box);
    const request = ++this.snapshotRequest;
    try {
      const info = await firstValueFrom(
        this.backend.getVideoInfo().pipe(takeUntilDestroyed(this.destroyRef)),
      );
      // Existing workflows still own optimistic maps until their migration. A snapshot
      // requested before an edit must not erase that edit, even if it has since finished.
      if (
        this.destroyRef.destroyed ||
        request !== this.snapshotRequest ||
        backendUrl !== this.backend.getApiUrl() ||
        sessionId !== this.store.sessionId() ||
        epoch !== this.store.stateEpoch() ||
        objects !== this.store.objects() ||
        points !== this.store.points() ||
        this.isBusy ||
        this.store.pendingObjectUpdates() > 0
      )
        return false;
      if (info.session_id !== sessionId || !info.annotations) {
        this.store.isInitialized.set(false);
        throw new Error('The backend session changed. Load the video again to continue.');
      }
      if (!this.store.applyAnnotationSnapshot(info, backendUrl)) return false;
      this.store.numFrames.set(info.num_frames);
      if (
        this.pointSignature(points) !== this.pointSignature(this.store.points()) ||
        JSON.stringify(boxes) !==
          JSON.stringify(this.store.annotations()?.prompts.filter((prompt) => prompt.box))
      ) {
        this.store.masks.set(new Map());
        this.store.liveEditedObjectFrames.set(new Map());
        this.store.invalidateTracking();
      }
      this.framePipeline.clearMaskDataCache();
      this.framePipeline.redraw();
      this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
      return true;
    } catch (error) {
      if (
        !this.destroyRef.destroyed &&
        request === this.snapshotRequest &&
        backendUrl === this.backend.getApiUrl()
      ) {
        this.toast.show(
          'warning',
          'Annotation refresh failed',
          getErrorMessage(error, 'Unable to refresh annotations.'),
        );
      }
      return false;
    }
  }

  private pointSignature(points: Map<number, Map<number, Point[]>>): string {
    return JSON.stringify(
      [...points]
        .sort(([a], [b]) => a - b)
        .map(([frame, objects]) => [frame, [...objects].sort(([a], [b]) => a - b)]),
    );
  }

  async initVideo(): Promise<boolean> {
    if (this.isBusy || this.store.pendingObjectUpdates() > 0) return false;
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

    const backendUrl = this.backend.getApiUrl();
    const request = ++this.snapshotRequest;
    this.store.isInitialized.set(false);
    this.store.videoDir.set(enteredPath);
    const { result: res } = await this.jobs.run<VideoInitStateResponse>('Loading video', () =>
      firstValueFrom(this.backend.initVideoState(enteredPath)),
    );
    if (!res) {
      return false;
    }

    this.store.isLoading.set(true);
    try {
      const info = await firstValueFrom(
        this.backend.getVideoInfo().pipe(takeUntilDestroyed(this.destroyRef)),
      );
      if (
        this.destroyRef.destroyed ||
        request !== this.snapshotRequest ||
        backendUrl !== this.backend.getApiUrl()
      )
        return false;
      if (!info.session_id || !info.annotations || info.state_epoch !== res.state_epoch) {
        throw new Error('The initialized session is no longer current. Load the video again.');
      }
      this.store.targetFrameIdx.set(0);
      this.store.displayedFrameIdx.set(-1);
      this.store.propagationStartFrameIdx.set(0);
      this.store.propagationEndFrameIdx.set(Math.max(0, info.num_frames - 1));
      this.store.saveName.set('');
      this.store.invalidateTracking();
      this.store.resetInteractiveMaps();
      this.framePipeline.clearFrameCaches();
      this.store.selectedObjectId.set(null);
      this.store.applyAnnotationSnapshot(info, backendUrl, true);
      this.store.numFrames.set(info.num_frames);
      this.store.hasManifestMasks.set(Boolean(res.restored_session?.has_mask_manifest));
      this.restoreInteractiveSessionState(res.restored_session, info.annotations.revision);
      if (!this.store.objects().some((object) => object.id === this.store.selectedObjectId())) {
        this.store.selectedObjectId.set(this.store.objects()[0]?.id ?? null);
      }
      const resultId = res.restored_session?.tracking_result?.result_id;
      if (resultId) await this.tracking.loadTrackingResult(resultId, 'Restored tracking');
      if (
        this.destroyRef.destroyed ||
        request !== this.snapshotRequest ||
        backendUrl !== this.backend.getApiUrl()
      )
        return false;
      this.store.resetDebugState();
      this.store.isInitialized.set(true);
    } catch (error) {
      if (
        !this.destroyRef.destroyed &&
        request === this.snapshotRequest &&
        backendUrl === this.backend.getApiUrl()
      ) {
        this.toast.show(
          'error',
          'Loading video',
          getErrorMessage(error, 'Unable to load accepted annotations.'),
        );
      }
      return false;
    } finally {
      if (request === this.snapshotRequest && !this.destroyRef.destroyed)
        this.store.isLoading.set(false);
    }
    return true;
  }

  private restoreInteractiveSessionState(
    restored: RestoredSessionPayload | null | undefined,
    revision: number,
  ): void {
    if (!restored?.interactive_state) {
      return;
    }
    const interactive = restored.interactive_state;
    const { masks, liveEditedFrames } = deserializeLiveMasks(
      restored.annotations.revision === revision ? interactive.live_masks : [],
    );
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
