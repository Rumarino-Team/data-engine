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
import { VideoMaskerStateStore } from './video-masker-state.store';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import { EditorJobService } from './editor-job.service';
import { TrackingService } from './tracking.service';
import { getErrorMessage } from '../video-masker.util';

/** Loads, restores, and saves annotation sessions. */
@Injectable()
export class SessionService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly destroyRef = inject(DestroyRef);
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
      const busy = this.isBusy;
      const mutationRecovery = this.store.annotationRecoveryRequested();
      if (mutationRecovery && initialized && !busy) {
        this.store.annotationRecoveryRequested.set(false);
        untracked(() => void this.refreshAnnotations());
        return;
      }
      if (status === 'offline' && initialized) recoveryPending = true;
      if (status === 'online' && recoveryPending && initialized && !busy) {
        recoveryPending = false;
        untracked(() => void this.refreshAnnotations());
      }
    });
  }

  /**
   * Recover accepted inputs from the backend without reopening the video or resetting the viewport.
   * Edits are refused while the snapshot is fetched, so the response always applies cleanly.
   */
  async refreshAnnotations(): Promise<boolean> {
    if (!this.store.isInitialized() || this.isBusy) return false;
    const sessionId = this.store.sessionId();
    this.store.isRecoveringAnnotations.set(true);
    try {
      const info = await firstValueFrom(
        this.backend.getVideoInfo().pipe(takeUntilDestroyed(this.destroyRef)),
      );
      if (info.session_id !== sessionId || !info.annotations) {
        this.store.isInitialized.set(false);
        throw new Error('The backend session changed. Load the video again to continue.');
      }
      if (!this.store.applyAnnotationSnapshot(info)) return false;
      this.store.numFrames.set(info.num_frames);
      if (info.predictor_ready === false) {
        this.store.isInitialized.set(false);
        throw new Error('The predictor is unavailable. Reload the video to continue.');
      }
      this.framePipeline.clearMaskDataCache();
      this.framePipeline.redraw();
      this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
      return true;
    } catch (error) {
      if (!this.destroyRef.destroyed) {
        this.toast.show(
          'warning',
          'Annotation refresh failed',
          getErrorMessage(error, 'Unable to refresh annotations.'),
        );
      }
      return false;
    } finally {
      this.store.isRecoveringAnnotations.set(false);
    }
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

    this.store.videoDir.set(enteredPath);
    const { result: res } = await this.jobs.run<VideoInitStateResponse>('Loading video', () =>
      firstValueFrom(this.backend.initVideoState(enteredPath)),
    );
    if (this.destroyRef.destroyed) return false;
    if (!res) {
      // A failed replacement preserves the old session. Recover uncertain/lost responses.
      if (this.store.isInitialized()) this.store.annotationRecoveryRequested.set(true);
      return false;
    }

    this.store.isInitialized.set(false);
    this.store.isLoading.set(true);
    try {
      const info = await firstValueFrom(
        this.backend.getVideoInfo().pipe(takeUntilDestroyed(this.destroyRef)),
      );
      if (this.destroyRef.destroyed) return false;
      if (!info.session_id || !info.annotations) {
        throw new Error('The initialized session is no longer current. Load the video again.');
      }
      this.store.targetFrameIdx.set(0);
      this.store.displayedFrameIdx.set(-1);
      this.store.propagationStartFrameIdx.set(0);
      this.store.propagationEndFrameIdx.set(Math.max(0, info.num_frames - 1));
      this.store.saveName.set('');
      this.store.invalidateTracking();
      this.framePipeline.clearFrameCaches();
      this.store.selectedObjectId.set(null);
      this.store.applyAnnotationSnapshot(info, true);
      this.store.saveName.set(info.saved_state?.display_name ?? '');
      this.store.numFrames.set(info.num_frames);
      this.restoreInteractiveSessionState(res.restored_session);
      if (!this.store.objects().some((object) => object.id === this.store.selectedObjectId())) {
        this.store.selectedObjectId.set(this.store.objects()[0]?.id ?? null);
      }
      const resultId = res.restored_session?.tracking_result?.result_id;
      if (resultId) await this.tracking.loadTrackingResult(resultId, 'Restored tracking');
      if (this.destroyRef.destroyed) return false;
      this.store.isInitialized.set(true);
    } catch (error) {
      if (!this.destroyRef.destroyed) {
        this.toast.show(
          'error',
          'Loading video',
          getErrorMessage(error, 'Unable to load accepted annotations.'),
        );
      }
      return false;
    } finally {
      if (!this.destroyRef.destroyed) this.store.isLoading.set(false);
    }
    return true;
  }

  private restoreInteractiveSessionState(
    restored: RestoredSessionPayload | null | undefined,
  ): void {
    if (!restored?.interactive_state) {
      return;
    }
    const interactive = restored.interactive_state;
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
    return {
      version: 1,
      selected_object_id: this.store.selectedObjectId(),
      interaction_mode: this.store.interactionMode(),
      current_frame_idx: this.store.targetFrameIdx(),
    };
  }

  /**
   * Publishes the working state. Without `saveAs`, a session with a checkpoint re-saves it (the
   * typed name only renames it); a never-saved session, or `saveAs`, needs a name and creates a
   * new checkpoint. Resolves true once the backend accepted the save.
   */
  async save(saveAs = false): Promise<boolean> {
    if (this.isBusy || !this.store.isInitialized()) return false;
    const saved = this.store.savedState();
    const name = this.store.saveName().trim();
    const needsName = saveAs || !saved?.checkpoint_id;
    if (needsName && !name) {
      this.toast.show('warning', 'Missing save name', 'Enter a name for this saved session.');
      return false;
    }
    const sendName = needsName || (name !== '' && name !== saved?.display_name);
    const sessionId = this.store.sessionId();
    const revision = this.store.annotationRevision();
    if (!sessionId || revision === null) return false;
    this.store.isLoading.set(true);
    try {
      const response: VideoSaveResponse = await firstValueFrom(
        this.backend.saveVideoSession(
          { name: sendName ? name : undefined, saveAs },
          { session_id: sessionId, expected_revision: revision },
          this.buildSnapshot(),
        ),
      );
      if (this.destroyRef.destroyed) return true;
      this.store.savedState.set(response.saved_state);
      this.store.saveName.set(response.name);
      this.toast.show('success', 'Session saved', `Saved to ${response.saved_path}`);
      return true;
    } catch (error: unknown) {
      console.error(error);
      // The session or its annotations changed elsewhere since this tab last saw them: nothing was saved,
      // and the tab reloads what the backend holds before the user saves again.
      if ((error as { status?: number })?.status === 409) this.store.annotationRecoveryRequested.set(true);
      this.toast.show('error', 'Save failed', getErrorMessage(error, 'Session save failed'));
      return false;
    } finally {
      this.store.isLoading.set(false);
    }
  }
}
