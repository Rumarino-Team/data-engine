import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import {
  BackendService,
  RestoredSessionPayload,
  TrackPromptPointsJobResponse,
  TrackPromptPointsResult,
  VideoAddPointsOrBoxRequest,
  VideoInitStateResponse,
  VideoPropagateResponse,
  VideoSaveInteractiveState,
  VideoSaveResponse,
} from '../../services/backend.service';
import { DesktopBridgeService } from '../../services/desktop-bridge.service';
import { LoadSourceMode, ToastSeverity } from '../state/video-masker-ui.types';
import { Point, TrackedPointSeries, VideoMaskerStateStore } from './video-masker-state.store';
import { FrameCanvasService } from './frame-canvas.service';
import { ToastService } from './toast.service';
import { VideoJobsService } from './video-jobs.service';
import {
  buildInteractiveStateSnapshot,
  deserializeLiveMasks,
  deserializePoints,
  getErrorMessage,
  getMaskPixelCount,
  isObjectLiveEdited,
  markObjectLiveEdited,
  normalizeInteractiveObjects,
  randomColor,
  resolveStateEpoch,
  unmarkObjectLiveEdited,
} from '../video-masker.util';

/**
 * All backend-driven workflows for the video masker: session load/save, object
 * management, point prompting (with the epoch/frame discard rules), mask propagation and
 * prompt-point tracking, plus API-URL and health handling. Mutates
 * {@link VideoMaskerStateStore} directly and delegates painting to {@link FrameCanvasService}.
 */
@Injectable()
export class VideoMaskerActionsService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly desktopBridge = inject(DesktopBridgeService);
  private readonly jobs = inject(VideoJobsService);
  private readonly toastSvc = inject(ToastService);
  private readonly frameCanvas = inject(FrameCanvasService);

  private lastCompletedJobId: string | null = null;

  // --- API URL + health ------------------------------------------------------

  initApiUrlFromBackend(): void {
    this.store.apiUrlInput.set(this.backend.getApiUrl());
  }

  async checkApiHealth(showChecking = false): Promise<void> {
    if (showChecking || this.store.apiHealthStatus() === 'checking') {
      this.store.apiHealthStatus.set('checking');
    }
    try {
      await firstValueFrom(this.backend.health());
      this.store.apiHealthStatus.set('online');
    } catch {
      this.store.apiHealthStatus.set('offline');
    }
  }

  applyApiUrl(): void {
    this.store.apiUrlInput.set(this.backend.setApiUrl(this.store.apiUrlInput()));
    void this.checkApiHealth(true);
  }

  resetApiUrl(): void {
    this.store.apiUrlInput.set(this.backend.resetApiUrl());
    void this.checkApiHealth(true);
  }

  isApiUrlDirty(): boolean {
    return this.store.apiUrlInput().trim() !== this.backend.getApiUrl();
  }

  // --- toasts --------------------------------------------------------------

  dismissToast(id: number): void {
    this.toastSvc.dismiss(id);
    this.store.toasts.set(this.toastSvc.toasts());
  }

  /** Public entry point for component-level toasts (e.g. browser file-path fallbacks). */
  pushToast(severity: ToastSeverity, title: string, message: string): void {
    this.notify(severity, title, message);
  }

  private notify(severity: ToastSeverity, title: string, message: string): void {
    this.toastSvc.show(severity, title, message);
    this.store.toasts.set(this.toastSvc.toasts());
  }

  // --- source picking -----------------------------------------------------

  async pickNativePath(mode: LoadSourceMode): Promise<string | null> {
    if (!this.desktopBridge.isTauri()) {
      return null;
    }
    if (mode === 'video_file') {
      return this.desktopBridge.pickVideoFile();
    }
    return this.desktopBridge.pickFramesDirectory();
  }

  // --- job + epoch plumbing ---------------------------------------------------

  private async runJob<T>(
    title: string,
    startJob: () => Promise<{ job_id: string }>,
  ): Promise<T | null> {
    const { result, completedJobId } = await this.jobs.run<T>({
      title,
      startJob,
      onStart: () => {
        this.store.isLoading.set(true);
        this.store.activeJobTitle.set(title);
        this.store.activeJob.set(null);
        this.lastCompletedJobId = null;
      },
      onStatus: (job) => this.store.activeJob.set(job),
      onFailure: (message) => this.notify('error', title, message),
      onFinish: () => {
        this.store.activeJob.set(null);
        this.store.activeJobTitle.set('');
        this.store.isLoading.set(false);
      },
      fallbackErrorMessage: `${title} failed`,
    });
    this.lastCompletedJobId = completedJobId;
    return result;
  }

  private updateStateEpoch(nextEpoch: number | undefined, source: string): void {
    const resolution = resolveStateEpoch(this.store.stateEpoch(), nextEpoch);
    if (!resolution) {
      return;
    }
    if (resolution.shouldClearLiveState) {
      this.store.masks.set(new Map());
      this.store.liveEditedObjectFrames.set(new Map());
      this.store.lastDiscardReason.set(
        `State epoch changed (${this.store.stateEpoch()} -> ${resolution.normalizedEpoch}) during ${source}; cleared live masks.`,
      );
    }
    this.store.stateEpoch.set(resolution.normalizedEpoch);
  }

  private markObjectAsLiveEdited(frameIdx: number, objId: number): void {
    this.store.liveEditedObjectFrames.set(
      markObjectLiveEdited(this.store.liveEditedObjectFrames(), frameIdx, objId),
    );
  }

  private unmarkObjectAsLiveEdited(frameIdx: number, objId: number): void {
    this.store.liveEditedObjectFrames.set(
      unmarkObjectLiveEdited(this.store.liveEditedObjectFrames(), frameIdx, objId),
    );
  }

  private isObjectLiveEdited(frameIdx: number, objId: number): boolean {
    return isObjectLiveEdited(this.store.liveEditedObjectFrames(), frameIdx, objId);
  }

  private resetDebugState(): void {
    this.store.lastClickRequestFrameIdx.set(null);
    this.store.lastBackendResponseFrameIdx.set(null);
    this.store.lastBackendResponseFrameFile.set('n/a');
    this.store.lastBackendResponseStateEpoch.set(null);
    this.store.lastDebugObjectId.set(null);
    this.store.lastMaskPixelCount.set(null);
    this.store.lastFallbackUsed.set(false);
    this.store.lastMaskSource.set('none');
    this.store.lastDiscardReason.set(null);
  }

  private resetInteractiveMaps(): void {
    this.store.masks.set(new Map());
    this.store.points.set(new Map());
    this.store.liveEditedObjectFrames.set(new Map());
  }

  private showError(title: string, fallbackMessage: string, error: unknown): void {
    console.error(error);
    this.notify('error', title, getErrorMessage(error, fallbackMessage));
  }

  // --- session load / save -------------------------------------------------

  async initVideo(): Promise<boolean> {
    const enteredPath = this.store
      .videoDir()
      .trim()
      .replace(/^['"]|['"]$/g, '');
    if (!enteredPath) {
      if (this.store.loadSourceMode() === 'saved_session_dir') {
        this.notify('warning', 'Missing session path', 'Enter a saved session directory path or browse for one.');
        return false;
      }
      if (this.store.loadSourceMode() === 'video_file') {
        this.notify('warning', 'Missing video path', 'Enter a video file path or browse for one.');
        return false;
      }
      this.notify('warning', 'Missing frames path', 'Enter a frames directory path or browse for one.');
      return false;
    }

    this.store.videoDir.set(enteredPath);
    const res = await this.runJob<VideoInitStateResponse>('Loading video', () =>
      firstValueFrom(this.backend.initVideoState(enteredPath)),
    );
    if (!res) {
      return false;
    }

    this.store.numFrames.set(res.num_frames);
    this.store.targetFrameIdx.set(0);
    this.store.displayedFrameIdx.set(-1);
    this.store.saveName.set('');
    this.store.trackedPoints.set([]);
    this.resetInteractiveMaps();
    this.frameCanvas.clearFrameCaches();
    this.store.objects.set([{ id: 1, name: 'Object 1', color: randomColor() }]);
    this.store.selectedObjectId.set(1);
    this.store.hasManifestMasks.set(Boolean(res.restored_session?.has_mask_manifest));
    this.updateStateEpoch(res.state_epoch, 'video init');
    this.restoreInteractiveSessionState(res.restored_session);

    const restoredTrackingResultId = res.restored_session?.tracking_result?.result_id;
    if (restoredTrackingResultId) {
      await this.loadTrackingResult(restoredTrackingResultId, 'Restored tracking');
    }

    const restoredWarnings = res.restored_session?.interactive_state_warnings || [];
    if (restoredWarnings.length > 0) {
      this.notify('warning', 'Session partially restored', restoredWarnings.slice(0, 2).join(' '));
    }

    this.resetDebugState();
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
    const name = this.store.saveName().trim();
    if (!name) {
      this.notify('warning', 'Missing save name', 'Enter a name for this saved session.');
      return;
    }
    const interactiveState = this.buildSnapshot();
    this.store.isLoading.set(true);
    firstValueFrom(this.backend.saveVideoSession(name, interactiveState))
      .then((response: VideoSaveResponse) => {
        this.updateStateEpoch(response.state_epoch, 'save');
        this.store.saveName.set(response.name);
        this.notify('success', 'Session saved', `Saved to ${response.saved_path}`);
      })
      .catch((error: unknown) => {
        console.error(error);
        this.notify('error', 'Save failed', getErrorMessage(error, 'Session save failed'));
      })
      .finally(() => {
        this.store.isLoading.set(false);
      });
  }

  // --- mask propagation / tracking ---------------------------------------

  async propagate(): Promise<void> {
    const response = await this.runJob<VideoPropagateResponse>('Propagating masks', () =>
      firstValueFrom(
        this.backend.propagateInVideo({
          include_masks_in_response: false,
          include_saved_mask_paths: false,
        }),
      ),
    );
    if (!response) {
      return;
    }
    this.updateStateEpoch(response.state_epoch, 'propagation');
    const maskManifestPath = response.mask_manifest_path || response['state.mask_manifest_path'];
    this.store.hasManifestMasks.set(Boolean(maskManifestPath));
    if (response.tracked_points_skipped_reason) {
      this.notify('warning', 'Tracking guidance skipped', response.tracked_points_skipped_reason);
    }
    this.frameCanvas.clearMaskDataCache();
    this.frameCanvas.scheduleFrameLoad(this.store.targetFrameIdx());
  }

  async runTracking(): Promise<void> {
    const response = await this.runJob<TrackPromptPointsJobResponse>('Tracking prompt points', () =>
      firstValueFrom(
        this.backend.trackPromptPoints({ add_support_grid: this.store.trackingUseSupportGrid() }),
      ),
    );
    if (!response) {
      return;
    }
    this.updateStateEpoch(response.state_epoch, 'tracking restore');
    const loaded = await this.loadTrackingResult(response.tracking_result_id, 'Tracking');
    if (loaded && this.lastCompletedJobId) {
      try {
        await firstValueFrom(this.backend.clearJobResult(this.lastCompletedJobId));
      } catch (error) {
        console.error(error);
      }
    }
  }

  clearMasks(): void {
    this.backend.resetVideoState().subscribe({
      next: (response) => {
        this.updateStateEpoch(response?.state_epoch, 'reset');
        this.store.hasManifestMasks.set(false);
        this.store.trackedPoints.set([]);
        this.resetInteractiveMaps();
        this.frameCanvas.clearMaskDataCache();
        this.frameCanvas.scheduleFrameLoad(this.store.targetFrameIdx());
      },
      error: (error) => {
        console.error(error);
        this.notify('error', 'Clear failed', getErrorMessage(error, 'Failed to clear masks.'));
      },
    });
  }

  private async loadTrackingResult(resultId: string, sourceLabel: string): Promise<boolean> {
    try {
      const response = await firstValueFrom(this.backend.getTrackingResult(resultId));
      const result: TrackPromptPointsResult = response.result;
      const trackedSeries: TrackedPointSeries[] = result.points.map((point, index) => ({
        ...point,
        tracks: result.tracks[index] || [],
        visibility: result.visibility[index] || [],
      }));
      this.store.trackedPoints.set(trackedSeries);
      this.frameCanvas.redraw();
      return true;
    } catch (error) {
      console.error(error);
      this.notify(
        'warning',
        'Tracking result unavailable',
        getErrorMessage(error, `${sourceLabel} result could not be loaded.`),
      );
      return false;
    }
  }

  // --- objects -----------------------------------------------------------

  addObject(): void {
    const newId = this.store.objects().length + 1;
    this.store.objects.set([
      ...this.store.objects(),
      { id: newId, name: `Object ${newId}`, color: randomColor() },
    ]);
    this.store.selectedObjectId.set(newId);
  }

  removeObject(): void {
    const id = this.store.selectedObjectId();
    if (id === null) return;

    this.backend.removeObject(id).subscribe({
      next: () => {
        this.store.objects.set(this.store.objects().filter((entry) => entry.id !== id));
        this.removeObjectFromFrameMaps(id);
        const nextObjects = this.store.objects();
        this.store.selectedObjectId.set(nextObjects.length > 0 ? nextObjects[0].id : null);
        this.frameCanvas.redraw();
      },
      error: (error) => this.showError('Remove failed', 'Failed to remove object.', error),
    });
  }

  async removeAllObjects(): Promise<void> {
    const objectIds = this.store.objects().map((entry) => entry.id);
    if (objectIds.length === 0) return;

    try {
      await Promise.all(objectIds.map((id) => firstValueFrom(this.backend.removeObject(id))));
      this.store.objects.set([]);
      this.store.selectedObjectId.set(null);
      this.store.masks.set(new Map());
      this.store.points.set(new Map());
      this.store.liveEditedObjectFrames.set(new Map());
      this.store.trackedPoints.set([]);
      this.frameCanvas.redraw();
    } catch (error) {
      this.showError('Remove all failed', 'Failed to remove all objects.', error);
    }
  }

  private removeObjectFromFrameMaps(objectId: number): void {
    const nextMasks = new Map(this.store.masks());
    nextMasks.forEach((frameMap) => frameMap.delete(objectId));
    this.store.masks.set(nextMasks);

    const nextPoints = new Map(this.store.points());
    nextPoints.forEach((frameMap) => frameMap.delete(objectId));
    this.store.points.set(nextPoints);

    this.store.trackedPoints.set(
      this.store.trackedPoints().filter((series) => series.obj_id !== objectId),
    );
  }

  // --- point prompting -------------------------------------------------------

  async addPoint(x: number, y: number, label: number, frameIdx: number): Promise<void> {
    const objId = this.store.selectedObjectId();
    if (objId === null) return;

    const expectedEpoch = this.store.stateEpoch();

    const pointsMap = new Map(this.store.points());
    const framePointsMap = new Map(pointsMap.get(frameIdx) || new Map<number, Point[]>());
    const previousObjectPoints = framePointsMap.get(objId) || [];
    const objectPoints = [...previousObjectPoints, { x, y, label }];
    framePointsMap.set(objId, objectPoints);
    pointsMap.set(frameIdx, framePointsMap);
    this.store.points.set(pointsMap);

    const requestFrameIdx = frameIdx;
    this.store.lastClickRequestFrameIdx.set(requestFrameIdx);
    this.store.lastDebugObjectId.set(objId);
    this.store.lastMaskPixelCount.set(null);
    this.store.lastBackendResponseFrameIdx.set(null);
    this.store.lastBackendResponseFrameFile.set('n/a');
    this.store.lastBackendResponseStateEpoch.set(null);
    this.store.lastFallbackUsed.set(false);
    this.store.lastDiscardReason.set(null);

    const request: VideoAddPointsOrBoxRequest = {
      frame_idx: requestFrameIdx,
      obj_id: objId,
      points: objectPoints.map((point) => [point.x, point.y]),
      labels: objectPoints.map((point) => point.label),
      clear_old_points: true,
    };

    const liveEditedBeforeRequest = this.isObjectLiveEdited(frameIdx, objId);

    try {
      this.store.isPointRequestInFlight.set(true);
      const response = await firstValueFrom(this.backend.addNewPointsOrBox(request));
      if (
        (response as any)?.error ||
        typeof (response as any)?.request_frame_idx !== 'number' ||
        typeof (response as any)?.frame_idx !== 'number' ||
        typeof (response as any)?.frame_file !== 'string' ||
        typeof (response as any)?.state_epoch !== 'number' ||
        !Array.isArray((response as any)?.out_obj_ids) ||
        !Array.isArray((response as any)?.out_masks) ||
        typeof (response as any)?.mask_pixel_counts !== 'object'
      ) {
        throw new Error((response as any)?.error || 'Invalid mask response');
      }

      const responseStateEpoch = Math.trunc(response.state_epoch);
      this.store.lastBackendResponseStateEpoch.set(responseStateEpoch);
      if (responseStateEpoch !== expectedEpoch) {
        const reason = `Discarded stale response due to epoch mismatch (expected ${expectedEpoch}, got ${responseStateEpoch}).`;
        this.updateStateEpoch(responseStateEpoch, 'add_new_points_or_box mismatch response');
        this.store.lastDiscardReason.set(reason);
        throw new Error(reason);
      }

      const responseRequestFrameIdx = Math.trunc(response.request_frame_idx);
      const responseFrameIdx = Math.trunc(response.frame_idx);
      this.store.lastBackendResponseFrameIdx.set(responseFrameIdx);
      this.store.lastBackendResponseFrameFile.set(response.frame_file || 'n/a');
      if (responseRequestFrameIdx !== requestFrameIdx || responseFrameIdx !== requestFrameIdx) {
        const reason = `Discarded response due to frame mismatch (request=${requestFrameIdx}, response_request=${responseRequestFrameIdx}, response_frame=${responseFrameIdx}).`;
        this.store.lastDiscardReason.set(reason);
        throw new Error(reason);
      }

      if (this.store.displayedFrameIdx() !== requestFrameIdx) {
        const reason = `Discarded response because displayed frame moved from ${requestFrameIdx} to ${this.store.displayedFrameIdx()}.`;
        this.store.lastDiscardReason.set(reason);
        throw new Error(reason);
      }

      const maskPixelCount = getMaskPixelCount(response.mask_pixel_counts, objId);
      this.store.lastMaskPixelCount.set(maskPixelCount);
      this.store.lastFallbackUsed.set(Boolean(response.single_frame_fallback_used));

      const masksMap = new Map(this.store.masks());
      const frameMasksMap = new Map(masksMap.get(requestFrameIdx) || new Map<number, boolean[][]>());
      response.out_obj_ids.forEach((id, index) => {
        frameMasksMap.set(id, response.out_masks[index]);
      });
      masksMap.set(requestFrameIdx, frameMasksMap);
      this.markObjectAsLiveEdited(requestFrameIdx, objId);
      this.store.masks.set(masksMap);
      this.frameCanvas.redraw();
    } catch (error) {
      console.error(error);
      this.notify(
        'error',
        'Point update failed',
        getErrorMessage(error, 'Unable to update point mask.'),
      );

      const rollbackPointsMap = new Map(this.store.points());
      const rollbackFramePointsMap = new Map(
        rollbackPointsMap.get(frameIdx) || new Map<number, Point[]>(),
      );
      if (previousObjectPoints.length > 0) {
        rollbackFramePointsMap.set(objId, previousObjectPoints);
      } else {
        rollbackFramePointsMap.delete(objId);
      }
      if (rollbackFramePointsMap.size === 0) {
        rollbackPointsMap.delete(frameIdx);
      } else {
        rollbackPointsMap.set(frameIdx, rollbackFramePointsMap);
      }
      if (!liveEditedBeforeRequest) {
        this.unmarkObjectAsLiveEdited(frameIdx, objId);
      }
      this.store.points.set(rollbackPointsMap);
      this.frameCanvas.redraw();
    } finally {
      this.store.isPointRequestInFlight.set(false);
    }
  }
}
