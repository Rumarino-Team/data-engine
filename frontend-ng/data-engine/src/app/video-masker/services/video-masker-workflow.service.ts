import { Injectable } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import {
  BackendService,
  TrackPromptPointsJobResponse,
  TrackPromptPointsResult,
  VideoInitStateResponse,
  VideoPropagateResponse,
  VideoSaveInteractiveState,
  VideoSaveResponse,
} from '../../services/backend.service';

interface WorkflowDeps {
  backend: BackendService;
  runBackendJob: <T>(title: string, startJob: () => Promise<{ job_id: string }>) => Promise<T | null>;
  updateStateEpoch: (epoch: number | undefined, source: string) => void;
  setHasManifestMasks: (value: boolean) => void;
  invalidateMaskCache: () => void;
  scheduleFrameLoad: (frameIdx: number) => void;
  targetFrameIdx: () => number;
  selectedFrameRange?: () => { start_frame_idx?: number; end_frame_idx?: number };
  setTrackedPoints: (next: any[]) => void;
  drawCurrentFrame: () => void;
  showToast: (severity: 'error' | 'warning' | 'info' | 'success', title: string, message: string) => void;
  getErrorMessage: (error: unknown, fallback: string) => string;
  getLastCompletedJobId: () => string | null;
  trackingUseSupportGrid: () => boolean;
  setIsLoading: (value: boolean) => void;
  getSaveName: () => string;
  setSaveName: (value: string) => void;
  buildInteractiveStateSnapshot: () => VideoSaveInteractiveState;
  resetInteractiveMaps: () => void;
  getVideoDir?: () => string;
  setVideoDir?: (value: string) => void;
  loadSourceMode?: () => 'frames_dir' | 'video_file' | 'saved_session_dir';
  clearFrameCaches?: () => void;
  setNumFrames?: (value: number) => void;
  setTargetFrameIdx?: (value: number) => void;
  setDisplayedFrameIdx?: (value: number) => void;
  setSelectedObjectId?: (value: number | null) => void;
  setObjects?: (objects: Array<{ id: number; name: string; color: string }>) => void;
  randomColor?: () => string;
  restoreInteractiveSessionState?: (restored: any) => void;
  resetDebugState?: () => void;
  loadTrackingResult?: (resultId: string, sourceLabel: string) => Promise<boolean>;
  setIsInitialized?: (value: boolean) => void;
}


@Injectable({ providedIn: 'root' })
export class VideoMaskerWorkflowService {

  async initVideo(deps: WorkflowDeps): Promise<boolean> {
    if (!deps.getVideoDir || !deps.setVideoDir || !deps.loadSourceMode || !deps.clearFrameCaches || !deps.setNumFrames || !deps.setTargetFrameIdx || !deps.setDisplayedFrameIdx || !deps.setSelectedObjectId || !deps.setObjects || !deps.randomColor || !deps.restoreInteractiveSessionState || !deps.resetDebugState) {
      throw new Error('Missing initVideo workflow dependencies');
    }

    const enteredPath = deps
      .getVideoDir()
      .trim()
      .replace(/^['"]|['"]$/g, '');
    if (!enteredPath) {
      if (deps.loadSourceMode() === 'saved_session_dir') {
        deps.showToast('warning', 'Missing session path', 'Enter a saved session directory path or browse for one.');
        return false;
      }
      if (deps.loadSourceMode() === 'video_file') {
        deps.showToast('warning', 'Missing video path', 'Enter a video file path or browse for one.');
        return false;
      }
      deps.showToast('warning', 'Missing frames path', 'Enter a frames directory path or browse for one.');
      return false;
    }

    deps.setVideoDir(enteredPath);
    const res = await deps.runBackendJob<VideoInitStateResponse>('Loading video', () =>
      firstValueFrom(deps.backend.initVideoState(enteredPath)),
    );
    if (!res) {
      return false;
    }

    deps.setNumFrames(res.num_frames);
    deps.setTargetFrameIdx(0);
    deps.setDisplayedFrameIdx(-1);
    deps.setSaveName('');
    deps.setTrackedPoints([]);
    deps.resetInteractiveMaps();
    deps.clearFrameCaches();
    deps.setObjects([{ id: 1, name: 'Object 1', color: deps.randomColor() }]);
    deps.setSelectedObjectId(1);
    deps.setHasManifestMasks(Boolean(res.restored_session?.has_mask_manifest));
    deps.updateStateEpoch(res.state_epoch, 'video init');
    deps.restoreInteractiveSessionState(res.restored_session);

    const restoredTrackingResultId = res.restored_session?.tracking_result?.result_id;
    if (restoredTrackingResultId && deps.loadTrackingResult) {
      await deps.loadTrackingResult(restoredTrackingResultId, 'Restored tracking');
    }

    const restoredWarnings = res.restored_session?.interactive_state_warnings || [];
    if (restoredWarnings.length > 0) {
      deps.showToast('warning', 'Session partially restored', restoredWarnings.slice(0, 2).join(' '));
    }

    deps.resetDebugState();
    deps.setIsInitialized?.(true);
    return true;
  }

  async propagate(deps: WorkflowDeps): Promise<void> {
    const frameRange = deps.selectedFrameRange?.() || {};
    const response = await deps.runBackendJob<VideoPropagateResponse>('Propagating masks', () =>
      firstValueFrom(deps.backend.propagateInVideo(frameRange)),
    );
    if (!response) {
      return;
    }
    deps.updateStateEpoch(response.state_epoch, 'propagation');
    const maskManifestPath = response.mask_manifest_path || response['state.mask_manifest_path'];
    deps.setHasManifestMasks(Boolean(maskManifestPath));
    if (response.tracked_points_skipped_reason) {
      deps.showToast('warning', 'Tracking guidance skipped', response.tracked_points_skipped_reason);
    }
    deps.invalidateMaskCache();
    deps.scheduleFrameLoad(deps.targetFrameIdx());
  }

  async runTracking(deps: WorkflowDeps): Promise<void> {
    const frameRange = deps.selectedFrameRange?.() || {};
    const response = await deps.runBackendJob<TrackPromptPointsJobResponse>(
      'Tracking prompt points',
      () =>
        firstValueFrom(
          deps.backend.trackPromptPoints({
            add_support_grid: deps.trackingUseSupportGrid(),
            ...frameRange,
          }),
        ),
    );
    if (!response) {
      return;
    }
    deps.updateStateEpoch(response.state_epoch, 'tracking restore');
    const loaded = await this.loadTrackingResult(deps, response.tracking_result_id, 'Tracking');
    if (loaded && deps.getLastCompletedJobId()) {
      try {
        await firstValueFrom(deps.backend.clearJobResult(deps.getLastCompletedJobId()!));
      } catch (error) {
        console.error(error);
      }
    }
  }

  clearMasks(deps: WorkflowDeps): void {
    deps.backend.resetVideoState().subscribe({
      next: (response) => {
        deps.updateStateEpoch(response?.state_epoch, 'reset');
        deps.setHasManifestMasks(false);
        deps.setTrackedPoints([]);
        deps.resetInteractiveMaps();
        deps.invalidateMaskCache();
        deps.scheduleFrameLoad(deps.targetFrameIdx());
      },
      error: (error) => {
        console.error(error);
        deps.showToast('error', 'Clear failed', deps.getErrorMessage(error, 'Failed to clear masks.'));
      },
    });
  }

  save(deps: WorkflowDeps): void {
    const name = deps.getSaveName().trim();
    if (!name) {
      deps.showToast('warning', 'Missing save name', 'Enter a name for this saved session.');
      return;
    }
    const interactiveState = deps.buildInteractiveStateSnapshot();
    deps.setIsLoading(true);
    firstValueFrom(deps.backend.saveVideoSession(name, interactiveState))
      .then((response: VideoSaveResponse) => {
        deps.updateStateEpoch(response.state_epoch, 'save');
        deps.setSaveName(response.name);
        deps.showToast('success', 'Session saved', `Saved to ${response.saved_path}`);
      })
      .catch((error: unknown) => {
        console.error(error);
        deps.showToast('error', 'Save failed', deps.getErrorMessage(error, 'Session save failed'));
      })
      .finally(() => {
        deps.setIsLoading(false);
      });
  }

  private async loadTrackingResult(
    deps: WorkflowDeps,
    resultId: string,
    sourceLabel: string,
  ): Promise<boolean> {
    try {
      const response = await firstValueFrom(deps.backend.getTrackingResult(resultId));
      const trackedSeries = response.result.points.map((point: any, index: number) => ({
        ...point,
        tracks: response.result.tracks[index] || [],
        visibility: response.result.visibility[index] || [],
      })) as TrackPromptPointsResult['points'];
      deps.setTrackedPoints(trackedSeries as any[]);
      deps.drawCurrentFrame();
      return true;
    } catch (error) {
      console.error(error);
      deps.showToast(
        'warning',
        'Tracking result unavailable',
        deps.getErrorMessage(error, `${sourceLabel} result could not be loaded.`),
      );
      return false;
    }
  }
}
