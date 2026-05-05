import { ElementRef, Injectable, OnDestroy, effect, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import {
  BackendService,
  InteractiveMaskRle,
  JobStageHistoryEntry,
  RestoredSessionPayload,
  TrackPromptPointsResult,
  VideoSaveInteractiveState,
} from '../services/backend.service';
import { DesktopBridgeService } from '../services/desktop-bridge.service';
import { ToastService } from './services/toast.service';
import { VideoJobsService } from './services/video-jobs.service';
import { MaskStateService } from './services/mask-state.service';
import { VideoSessionService } from './services/video-session.service';
import { FrameRendererService } from './services/frame-renderer.service';
import { VideoMaskerSessionStateService } from './services/video-masker-session-state.service';
import { VideoMaskerRenderingService } from './services/video-masker-rendering.service';
import { VideoMaskerCommandsService } from './services/video-masker-commands.service';
import { VideoMaskerWorkflowService } from './services/video-masker-workflow.service';
import { FramePipelineState, VideoMaskerFramePipelineService } from './services/video-masker-frame-pipeline.service';
import { MaskOverlayCacheService } from './services/mask-overlay-cache.service';
import {
  Point,
  TrackedPointSeries,
  VideoMaskerStateStore,
} from './services/video-masker-state.store';
import { DebugMaskSource, LoadSourceMode, ToastSeverity } from './state/video-masker-ui.types';

@Injectable({ providedIn: 'root' })
export class VideoMaskerFacade implements OnDestroy {
  canvasRef?: ElementRef<HTMLCanvasElement>;
  videoFileInputRef?: ElementRef<HTMLInputElement>;
  framesDirInputRef?: ElementRef<HTMLInputElement>;

  bindViewRefs(refs: {
    canvasRef?: ElementRef<HTMLCanvasElement>;
    videoFileInputRef?: ElementRef<HTMLInputElement>;
    framesDirInputRef?: ElementRef<HTMLInputElement>;
  }): void {
    this.canvasRef = refs.canvasRef;
    this.videoFileInputRef = refs.videoFileInputRef;
    this.framesDirInputRef = refs.framesDirInputRef;
  }
  protected store = inject(VideoMaskerStateStore);

  videoDir = this.store.videoDir;
  loadSourceMode = this.store.loadSourceMode;
  apiUrlInput = this.store.apiUrlInput;
  isInitialized = this.store.isInitialized;
  numFrames = this.store.numFrames;
  targetFrameIdx = this.store.targetFrameIdx;
  displayedFrameIdx = this.store.displayedFrameIdx;
  rangeStartFrameIdx = this.store.rangeStartFrameIdx;
  rangeEndFrameIdx = this.store.rangeEndFrameIdx;
  stateEpoch = this.store.stateEpoch;

  objects = this.store.objects;
  selectedObjectId = this.store.selectedObjectId;
  editingObjectId = signal<number | null>(null);
  selectedPoint = this.store.selectedPoint;
  interactionMode = this.store.interactionMode;

  masks = this.store.masks;
  points = this.store.points;
  liveEditedObjectFrames = this.store.liveEditedObjectFrames;
  hasManifestMasks = this.store.hasManifestMasks;
  saveName = this.store.saveName;

  trackingOverlayStyle = this.store.trackingOverlayStyle;
  trackingUseSupportGrid = this.store.trackingUseSupportGrid;
  trackedPoints = this.store.trackedPoints;

  isLoading = this.store.isLoading;
  isFrameLoading = this.store.isFrameLoading;
  isPointRequestInFlight = this.store.isPointRequestInFlight;
  apiHealthStatus = this.store.apiHealthStatus;
  activeJob = this.store.activeJob;
  activeJobTitle = this.store.activeJobTitle;
  toasts = this.store.toasts;
  lastClickRequestFrameIdx = this.store.lastClickRequestFrameIdx;
  lastBackendResponseFrameIdx = this.store.lastBackendResponseFrameIdx;
  lastBackendResponseFrameFile = this.store.lastBackendResponseFrameFile;
  lastBackendResponseStateEpoch = this.store.lastBackendResponseStateEpoch;
  lastDebugObjectId = this.store.lastDebugObjectId;
  lastMaskPixelCount = this.store.lastMaskPixelCount;
  lastFallbackUsed = this.store.lastFallbackUsed;
  lastMaskSource = this.store.lastMaskSource;
  lastDiscardReason = this.store.lastDiscardReason;

  private framePipelineState: FramePipelineState = {
    frameLoadToken: 0,
    pendingFrameIdx: null,
    frameLoadAnimationId: null,
    frameImageCache: new Map<number, HTMLImageElement>(),
    maxFrameCacheSize: 24,
    currentBaseImage: null,
    previousFrameIdx: null,
  };
  private healthTimerId: ReturnType<typeof setInterval> | null = null;
  private lastCompletedJobId: string | null = null;

  constructor(
    protected backend: BackendService,
    protected desktopBridge: DesktopBridgeService,
    protected toastService: ToastService,
    protected jobsService: VideoJobsService,
    protected maskStateService: MaskStateService,
    protected videoSessionService: VideoSessionService,
    protected frameRendererService: FrameRendererService,
    protected sessionStateService: VideoMaskerSessionStateService,
    protected renderingService: VideoMaskerRenderingService,
    protected commandsService: VideoMaskerCommandsService,
    protected workflowService: VideoMaskerWorkflowService,
    protected framePipelineService: VideoMaskerFramePipelineService,
    protected maskOverlayCache: MaskOverlayCacheService,
  ) {
    this.apiUrlInput.set(this.backend.getApiUrl());

    effect(() => {
      if (this.isInitialized()) {
        this.reconcileSelectedPoint();
        this.scheduleFrameLoad(this.targetFrameIdx());
      }
    });

    effect(() => {
      this.trackingOverlayStyle();
      this.trackedPoints();
      if (this.framePipelineState.currentBaseImage) {
        this.drawCurrentFrame();
      }
    });

    this.checkApiHealth(true);
    this.healthTimerId = setInterval(() => this.checkApiHealth(), 3000);
  }

  ngOnDestroy(): void {
    if (this.healthTimerId !== null) {
      clearInterval(this.healthTimerId);
    }
    if (this.framePipelineState.frameLoadAnimationId !== null) {
      cancelAnimationFrame(this.framePipelineState.frameLoadAnimationId);
    }
  }

  private async checkApiHealth(showChecking = false): Promise<void> {
    if (showChecking || this.apiHealthStatus() === 'checking') {
      this.apiHealthStatus.set('checking');
    }
    try {
      await firstValueFrom(this.backend.health());
      this.apiHealthStatus.set('online');
    } catch {
      this.apiHealthStatus.set('offline');
    }
  }

  private showToast(severity: ToastSeverity, title: string, message: string): void {
    this.toastService.show(severity, title, message);
    this.toasts.set(this.toastService.toasts());
  }

  dismissToast(id: number): void {
    this.toastService.dismiss(id);
    this.toasts.set(this.toastService.toasts());
  }

  recentJobHistory(): JobStageHistoryEntry[] {
    const history = this.activeJob()?.stage_history || [];
    return history.slice(-3);
  }

  private getErrorMessage(error: any, fallback: string): string {
    return error?.error?.detail || error?.error?.error || error?.message || fallback;
  }

  private async runBackendJob<T>(
    title: string,
    startJob: () => Promise<{ job_id: string }>,
  ): Promise<T | null> {
    const { result, completedJobId } = await this.jobsService.run<T>({
      title,
      startJob,
      onStart: () => {
        this.isLoading.set(true);
        this.activeJobTitle.set(title);
        this.activeJob.set(null);
        this.lastCompletedJobId = null;
      },
      onStatus: (job) => this.activeJob.set(job),
      onFailure: (message) => this.showToast('error', title, message),
      onFinish: () => {
        this.activeJob.set(null);
        this.activeJobTitle.set('');
        this.isLoading.set(false);
      },
      fallbackErrorMessage: `${title} failed`,
    });
    this.lastCompletedJobId = completedJobId;
    return result;
  }

  private updateStateEpoch(nextEpoch: number | undefined, source: string): void {
    const resolution = this.maskStateService.updateStateEpoch(this.stateEpoch(), nextEpoch);
    if (!resolution) {
      return;
    }
    if (resolution.shouldClearLiveState) {
      this.masks.set(new Map());
      this.liveEditedObjectFrames.set(new Map());
      this.lastDiscardReason.set(
        `State epoch changed (${this.stateEpoch()} -> ${resolution.normalizedEpoch}) during ${source}; cleared live masks.`,
      );
    }
    this.stateEpoch.set(resolution.normalizedEpoch);
  }

  private selectedFrameRangeRequest(): { start_frame_idx?: number; end_frame_idx?: number } {
    const start = this.rangeStartFrameIdx();
    const end = this.rangeEndFrameIdx();
    if (start === null && end === null) {
      return {};
    }
    const maxFrame = this.numFrames() - 1;
    const fallbackFrame = this.frameRendererService.clampFrameIndex(this.targetFrameIdx(), maxFrame);
    const normalizedStart = this.frameRendererService.clampFrameIndex(
      start ?? end ?? fallbackFrame,
      maxFrame,
    );
    const normalizedEnd = this.frameRendererService.clampFrameIndex(
      end ?? start ?? fallbackFrame,
      maxFrame,
    );
    return {
      start_frame_idx: Math.min(normalizedStart, normalizedEnd),
      end_frame_idx: Math.max(normalizedStart, normalizedEnd),
    };
  }

  private markObjectAsLiveEdited(frameIdx: number, objId: number): void {
    this.liveEditedObjectFrames.set(
      this.maskStateService.markObjectAsLiveEdited(this.liveEditedObjectFrames(), frameIdx, objId),
    );
  }

  private unmarkObjectAsLiveEdited(frameIdx: number, objId: number): void {
    this.liveEditedObjectFrames.set(
      this.maskStateService.unmarkObjectAsLiveEdited(
        this.liveEditedObjectFrames(),
        frameIdx,
        objId,
      ),
    );
  }

  private isObjectLiveEdited(frameIdx: number, objId: number): boolean {
    return this.maskStateService.isObjectLiveEdited(this.liveEditedObjectFrames(), frameIdx, objId);
  }

  private resetDebugState(): void {
    this.lastClickRequestFrameIdx.set(null);
    this.lastBackendResponseFrameIdx.set(null);
    this.lastBackendResponseFrameFile.set('n/a');
    this.lastBackendResponseStateEpoch.set(null);
    this.lastDebugObjectId.set(null);
    this.lastMaskPixelCount.set(null);
    this.lastFallbackUsed.set(false);
    this.lastMaskSource.set('none');
    this.lastDiscardReason.set(null);
  }

  private getMaskPixelCount(
    pixelCounts: Record<number, number> | undefined,
    objId: number,
  ): number | null {
    if (!pixelCounts) {
      return null;
    }
    const direct = (pixelCounts as Record<number, number>)[objId];
    if (typeof direct === 'number' && Number.isFinite(direct)) {
      return Math.trunc(direct);
    }
    const stringLookup = (pixelCounts as unknown as Record<string, number>)[String(objId)];
    if (typeof stringLookup === 'number' && Number.isFinite(stringLookup)) {
      return Math.trunc(stringLookup);
    }
    return null;
  }

  openVideoFilePicker() {
    this.videoFileInputRef?.nativeElement.click();
  }

  openFramesDirPicker() {
    this.framesDirInputRef?.nativeElement.click();
  }

  onLoadSourceModeChange(value: string) {
    const nextMode: LoadSourceMode =
      value === 'video_file' || value === 'saved_session_dir' ? value : 'frames_dir';
    this.loadSourceMode.set(nextMode);
  }

  onVideoDirChange(value: string) {
    this.videoDir.set(value);
  }

  onApiUrlChange(value: string) {
    this.apiUrlInput.set(value);
  }

  onSaveNameChange(value: string) {
    this.saveName.set(value);
  }

  getLoadPathPlaceholder(): string {
    return this.videoSessionService.getLoadPathPlaceholder(this.loadSourceMode());
  }

  getBrowseLabel(): string {
    return this.videoSessionService.getBrowseLabel(this.loadSourceMode());
  }


  applyApiUrl() {
    this.apiUrlInput.set(this.backend.setApiUrl(this.apiUrlInput()));
    this.checkApiHealth(true);
  }

  resetApiUrl() {
    this.apiUrlInput.set(this.backend.resetApiUrl());
    this.checkApiHealth(true);
  }

  async browseSelectedSource() {
    const selectedPath = await this.videoSessionService.browse(this.loadSourceMode());
    if (selectedPath) {
      this.videoDir.set(selectedPath);
      return;
    }

    if (this.loadSourceMode() === 'video_file') {
      this.openVideoFilePicker();
      return;
    }

    this.openFramesDirPicker();
  }

  async browseVideo() {
    this.loadSourceMode.set('video_file');
    await this.browseSelectedSource();
  }

  async browseFramesDirectory() {
    this.loadSourceMode.set('frames_dir');
    await this.browseSelectedSource();
  }

  onVideoFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    const nativePath = this.getNativeFilePath(file);
    if (nativePath) {
      this.videoDir.set(nativePath);
    } else {
      this.videoDir.set(file.name);
      this.showPathUnavailableMessage('video');
    }

    input.value = '';
  }

  onFramesDirSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    const nativePath = this.getNativeFilePath(file);
    if (nativePath) {
      this.videoDir.set(this.getParentDirectory(nativePath));
    } else {
      this.showPathUnavailableMessage('directory');
    }

    input.value = '';
  }

  private getNativeFilePath(file: File): string | null {
    const fileWithPath = file as File & { path?: string };
    if (typeof fileWithPath.path === 'string' && fileWithPath.path.trim()) {
      return fileWithPath.path.trim();
    }
    return null;
  }

  private getParentDirectory(filePath: string): string {
    const separatorIndex = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'));
    if (separatorIndex <= 0) {
      return filePath;
    }
    return filePath.slice(0, separatorIndex);
  }

  private showPathUnavailableMessage(target: 'video' | 'directory') {
    if (target === 'video') {
      this.showToast(
        'warning',
        'Path unavailable',
        'Selected video file name is available, but this browser does not expose the full local path. Paste the full video path manually.',
      );
      return;
    }
    if (this.loadSourceMode() === 'saved_session_dir') {
      this.showToast(
        'warning',
        'Path unavailable',
        'Selected folder contents are available, but this browser does not expose the full local directory path. Paste the full saved session directory path manually.',
      );
      return;
    }
    this.showToast(
      'warning',
      'Path unavailable',
      'Selected folder contents are available, but this browser does not expose the full local directory path. Paste the full frames directory path manually.',
    );
  }

  isApiUrlDirty(): boolean {
    return this.apiUrlInput().trim() !== this.backend.getApiUrl();
  }

  async initVideo() {
    await this.workflowService.initVideo({
      backend: this.backend,
      runBackendJob: (title, startJob) => this.runBackendJob(title, startJob),
      updateStateEpoch: (epoch, source) => this.updateStateEpoch(epoch, source),
      setHasManifestMasks: (value) => this.hasManifestMasks.set(value),
      invalidateMaskCache: () => this.maskOverlayCache.clearAll(),
      scheduleFrameLoad: (frameIdx) => this.scheduleFrameLoad(frameIdx),
      targetFrameIdx: () => this.targetFrameIdx(),
      setTrackedPoints: (next) => this.trackedPoints.set(next as any),
      drawCurrentFrame: () => this.drawCurrentFrame(),
      showToast: (severity, title, message) => this.showToast(severity, title, message),
      getErrorMessage: (error, fallback) => this.getErrorMessage(error, fallback),
      getLastCompletedJobId: () => this.lastCompletedJobId,
      trackingUseSupportGrid: () => this.trackingUseSupportGrid(),
      setIsLoading: (value) => this.isLoading.set(value),
      getSaveName: () => this.saveName(),
      setSaveName: (value) => this.saveName.set(value),
      buildInteractiveStateSnapshot: () => this.buildInteractiveStateSnapshot(),
      resetInteractiveMaps: () => {
        this.masks.set(new Map());
        this.points.set(new Map());
        this.liveEditedObjectFrames.set(new Map());
        this.selectedPoint.set(null);
        this.clearFrameRange();
      },
      getVideoDir: () => this.videoDir(),
      setVideoDir: (value) => this.videoDir.set(value),
      loadSourceMode: () => this.loadSourceMode(),
      clearFrameCaches: () => this.clearFrameCaches(),
      setNumFrames: (value) => this.numFrames.set(value),
      setTargetFrameIdx: (value) => this.targetFrameIdx.set(value),
      setDisplayedFrameIdx: (value) => this.displayedFrameIdx.set(value),
      setSelectedObjectId: (value) => this.selectedObjectId.set(value),
      setObjects: (objects) => this.objects.set(objects as any),
      randomColor: () => this.getRandomColor(),
      restoreInteractiveSessionState: (restored) => this.restoreInteractiveSessionState(restored),
      resetDebugState: () => this.resetDebugState(),
      loadTrackingResult: (resultId, sourceLabel) => this.loadTrackingResult(resultId, sourceLabel),
      setIsInitialized: (value) => this.isInitialized.set(value),
    });
  }

  private restoreInteractiveSessionState(
    restored: RestoredSessionPayload | null | undefined,
  ): void {
    if (!restored?.interactive_state) {
      return;
    }
    const interactive = restored.interactive_state;
    const restoredObjects = this.sessionStateService.normalizeInteractiveObjects(
      interactive.objects || [],
      () => this.getRandomColor(),
    );
    if (restoredObjects.length > 0) {
      this.objects.set(restoredObjects);
    }

    const restoredPoints = this.sessionStateService.deserializePoints(interactive.points || []);
    if (restoredPoints.size > 0) {
      this.points.set(restoredPoints);
    }

    const { masks, liveEditedFrames } = this.sessionStateService.deserializeLiveMasks(
      interactive.live_masks || [],
      (entry) => this.decodeMaskFromCounts(entry),
    );
    if (masks.size > 0) {
      this.masks.set(masks);
      this.liveEditedObjectFrames.set(liveEditedFrames);
    }
    this.ensureObjectsForRestoredState();

    const availableObjectIds = new Set(this.objects().map((objectEntry) => objectEntry.id));
    const requestedObjectId = interactive.selected_object_id ?? null;
    if (requestedObjectId !== null && availableObjectIds.has(requestedObjectId)) {
      this.selectedObjectId.set(requestedObjectId);
    } else if (this.objects().length > 0) {
      this.selectedObjectId.set(this.objects()[0].id);
    }

    if (
      interactive.interaction_mode === 'positive' ||
      interactive.interaction_mode === 'negative'
    ) {
      this.interactionMode.set(interactive.interaction_mode);
    }
    if (
      typeof interactive.current_frame_idx === 'number' &&
      Number.isFinite(interactive.current_frame_idx) &&
      interactive.current_frame_idx >= 0 &&
      interactive.current_frame_idx < this.numFrames()
    ) {
      this.targetFrameIdx.set(Math.trunc(interactive.current_frame_idx));
    }
  }

  private ensureObjectsForRestoredState(): void {
    const existing = new Map(this.objects().map((entry) => [entry.id, entry]));
    const requiredObjectIds = new Set<number>();
    this.points().forEach((framePoints) => {
      framePoints.forEach((_objPoints, objId) => requiredObjectIds.add(objId));
    });
    this.masks().forEach((frameMasks) => {
      frameMasks.forEach((_mask, objId) => requiredObjectIds.add(objId));
    });
    let changed = false;
    for (const objId of requiredObjectIds) {
      if (existing.has(objId)) {
        continue;
      }
      existing.set(objId, {
        id: objId,
        name: `Object ${objId}`,
        color: this.getRandomColor(),
      });
      changed = true;
    }
    if (changed) {
      this.objects.set(Array.from(existing.values()).sort((a, b) => a.id - b.id));
    }
  }

  private buildInteractiveStateSnapshot(): VideoSaveInteractiveState {
    return this.sessionStateService.buildInteractiveStateSnapshot({
      objects: this.objects(),
      selectedObjectId: this.selectedObjectId(),
      interactionMode: this.interactionMode(),
      currentFrameIdx: this.targetFrameIdx(),
      pointsByFrame: this.points(),
      masksByFrame: this.masks(),
      encodeMaskToCounts: (mask) => this.encodeMaskToCounts(mask),
    });
  }

  private encodeMaskToCounts(
    mask: boolean[][],
  ): { width: number; height: number; counts: number[] } | null {
    const normalizedMask = this.renderingService.normalizeMask2d(mask);
    if (!normalizedMask || normalizedMask.length === 0 || normalizedMask[0].length === 0) {
      return null;
    }
    const height = normalizedMask.length;
    const width = normalizedMask[0].length;
    const counts: number[] = [];
    let currentValue = false;
    let currentRun = 0;
    for (let y = 0; y < height; y++) {
      const row = normalizedMask[y];
      if (!Array.isArray(row) || row.length !== width) {
        return null;
      }
      for (let x = 0; x < width; x++) {
        const value = Boolean(row[x]);
        if (value === currentValue) {
          currentRun += 1;
          continue;
        }
        counts.push(currentRun);
        currentRun = 1;
        currentValue = value;
      }
    }
    counts.push(currentRun);
    return { width, height, counts };
  }

  private decodeMaskFromCounts(maskRle: InteractiveMaskRle): boolean[][] | null {
    const height = Math.trunc(maskRle.height);
    const width = Math.trunc(maskRle.width);
    if (height <= 0 || width <= 0) {
      return null;
    }
    if (!Array.isArray(maskRle.counts) || maskRle.counts.length === 0) {
      return null;
    }
    const totalPixels = width * height;
    const flatMask = new Array<boolean>(totalPixels).fill(false);
    let index = 0;
    let foreground = false;
    for (const rawCount of maskRle.counts) {
      const count = Math.trunc(rawCount);
      if (!Number.isFinite(count) || count < 0) {
        return null;
      }
      const end = index + count;
      if (end > totalPixels) {
        return null;
      }
      if (foreground) {
        for (let cursor = index; cursor < end; cursor++) {
          flatMask[cursor] = true;
        }
      }
      index = end;
      foreground = !foreground;
    }
    if (index !== totalPixels) {
      return null;
    }
    const mask2d: boolean[][] = [];
    for (let y = 0; y < height; y++) {
      const rowStart = y * width;
      mask2d.push(flatMask.slice(rowStart, rowStart + width));
    }
    return mask2d;
  }

  private clearFrameCaches(): void {
    this.framePipelineService.clearFrameCaches(this.framePipelineState);
    this.maskOverlayCache.clearAll();
  }

  selectObject(objectId: number): void {
    this.selectedObjectId.set(objectId);
    const selectedPoint = this.selectedPoint();
    if (selectedPoint && selectedPoint.objId !== objectId) {
      this.selectedPoint.set(null);
    }
  }

  pointLayersForObject(objectId: number): Array<{ frameIdx: number; pointIdx: number; point: Point }> {
    const frameIdx = this.displayedFrameIdx();
    if (frameIdx < 0) {
      return [];
    }
    return (this.points().get(frameIdx)?.get(objectId) || []).map((point, pointIdx) => ({
      frameIdx,
      pointIdx,
      point,
    }));
  }

  selectPoint(frameIdx: number, objId: number, pointIdx: number): void {
    this.selectedObjectId.set(objId);
    this.selectedPoint.set({ frameIdx, objId, pointIdx });
  }

  isSelectedPoint(frameIdx: number, objId: number, pointIdx: number): boolean {
    const selectedPoint = this.selectedPoint();
    return Boolean(
      selectedPoint &&
        selectedPoint.frameIdx === frameIdx &&
        selectedPoint.objId === objId &&
        selectedPoint.pointIdx === pointIdx,
    );
  }

  private reconcileSelectedPoint(): void {
    const selectedPoint = this.selectedPoint();
    if (!selectedPoint) {
      return;
    }
    const pointExists = Boolean(
      this.points().get(selectedPoint.frameIdx)?.get(selectedPoint.objId)?.[selectedPoint.pointIdx],
    );
    const objectExists = this.objects().some((entry) => entry.id === selectedPoint.objId);
    if (!pointExists || !objectExists || selectedPoint.frameIdx !== this.displayedFrameIdx()) {
      this.selectedPoint.set(null);
    }
  }

  private refreshManifestMasks(): void {
    this.maskOverlayCache.clearAll();
    this.scheduleFrameLoad(this.targetFrameIdx());
  }

  private scheduleFrameLoad(frameIdx: number) {
    this.framePipelineService.scheduleFrameLoad(frameIdx, this.framePipelineState, this.framePipelineDeps());
  }

  loadFrame(frameIdx: number) {
    void this.framePipelineService.loadFrame(frameIdx, this.framePipelineState, this.framePipelineDeps());
  }

  private drawCurrentFrame() {
    const frameIdx = this.displayedFrameIdx();
    if (frameIdx < 0 || !this.framePipelineState.currentBaseImage || !this.canvasRef?.nativeElement) {
      return;
    }
    this.draw(this.framePipelineState.currentBaseImage, frameIdx);
  }

  private framePipelineDeps() {
    return {
      canvasRef: this.canvasRef,
      getVideoFrameUrl: (frameIdx: number) => this.backend.getVideoFrameUrl(frameIdx),
      hasManifestMasks: () => this.hasManifestMasks(),
      numFrames: () => this.numFrames(),
      objects: () => this.objects(),
      liveEditedObjectIdsForFrame: (frameIdx: number) => (
        this.liveEditedObjectFrames().get(frameIdx) ?? new Set<number>()
      ),
      draw: (image: HTMLImageElement, frameIdx: number) => this.draw(image, frameIdx),
      onDisplayedFrame: (frameIdx: number) => this.displayedFrameIdx.set(frameIdx),
      setIsFrameLoading: (loading: boolean) => this.isFrameLoading.set(loading),
    };
  }

  draw(img: HTMLImageElement, frameIdx: number) {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const liveFrameMasks = this.masks().get(frameIdx);
    const liveEditedObjectIds = this.liveEditedObjectFrames().get(frameIdx) ?? new Set<number>();

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    if (this.hasManifestMasks()) {
      const overlay = this.maskOverlayCache.getPreparedOverlay(
        frameIdx,
        this.objects(),
        liveEditedObjectIds,
      );
      if (overlay) {
        this.renderingService.drawPreparedMaskOverlay(ctx, overlay.bitmap);
      }
    }

    if (liveFrameMasks) {
      liveFrameMasks.forEach((mask, objId) => {
        const normalizedMask = this.renderingService.normalizeMask2d(mask);
        if (!normalizedMask || !this.renderingService.maskHasForeground(normalizedMask)) {
          return;
        }
        const obj = this.objects().find((candidate) => candidate.id === objId);
        if (obj) {
          this.renderingService.drawMask(ctx, normalizedMask, obj.color);
        }
      });
    }

    const framePoints = this.points().get(frameIdx);
    if (framePoints) {
      framePoints.forEach((frameObjPoints, objId) => {
        frameObjPoints.forEach((point, pointIdx) => {
          this.renderingService.drawPoint(
            ctx,
            point,
            this.isSelectedPoint(frameIdx, objId, pointIdx),
          );
        });
      });
    }

    const selectedObjectId = this.selectedObjectId();
    let maskSource: DebugMaskSource = 'none';
    if (selectedObjectId !== null) {
      const selectedLiveMask = liveFrameMasks?.get(selectedObjectId);
      const normalizedLiveMask = selectedLiveMask
      ? this.renderingService.normalizeMask2d(selectedLiveMask)
      : null;
      const hasLiveMask = Boolean(normalizedLiveMask && this.renderingService.maskHasForeground(normalizedLiveMask));
      if (hasLiveMask) {
        maskSource = 'live';
      } else if (
        !this.isObjectLiveEdited(frameIdx, selectedObjectId) &&
        this.maskOverlayCache.hasMaskForObject(frameIdx, selectedObjectId)
      ) {
        maskSource = 'manifest';
      }
    }
    this.lastMaskSource.set(maskSource);

    this.renderingService.drawTrackingOverlay(
      ctx,
      frameIdx,
      this.trackingOverlayStyle(),
      this.trackedPoints(),
      this.objects(),
    );
  }







  onCanvasClick(event: MouseEvent) {
    if (
      !this.isInitialized() ||
      this.selectedObjectId() === null ||
      this.isFrameLoading() ||
      this.isPointRequestInFlight() ||
      !this.framePipelineState.currentBaseImage
    ) {
      return;
    }

    const canvasEl = this.canvasRef?.nativeElement;
    if (!canvasEl) return;
    const rect = canvasEl.getBoundingClientRect();
    const scaleX = canvasEl.width / rect.width;
    const scaleY = canvasEl.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    const label = this.interactionMode() === 'positive' ? 1 : 0;
    const frameIdx = this.displayedFrameIdx();
    if (frameIdx < 0) {
      return;
    }
    this.addPoint(x, y, label, frameIdx);
  }

  async addPoint(x: number, y: number, label: number, frameIdx: number) {
    const objId = this.selectedObjectId();
    if (objId === null) return;

    await this.commandsService.addPoint({
      backend: this.backend,
      getObjects: () => this.objects(),
      frameIdx,
      label,
      objId,
      x,
      y,
      expectedEpoch: this.stateEpoch(),
      displayedFrameIdx: () => this.displayedFrameIdx(),
      getPointsMap: () => this.points(),
      setPointsMap: (map) => this.points.set(map),
      getMasksMap: () => this.masks(),
      setMasksMap: (map) => this.masks.set(map),
      setPointRequestInFlight: (loading) => this.isPointRequestInFlight.set(loading),
      onBeforeRequest: (requestFrameIdx, requestObjId) => {
        this.lastClickRequestFrameIdx.set(requestFrameIdx);
        this.lastDebugObjectId.set(requestObjId);
        this.lastMaskPixelCount.set(null);
        this.lastBackendResponseFrameIdx.set(null);
        this.lastBackendResponseFrameFile.set('n/a');
        this.lastBackendResponseStateEpoch.set(null);
        this.lastFallbackUsed.set(false);
        this.lastDiscardReason.set(null);
      },
      getMaskPixelCount: (pixelCounts, objectId) => this.getMaskPixelCount(pixelCounts, objectId),
      setResponseEpoch: (epoch) => this.lastBackendResponseStateEpoch.set(epoch),
      setResponseFrame: (responseFrameIdx, frameFile) => {
        this.lastBackendResponseFrameIdx.set(responseFrameIdx);
        this.lastBackendResponseFrameFile.set(frameFile);
      },
      setMaskDebug: (pixelCount, fallbackUsed) => {
        this.lastMaskPixelCount.set(pixelCount);
        this.lastFallbackUsed.set(fallbackUsed);
      },
      setDiscardReason: (reason) => this.lastDiscardReason.set(reason),
      onEpochMismatch: (responseEpoch) =>
        this.updateStateEpoch(responseEpoch, 'add_new_points_or_box mismatch response'),
      markObjectAsLiveEdited: (requestFrameIdx, objectId) =>
        this.markObjectAsLiveEdited(requestFrameIdx, objectId),
      wasLiveEditedBeforeRequest: (requestFrameIdx, objectId) =>
        this.isObjectLiveEdited(requestFrameIdx, objectId),
      unmarkObjectAsLiveEdited: (requestFrameIdx, objectId) =>
        this.unmarkObjectAsLiveEdited(requestFrameIdx, objectId),
      redraw: () => this.drawCurrentFrame(),
      onFailure: (error) => {
        console.error(error);
        this.showToast(
          'error',
          'Point update failed',
          this.getErrorMessage(error, 'Unable to update point mask.'),
        );
      },
    });
  }

  async removeSelectedPoint(): Promise<void> {
    const selectedPoint = this.selectedPoint();
    if (!selectedPoint || this.isPointRequestInFlight()) {
      return;
    }
    const framePointsMap = this.points().get(selectedPoint.frameIdx);
    const objectPoints = framePointsMap?.get(selectedPoint.objId) || [];
    if (!objectPoints[selectedPoint.pointIdx]) {
      this.selectedPoint.set(null);
      return;
    }

    const remainingPoints = objectPoints.filter((_, index) => index !== selectedPoint.pointIdx);
    this.isPointRequestInFlight.set(true);
    try {
      if (remainingPoints.length > 0) {
        const response = await firstValueFrom(
          this.backend.addNewPointsOrBox({
            frame_idx: selectedPoint.frameIdx,
            obj_id: selectedPoint.objId,
            points: remainingPoints.map((point) => [point.x, point.y]),
            labels: remainingPoints.map((point) => point.label),
            clear_old_points: true,
          }),
        );
        if ((response as any)?.error) {
          throw new Error((response as any).error);
        }
        this.updateStateEpoch(response.state_epoch, 'remove point');
        this.replaceObjectPoints(selectedPoint.frameIdx, selectedPoint.objId, remainingPoints);
        const masksMap = new Map(this.masks());
        const frameMasksMap = new Map(
          masksMap.get(selectedPoint.frameIdx) || new Map<number, boolean[][]>(),
        );
        response.out_obj_ids.forEach((id, index) => {
          if (this.objects().some((entry) => entry.id === id)) {
            frameMasksMap.set(id, response.out_masks[index]);
          }
        });
        masksMap.set(selectedPoint.frameIdx, frameMasksMap);
        this.masks.set(masksMap);
        this.markObjectAsLiveEdited(selectedPoint.frameIdx, selectedPoint.objId);
      } else {
        await firstValueFrom(
          this.backend.clearAllPromptsInFrame(selectedPoint.frameIdx, selectedPoint.objId),
        );
        this.replaceObjectPoints(selectedPoint.frameIdx, selectedPoint.objId, []);
        const masksMap = new Map(this.masks());
        const frameMasksMap = new Map(
          masksMap.get(selectedPoint.frameIdx) || new Map<number, boolean[][]>(),
        );
        frameMasksMap.delete(selectedPoint.objId);
        if (frameMasksMap.size > 0) {
          masksMap.set(selectedPoint.frameIdx, frameMasksMap);
        } else {
          masksMap.delete(selectedPoint.frameIdx);
        }
        this.masks.set(masksMap);
        this.unmarkObjectAsLiveEdited(selectedPoint.frameIdx, selectedPoint.objId);
      }
      this.selectedPoint.set(null);
      this.drawCurrentFrame();
    } catch (error) {
      console.error(error);
      this.showToast(
        'error',
        'Remove point failed',
        this.getErrorMessage(error, 'Unable to remove selected point.'),
      );
    } finally {
      this.isPointRequestInFlight.set(false);
    }
  }

  private replaceObjectPoints(frameIdx: number, objId: number, points: Point[]): void {
    const pointsMap = new Map(this.points());
    const framePointsMap = new Map(pointsMap.get(frameIdx) || new Map<number, Point[]>());
    if (points.length > 0) {
      framePointsMap.set(objId, points);
    } else {
      framePointsMap.delete(objId);
    }
    if (framePointsMap.size > 0) {
      pointsMap.set(frameIdx, framePointsMap);
    } else {
      pointsMap.delete(frameIdx);
    }
    this.points.set(pointsMap);
  }

  onScrubberFrameChange(value: number | string) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return;
    }
    const maxFrame = this.numFrames() - 1;
    this.targetFrameIdx.set(this.frameRendererService.clampFrameIndex(parsed, maxFrame));
  }

  setRangeStartToPlayhead(): void {
    this.rangeStartFrameIdx.set(
      this.frameRendererService.clampFrameIndex(this.targetFrameIdx(), this.numFrames() - 1),
    );
  }

  setRangeEndToPlayhead(): void {
    this.rangeEndFrameIdx.set(
      this.frameRendererService.clampFrameIndex(this.targetFrameIdx(), this.numFrames() - 1),
    );
  }

  clearFrameRange(): void {
    this.rangeStartFrameIdx.set(null);
    this.rangeEndFrameIdx.set(null);
  }

  addObject() {
    this.commandsService.addObject({
      backend: this.backend,
      getObjects: () => this.objects(),
      setObjects: (objects) => this.objects.set(objects as any),
      getSelectedObjectId: () => this.selectedObjectId(),
      setSelectedObjectId: (value) => this.selectedObjectId.set(value),
      getMasksMap: () => this.masks(),
      setMasksMap: (map) => this.masks.set(map),
      getPointsMap: () => this.points(),
      setPointsMap: (map) => this.points.set(map),
      getTrackedPoints: () => this.trackedPoints(),
      setTrackedPoints: (series) => this.trackedPoints.set(series as any),
      getLiveEditedObjectFrames: () => this.liveEditedObjectFrames(),
      setLiveEditedObjectFrames: (map) => this.liveEditedObjectFrames.set(map),
      updateStateEpoch: (epoch, source) => this.updateStateEpoch(epoch, source),
      refreshManifestMasks: () => this.refreshManifestMasks(),
      randomColor: () => this.getRandomColor(),
      redraw: () => this.drawCurrentFrame(),
      showError: (title, fallbackMessage, error) => {
        console.error(error);
        this.showToast('error', title, this.getErrorMessage(error, fallbackMessage));
      },
    });
  }

  renameObject(id: number, newName: string) {
    this.commandsService.renameObject(id, newName, {
      backend: this.backend,
      getObjects: () => this.objects(),
      setObjects: (objects) => this.objects.set(objects as any),
      getSelectedObjectId: () => this.selectedObjectId(),
      setSelectedObjectId: (value) => this.selectedObjectId.set(value),
      getMasksMap: () => this.masks(),
      setMasksMap: (map) => this.masks.set(map),
      getPointsMap: () => this.points(),
      setPointsMap: (map) => this.points.set(map),
      getTrackedPoints: () => this.trackedPoints(),
      setTrackedPoints: (series) => this.trackedPoints.set(series as any),
      getLiveEditedObjectFrames: () => this.liveEditedObjectFrames(),
      setLiveEditedObjectFrames: (map) => this.liveEditedObjectFrames.set(map),
      updateStateEpoch: (epoch, source) => this.updateStateEpoch(epoch, source),
      refreshManifestMasks: () => this.refreshManifestMasks(),
      randomColor: () => this.getRandomColor(),
      redraw: () => this.drawCurrentFrame(),
      showError: (title, fallbackMessage, error) => {
        console.error(error);
        this.showToast('error', title, this.getErrorMessage(error, fallbackMessage));
      },
    });
  }

  removeObject() {
    this.commandsService.removeObject({
      backend: this.backend,
      getObjects: () => this.objects(),
      setObjects: (objects) => this.objects.set(objects as any),
      getSelectedObjectId: () => this.selectedObjectId(),
      setSelectedObjectId: (value) => this.selectedObjectId.set(value),
      getMasksMap: () => this.masks(),
      setMasksMap: (map) => this.masks.set(map),
      getPointsMap: () => this.points(),
      setPointsMap: (map) => this.points.set(map),
      getTrackedPoints: () => this.trackedPoints(),
      setTrackedPoints: (series) => this.trackedPoints.set(series as any),
      getLiveEditedObjectFrames: () => this.liveEditedObjectFrames(),
      setLiveEditedObjectFrames: (map) => this.liveEditedObjectFrames.set(map),
      updateStateEpoch: (epoch, source) => this.updateStateEpoch(epoch, source),
      refreshManifestMasks: () => this.refreshManifestMasks(),
      randomColor: () => this.getRandomColor(),
      redraw: () => this.drawCurrentFrame(),
      showError: (title, fallbackMessage, error) => {
        console.error(error);
        this.showToast('error', title, this.getErrorMessage(error, fallbackMessage));
      },
    });
  }

  async removeAllObjects() {
    await this.commandsService.removeAllObjects({
      backend: this.backend,
      getObjects: () => this.objects(),
      setObjects: (objects) => this.objects.set(objects as any),
      getSelectedObjectId: () => this.selectedObjectId(),
      setSelectedObjectId: (value) => this.selectedObjectId.set(value),
      getMasksMap: () => this.masks(),
      setMasksMap: (map) => this.masks.set(map),
      getPointsMap: () => this.points(),
      setPointsMap: (map) => this.points.set(map),
      getTrackedPoints: () => this.trackedPoints(),
      setTrackedPoints: (series) => this.trackedPoints.set(series as any),
      getLiveEditedObjectFrames: () => this.liveEditedObjectFrames(),
      setLiveEditedObjectFrames: (map) => this.liveEditedObjectFrames.set(map),
      updateStateEpoch: (epoch, source) => this.updateStateEpoch(epoch, source),
      refreshManifestMasks: () => this.refreshManifestMasks(),
      randomColor: () => this.getRandomColor(),
      redraw: () => this.drawCurrentFrame(),
      showError: (title, fallbackMessage, error) => {
        console.error(error);
        this.showToast('error', title, this.getErrorMessage(error, fallbackMessage));
      },
    });
  }

  async propagate() {
    await this.workflowService.propagate({
      backend: this.backend,
      runBackendJob: (title, startJob) => this.runBackendJob(title, startJob),
      updateStateEpoch: (epoch, source) => this.updateStateEpoch(epoch, source),
      setHasManifestMasks: (value) => this.hasManifestMasks.set(value),
      invalidateMaskCache: () => this.maskOverlayCache.clearAll(),
      scheduleFrameLoad: (frameIdx) => this.scheduleFrameLoad(frameIdx),
      targetFrameIdx: () => this.targetFrameIdx(),
      selectedFrameRange: () => this.selectedFrameRangeRequest(),
      setTrackedPoints: (next) => this.trackedPoints.set(next as any),
      drawCurrentFrame: () => this.drawCurrentFrame(),
      showToast: (severity, title, message) => this.showToast(severity, title, message),
      getErrorMessage: (error, fallback) => this.getErrorMessage(error, fallback),
      getLastCompletedJobId: () => this.lastCompletedJobId,
      trackingUseSupportGrid: () => this.trackingUseSupportGrid(),
      setIsLoading: (value) => this.isLoading.set(value),
      getSaveName: () => this.saveName(),
      setSaveName: (value) => this.saveName.set(value),
      buildInteractiveStateSnapshot: () => this.buildInteractiveStateSnapshot(),
      resetInteractiveMaps: () => {
        this.masks.set(new Map());
        this.points.set(new Map());
        this.liveEditedObjectFrames.set(new Map());
      },
    });
  }

  private applyTrackingResult(result: TrackPromptPointsResult): void {
    const trackedSeries: TrackedPointSeries[] = result.points.map((point, index) => ({
      ...point,
      tracks: result.tracks[index] || [],
      visibility: result.visibility[index] || [],
    }));
    this.trackedPoints.set(trackedSeries);
    this.drawCurrentFrame();
  }

  private async loadTrackingResult(resultId: string, sourceLabel: string): Promise<boolean> {
    try {
      const response = await firstValueFrom(this.backend.getTrackingResult(resultId));
      this.applyTrackingResult(response.result);
      return true;
    } catch (error: any) {
      console.error(error);
      this.showToast(
        'warning',
        'Tracking result unavailable',
        this.getErrorMessage(error, `${sourceLabel} result could not be loaded.`),
      );
      return false;
    }
  }

  async runTracking() {
    await this.workflowService.runTracking({
      backend: this.backend,
      runBackendJob: (title, startJob) => this.runBackendJob(title, startJob),
      updateStateEpoch: (epoch, source) => this.updateStateEpoch(epoch, source),
      setHasManifestMasks: (value) => this.hasManifestMasks.set(value),
      invalidateMaskCache: () => this.maskOverlayCache.clearAll(),
      scheduleFrameLoad: (frameIdx) => this.scheduleFrameLoad(frameIdx),
      targetFrameIdx: () => this.targetFrameIdx(),
      selectedFrameRange: () => this.selectedFrameRangeRequest(),
      setTrackedPoints: (next) => this.trackedPoints.set(next as any),
      drawCurrentFrame: () => this.drawCurrentFrame(),
      showToast: (severity, title, message) => this.showToast(severity, title, message),
      getErrorMessage: (error, fallback) => this.getErrorMessage(error, fallback),
      getLastCompletedJobId: () => this.lastCompletedJobId,
      trackingUseSupportGrid: () => this.trackingUseSupportGrid(),
      setIsLoading: (value) => this.isLoading.set(value),
      getSaveName: () => this.saveName(),
      setSaveName: (value) => this.saveName.set(value),
      buildInteractiveStateSnapshot: () => this.buildInteractiveStateSnapshot(),
      resetInteractiveMaps: () => {
        this.masks.set(new Map());
        this.points.set(new Map());
        this.liveEditedObjectFrames.set(new Map());
      },
    });
  }

  clearMasks() {
    this.workflowService.clearMasks({
      backend: this.backend,
      runBackendJob: (title, startJob) => this.runBackendJob(title, startJob),
      updateStateEpoch: (epoch, source) => this.updateStateEpoch(epoch, source),
      setHasManifestMasks: (value) => this.hasManifestMasks.set(value),
      invalidateMaskCache: () => this.maskOverlayCache.clearAll(),
      scheduleFrameLoad: (frameIdx) => this.scheduleFrameLoad(frameIdx),
      targetFrameIdx: () => this.targetFrameIdx(),
      setTrackedPoints: (next) => this.trackedPoints.set(next as any),
      drawCurrentFrame: () => this.drawCurrentFrame(),
      showToast: (severity, title, message) => this.showToast(severity, title, message),
      getErrorMessage: (error, fallback) => this.getErrorMessage(error, fallback),
      getLastCompletedJobId: () => this.lastCompletedJobId,
      trackingUseSupportGrid: () => this.trackingUseSupportGrid(),
      setIsLoading: (value) => this.isLoading.set(value),
      getSaveName: () => this.saveName(),
      setSaveName: (value) => this.saveName.set(value),
      buildInteractiveStateSnapshot: () => this.buildInteractiveStateSnapshot(),
      resetInteractiveMaps: () => {
        this.masks.set(new Map());
        this.points.set(new Map());
        this.liveEditedObjectFrames.set(new Map());
      },
    });
  }

  save() {
    this.workflowService.save({
      backend: this.backend,
      runBackendJob: (title, startJob) => this.runBackendJob(title, startJob),
      updateStateEpoch: (epoch, source) => this.updateStateEpoch(epoch, source),
      setHasManifestMasks: (value) => this.hasManifestMasks.set(value),
      invalidateMaskCache: () => this.maskOverlayCache.clearAll(),
      scheduleFrameLoad: (frameIdx) => this.scheduleFrameLoad(frameIdx),
      targetFrameIdx: () => this.targetFrameIdx(),
      setTrackedPoints: (next) => this.trackedPoints.set(next as any),
      drawCurrentFrame: () => this.drawCurrentFrame(),
      showToast: (severity, title, message) => this.showToast(severity, title, message),
      getErrorMessage: (error, fallback) => this.getErrorMessage(error, fallback),
      getLastCompletedJobId: () => this.lastCompletedJobId,
      trackingUseSupportGrid: () => this.trackingUseSupportGrid(),
      setIsLoading: (value) => this.isLoading.set(value),
      getSaveName: () => this.saveName(),
      setSaveName: (value) => this.saveName.set(value),
      buildInteractiveStateSnapshot: () => this.buildInteractiveStateSnapshot(),
      resetInteractiveMaps: () => {
        this.masks.set(new Map());
        this.points.set(new Map());
        this.liveEditedObjectFrames.set(new Map());
      },
    });
  }

  getRandomColor() {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  }

}
