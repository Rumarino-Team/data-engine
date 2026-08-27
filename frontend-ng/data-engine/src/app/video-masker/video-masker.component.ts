import {
  AfterViewInit,
  Component,
  ElementRef,
  OnDestroy,
  ViewChild,
  effect,
  inject,
  isDevMode,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { JobStatusPanelComponent } from './components/job-status-panel/job-status-panel.component';
import { ToastStackComponent } from './components/toast-stack/toast-stack.component';
import { VideoMaskerStateStore } from './services/video-masker-state.store';
import { FrameCanvasService } from './services/frame-canvas.service';
import { VideoMaskerActionsService } from './services/video-masker-actions.service';
import { LoadSourceMode } from './state/video-masker-ui.types';
import { browseLabel, clampFrameIndex, loadModeHint, loadPathPlaceholder } from './video-masker.util';

/**
 * Composition root for the video masker route. Owns the view refs and lifecycle, wires
 * two signal effects, and exposes the shared {@link VideoMaskerStateStore} plus the
 * {@link VideoMaskerActionsService} / {@link FrameCanvasService} to the template. All
 * non-trivial logic lives in those collaborators.
 */
@Component({
  selector: 'app-video-masker',
  standalone: true,
  imports: [CommonModule, FormsModule, ToastStackComponent, JobStatusPanelComponent],
  templateUrl: './video-masker.component.html',
  styleUrls: ['./video-masker.component.css'],
  providers: [VideoMaskerStateStore, FrameCanvasService, VideoMaskerActionsService],
})
export class VideoMaskerComponent implements AfterViewInit, OnDestroy {
  @ViewChild('canvas') canvasRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('videoFileInput') videoFileInputRef?: ElementRef<HTMLInputElement>;
  @ViewChild('framesDirInput') framesDirInputRef?: ElementRef<HTMLInputElement>;

  readonly store = inject(VideoMaskerStateStore);
  readonly actions = inject(VideoMaskerActionsService);
  readonly frameCanvas = inject(FrameCanvasService);
  readonly showDebugUi = isDevMode();

  private healthTimerId: ReturnType<typeof setInterval> | null = null;

  constructor() {
    this.actions.initApiUrlFromBackend();

    effect(() => {
      if (this.store.isInitialized()) {
        this.frameCanvas.scheduleFrameLoad(this.store.targetFrameIdx());
      }
    });

    effect(() => {
      this.store.trackingOverlayStyle();
      this.store.trackedPoints();
      this.frameCanvas.redraw();
    });
  }

  ngAfterViewInit(): void {
    if (this.canvasRef?.nativeElement) {
      this.frameCanvas.attachCanvas(this.canvasRef.nativeElement);
    }
    void this.actions.checkApiHealth(true);
    this.healthTimerId = setInterval(() => void this.actions.checkApiHealth(), 3000);
  }

  ngOnDestroy(): void {
    if (this.healthTimerId !== null) {
      clearInterval(this.healthTimerId);
    }
    this.frameCanvas.detach();
  }

  // --- load-source copy (pure) -------------------------------------------

  getLoadPathPlaceholder(): string {
    return loadPathPlaceholder(this.store.loadSourceMode());
  }

  getBrowseLabel(): string {
    return browseLabel(this.store.loadSourceMode());
  }

  getLoadModeHint(): string {
    return loadModeHint(this.store.loadSourceMode());
  }

  // --- template input bindings -----------------------------------------

  onLoadSourceModeChange(value: string): void {
    const nextMode: LoadSourceMode =
      value === 'video_file' || value === 'saved_session_dir' ? value : 'frames_dir';
    this.store.loadSourceMode.set(nextMode);
  }

  onVideoDirChange(value: string): void {
    this.store.videoDir.set(value);
  }

  onApiUrlChange(value: string): void {
    this.store.apiUrlInput.set(value);
  }

  onSaveNameChange(value: string): void {
    this.store.saveName.set(value);
  }

  // --- API URL (delegates) --------------------------------------------------

  applyApiUrl(): void {
    this.actions.applyApiUrl();
  }

  resetApiUrl(): void {
    this.actions.resetApiUrl();
  }

  isApiUrlDirty(): boolean {
    return this.actions.isApiUrlDirty();
  }

  // --- source picking ---------------------------------------------------

  openVideoFilePicker(): void {
    this.videoFileInputRef?.nativeElement.click();
  }

  openFramesDirPicker(): void {
    this.framesDirInputRef?.nativeElement.click();
  }

  async browseSelectedSource(): Promise<void> {
    const selectedPath = await this.actions.pickNativePath(this.store.loadSourceMode());
    if (selectedPath) {
      this.store.videoDir.set(selectedPath);
      return;
    }
    if (this.store.loadSourceMode() === 'video_file') {
      this.openVideoFilePicker();
      return;
    }
    this.openFramesDirPicker();
  }

  async browseVideo(): Promise<void> {
    this.store.loadSourceMode.set('video_file');
    await this.browseSelectedSource();
  }

  async browseFramesDirectory(): Promise<void> {
    this.store.loadSourceMode.set('frames_dir');
    await this.browseSelectedSource();
  }

  onVideoFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    const nativePath = this.getNativeFilePath(file);
    if (nativePath) {
      this.store.videoDir.set(nativePath);
    } else {
      this.store.videoDir.set(file.name);
      this.showPathUnavailableMessage('video');
    }

    input.value = '';
  }

  onFramesDirSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    const nativePath = this.getNativeFilePath(file);
    if (nativePath) {
      this.store.videoDir.set(this.getParentDirectory(nativePath));
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

  private showPathUnavailableMessage(target: 'video' | 'directory'): void {
    if (target === 'video') {
      this.actions.pushToast(
        'warning',
        'Path unavailable',
        'Selected video file name is available, but this browser does not expose the full local path. Paste the full video path manually.',
      );
      return;
    }
    if (this.store.loadSourceMode() === 'saved_session_dir') {
      this.actions.pushToast(
        'warning',
        'Path unavailable',
        'Selected folder contents are available, but this browser does not expose the full local directory path. Paste the full saved session directory path manually.',
      );
      return;
    }
    this.actions.pushToast(
      'warning',
      'Path unavailable',
      'Selected folder contents are available, but this browser does not expose the full local directory path. Paste the full frames directory path manually.',
    );
  }

  // --- session / workflow (delegates) -----------------------------------

  initVideo(): Promise<boolean> {
    return this.actions.initVideo();
  }

  propagate(): Promise<void> {
    return this.actions.propagate();
  }

  runTracking(): Promise<void> {
    return this.actions.runTracking();
  }

  clearMasks(): void {
    this.actions.clearMasks();
  }

  save(): void {
    this.actions.save();
  }

  addObject(): void {
    this.actions.addObject();
  }

  removeObject(): void {
    this.actions.removeObject();
  }

  removeAllObjects(): Promise<void> {
    return this.actions.removeAllObjects();
  }

  dismissToast(id: number): void {
    this.actions.dismissToast(id);
  }

  // --- canvas / scrubber interaction ----------------------------------

  onCanvasClick(event: MouseEvent): void {
    if (
      !this.store.isInitialized() ||
      this.store.selectedObjectId() === null ||
      this.store.isFrameLoading() ||
      this.store.isPointRequestInFlight() ||
      !this.frameCanvas.currentBaseImage
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
    const label = this.store.interactionMode() === 'positive' ? 1 : 0;
    const frameIdx = this.store.displayedFrameIdx();
    if (frameIdx < 0) {
      return;
    }
    void this.addPoint(x, y, label, frameIdx);
  }

  addPoint(x: number, y: number, label: number, frameIdx: number): Promise<void> {
    return this.actions.addPoint(x, y, label, frameIdx);
  }

  onScrubberFrameChange(value: number | string): void {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return;
    }
    const maxFrame = this.store.numFrames() - 1;
    this.store.targetFrameIdx.set(clampFrameIndex(parsed, maxFrame));
  }
}
