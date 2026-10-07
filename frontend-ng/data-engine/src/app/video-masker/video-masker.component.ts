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
import { FramePipelineService } from './services/frame-pipeline.service';
import { CanvasViewportService } from './services/canvas-viewport.service';
import { ObjectSidebarComponent } from './components/object-sidebar/object-sidebar.component';
import { ObjectOperationsService } from './services/object-operations.service';
import { VideoMaskerActionsService } from './services/video-masker-actions.service';
import { LoadSourceMode } from './state/video-masker-ui.types';
import {
  browseLabel,
  clampFrameIndex,
  loadModeHint,
  loadPathPlaceholder,
} from './video-masker.util';

/**
 * Composition root for the video masker route. Owns the view refs and lifecycle, wires
 * two signal effects, and exposes the shared {@link VideoMaskerStateStore} plus the
 * {@link VideoMaskerActionsService} / {@link FramePipelineService} to the template. All
 * non-trivial logic lives in those collaborators.
 */
@Component({
  selector: 'app-video-masker',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ToastStackComponent,
    JobStatusPanelComponent,
    ObjectSidebarComponent,
  ],
  templateUrl: './video-masker.component.html',
  styleUrls: ['./video-masker.component.css'],
  providers: [
    VideoMaskerStateStore,
    FramePipelineService,
    CanvasViewportService,
    VideoMaskerActionsService,
    ObjectOperationsService,
  ],
})
export class VideoMaskerComponent implements AfterViewInit, OnDestroy {
  @ViewChild('canvasHost') canvasHostRef?: ElementRef<HTMLDivElement>;
  @ViewChild('videoFileInput') videoFileInputRef?: ElementRef<HTMLInputElement>;
  @ViewChild('framesDirInput') framesDirInputRef?: ElementRef<HTMLInputElement>;

  readonly store = inject(VideoMaskerStateStore);
  readonly actions = inject(VideoMaskerActionsService);
  readonly framePipeline = inject(FramePipelineService);
  readonly viewport = inject(CanvasViewportService);
  readonly showDebugUi = isDevMode();
  apiUrlInputHasFocus = false;

  private healthTimerId: ReturnType<typeof setInterval> | null = null;

  constructor() {
    this.actions.initApiUrlFromBackend();

    effect(() => {
      if (this.store.isInitialized()) {
        this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
      }
    });

    effect(() => {
      this.store.trackingOverlayStyle();
      this.store.trackedPoints();
      this.framePipeline.redraw();
    });
  }

  ngAfterViewInit(): void {
    if (this.canvasHostRef?.nativeElement) {
      this.framePipeline.attach(this.canvasHostRef.nativeElement, (point) =>
        this.onCanvasPoint(point),
      );
    }
    void this.actions.checkApiHealth(true);
    this.healthTimerId = setInterval(() => void this.actions.checkApiHealth(), 3000);
  }

  ngOnDestroy(): void {
    if (this.healthTimerId !== null) {
      clearInterval(this.healthTimerId);
    }
    this.framePipeline.detach();
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

  onVideoPathBlur(event: FocusEvent): void {
    const input = event.target as HTMLInputElement | null;
    if (!input) {
      return;
    }
    requestAnimationFrame(() => {
      input.scrollLeft = input.scrollWidth;
    });
  }

  onApiUrlChange(value: string): void {
    this.store.apiUrlInput.set(value);
  }

  onApiUrlFocus(): void {
    this.apiUrlInputHasFocus = true;
    const apiUrl = this.store.apiUrlInput().trim();
    if (apiUrl && !/^[a-z][a-z\d+\-.]*:\/\//i.test(apiUrl)) {
      this.store.apiUrlInput.set(`http://${apiUrl}`);
    }
  }

  onApiUrlBlur(): void {
    this.apiUrlInputHasFocus = false;
  }

  getApiUrlInputValue(): string {
    const apiUrl = this.store.apiUrlInput();
    if (this.apiUrlInputHasFocus) {
      return apiUrl;
    }
    return apiUrl.replace(/^https?:\/\//i, '');
  }

  onSaveNameChange(value: string): void {
    this.store.saveName.set(value);
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

  // --- canvas / scrubber interaction ----------------------------------

  onCanvasPoint(point: { x: number; y: number }): void {
    if (
      !this.store.isInitialized() ||
      this.store.isLoading() ||
      this.store.selectedObjectId() === null ||
      this.store.isFrameLoading() ||
      this.store.isPointRequestInFlight() ||
      !this.framePipeline.currentBaseImage
    ) {
      return;
    }

    const label = this.store.interactionMode() === 'positive' ? 1 : 0;
    const frameIdx = this.store.displayedFrameIdx();
    if (frameIdx < 0) {
      return;
    }
    void this.actions.addPoint(point.x, point.y, label, frameIdx);
  }

  onScrubberFrameChange(value: number | string): void {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return;
    }
    const maxFrame = this.store.numFrames() - 1;
    this.store.targetFrameIdx.set(clampFrameIndex(parsed, maxFrame));
  }

  getPropagationFramePercent(frameIdx: number): number {
    const maxFrame = this.store.numFrames() - 1;
    if (maxFrame <= 0) {
      return 0;
    }
    return (clampFrameIndex(frameIdx, maxFrame) / maxFrame) * 100;
  }

  setPropagationStartFrame(): void {
    const frameIdx = this.store.displayedFrameIdx();
    if (frameIdx < 0) {
      return;
    }
    this.store.propagationStartFrameIdx.set(frameIdx);
    if (frameIdx > this.store.propagationEndFrameIdx()) {
      this.store.propagationEndFrameIdx.set(frameIdx);
    }
  }

  setPropagationEndFrame(): void {
    const frameIdx = this.store.displayedFrameIdx();
    if (frameIdx < 0) {
      return;
    }
    this.store.propagationEndFrameIdx.set(frameIdx);
    if (frameIdx < this.store.propagationStartFrameIdx()) {
      this.store.propagationStartFrameIdx.set(frameIdx);
    }
  }
}
