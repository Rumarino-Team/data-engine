import { Component, ElementRef, OnDestroy, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { BackendService } from '../services/backend.service';
import { DesktopBridgeService } from '../services/desktop-bridge.service';
import { JobStatusPanelComponent } from './components/job-status-panel/job-status-panel.component';
import { ToastStackComponent } from './components/toast-stack/toast-stack.component';
import { MaskStateService } from './services/mask-state.service';
import { ToastService } from './services/toast.service';
import { VideoJobsService } from './services/video-jobs.service';
import { VideoSessionService } from './services/video-session.service';
import { FrameRendererService } from './services/frame-renderer.service';
import { VideoMaskerFacade } from './video-masker.facade';
import { VideoMaskerSessionStateService } from './services/video-masker-session-state.service';
import { VideoMaskerRenderingService } from './services/video-masker-rendering.service';
import { VideoMaskerCommandsService } from './services/video-masker-commands.service';
import { VideoMaskerWorkflowService } from './services/video-masker-workflow.service';
import { VideoMaskerFramePipelineService } from './services/video-masker-frame-pipeline.service';

@Component({
  selector: 'app-video-masker',
  standalone: true,
  imports: [CommonModule, FormsModule, ToastStackComponent, JobStatusPanelComponent],
  templateUrl: './video-masker.component.html',
  styleUrls: ['./video-masker.component.css'],
})
export class VideoMaskerComponent extends VideoMaskerFacade implements OnDestroy {
  @ViewChild('canvas') declare canvasRef: ElementRef<HTMLCanvasElement>;
  @ViewChild('videoFileInput') declare videoFileInputRef?: ElementRef<HTMLInputElement>;
  @ViewChild('framesDirInput') declare framesDirInputRef?: ElementRef<HTMLInputElement>;

  constructor(
    backend: BackendService,
    desktopBridge: DesktopBridgeService,
    toastService: ToastService,
    jobsService: VideoJobsService,
    maskStateService: MaskStateService,
    videoSessionService: VideoSessionService,
    frameRendererService: FrameRendererService,
    sessionStateService: VideoMaskerSessionStateService,
    renderingService: VideoMaskerRenderingService,
    commandsService: VideoMaskerCommandsService,
    workflowService: VideoMaskerWorkflowService,
    framePipelineService: VideoMaskerFramePipelineService,
  ) {
    super(
      backend,
      desktopBridge,
      toastService,
      jobsService,
      maskStateService,
      videoSessionService,
      frameRendererService,
      sessionStateService,
      renderingService,
      commandsService,
      workflowService,
      framePipelineService,
    );
  }

  override ngOnDestroy(): void {
    super.ngOnDestroy();
  }
}
