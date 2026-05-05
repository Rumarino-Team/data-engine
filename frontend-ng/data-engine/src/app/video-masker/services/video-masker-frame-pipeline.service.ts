import { ElementRef, Injectable } from '@angular/core';
import { FrameRendererService } from './frame-renderer.service';
import { MaskOverlayCacheService } from './mask-overlay-cache.service';
import { MaskObject } from './video-masker-state.store';

export interface FramePipelineState {
  frameLoadToken: number;
  pendingFrameIdx: number | null;
  frameLoadAnimationId: number | null;
  frameImageCache: Map<number, HTMLImageElement>;
  maxFrameCacheSize: number;
  currentBaseImage: HTMLImageElement | null;
  previousFrameIdx: number | null;
}

export interface FramePipelineDeps {
  canvasRef?: ElementRef<HTMLCanvasElement>;
  getVideoFrameUrl: (frameIdx: number) => string;
  hasManifestMasks: () => boolean;
  numFrames: () => number;
  objects: () => MaskObject[];
  liveEditedObjectIdsForFrame: (frameIdx: number) => Set<number>;
  draw: (image: HTMLImageElement, frameIdx: number) => void;
  onDisplayedFrame: (frameIdx: number) => void;
  setIsFrameLoading: (loading: boolean) => void;
}

@Injectable({ providedIn: 'root' })
export class VideoMaskerFramePipelineService {
  constructor(
    private frameRendererService: FrameRendererService,
    private maskOverlayCache: MaskOverlayCacheService,
  ) {}

  clearFrameCaches(state: FramePipelineState): void {
    state.frameImageCache.clear();
    state.currentBaseImage = null;
    state.previousFrameIdx = null;
    state.frameLoadToken++;
    if (state.frameLoadAnimationId !== null) {
      cancelAnimationFrame(state.frameLoadAnimationId);
      state.frameLoadAnimationId = null;
    }
  }

  scheduleFrameLoad(frameIdx: number, state: FramePipelineState, deps: FramePipelineDeps): void {
    state.pendingFrameIdx = frameIdx;
    if (state.frameLoadAnimationId !== null) {
      return;
    }
    state.frameLoadAnimationId = requestAnimationFrame(() => {
      state.frameLoadAnimationId = null;
      const nextFrameIdx = state.pendingFrameIdx;
      state.pendingFrameIdx = null;
      if (nextFrameIdx !== null) {
        this.loadFrame(nextFrameIdx, state, deps);
      }
    });
  }

  async loadFrame(frameIdx: number, state: FramePipelineState, deps: FramePipelineDeps): Promise<void> {
    if (!deps.canvasRef?.nativeElement) {
      return;
    }

    const token = ++state.frameLoadToken;
    const cachedImage = state.frameImageCache.get(frameIdx);
    if (cachedImage?.complete) {
      await this.paintLoadedFrame(cachedImage, frameIdx, token, state, deps);
      return;
    }

    deps.setIsFrameLoading(true);
    const image = new Image();
    const frameUrl = deps.getVideoFrameUrl(frameIdx);

    image.onerror = () => {
      if (token !== state.frameLoadToken) {
        return;
      }
      console.error(`Failed to load frame image: ${frameUrl}`);
      deps.setIsFrameLoading(false);
    };

    image.onload = async () => {
      if (token !== state.frameLoadToken) {
        return;
      }
      this.cacheFrameImage(frameIdx, image, state);
      await this.paintLoadedFrame(image, frameIdx, token, state, deps);
    };

    image.src = frameUrl;
  }

  drawCurrentFrame(state: FramePipelineState, deps: FramePipelineDeps): void {
    if (!state.currentBaseImage || !deps.canvasRef?.nativeElement) {
      return;
    }
    const frameIdx = deps.canvasRef.nativeElement.dataset['frameIdx']
      ? Number(deps.canvasRef.nativeElement.dataset['frameIdx'])
      : undefined;
    if (frameIdx === undefined || Number.isNaN(frameIdx) || frameIdx < 0) {
      return;
    }
    deps.draw(state.currentBaseImage, frameIdx);
  }

  private async paintLoadedFrame(
    image: HTMLImageElement,
    frameIdx: number,
    token: number,
    state: FramePipelineState,
    deps: FramePipelineDeps,
  ): Promise<void> {
    state.currentBaseImage = image;
    this.ensureCanvasSize(image.width, image.height, deps.canvasRef);
    if (deps.canvasRef?.nativeElement) {
      deps.canvasRef.nativeElement.dataset['frameIdx'] = String(frameIdx);
    }
    deps.draw(image, frameIdx);
    deps.onDisplayedFrame(frameIdx);
    deps.setIsFrameLoading(false);
    this.preloadNeighborFrames(frameIdx, state, deps);
    this.prefetchMaskOverlay(image, frameIdx, token, state, deps);
    state.previousFrameIdx = frameIdx;
  }

  private cacheFrameImage(frameIdx: number, image: HTMLImageElement, state: FramePipelineState): void {
    if (state.frameImageCache.has(frameIdx)) {
      state.frameImageCache.delete(frameIdx);
    }
    state.frameImageCache.set(frameIdx, image);
    this.frameRendererService.evictWithLimit(state.frameImageCache, state.maxFrameCacheSize);
  }

  private preloadNeighborFrames(frameIdx: number, state: FramePipelineState, deps: FramePipelineDeps): void {
    for (const neighborIdx of [frameIdx + 1, frameIdx - 1]) {
      if (neighborIdx < 0 || neighborIdx >= deps.numFrames() || state.frameImageCache.has(neighborIdx)) {
        continue;
      }
      const image = new Image();
      image.onload = () => this.cacheFrameImage(neighborIdx, image, state);
      image.src = deps.getVideoFrameUrl(neighborIdx);
    }
  }

  private ensureCanvasSize(width: number, height: number, canvasRef?: ElementRef<HTMLCanvasElement>): void {
    if (!canvasRef?.nativeElement) {
      return;
    }
    const canvas = canvasRef.nativeElement;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  private prefetchMaskOverlay(
    image: HTMLImageElement,
    frameIdx: number,
    token: number,
    state: FramePipelineState,
    deps: FramePipelineDeps,
  ): void {
    const canvas = deps.canvasRef?.nativeElement;
    if (!canvas || !deps.hasManifestMasks() || deps.objects().length === 0) {
      return;
    }
    void this.maskOverlayCache.prefetchAroundFrame({
      frameIdx,
      previousFrameIdx: state.previousFrameIdx,
      numFrames: deps.numFrames(),
      objects: deps.objects(),
      hasManifestMasks: deps.hasManifestMasks(),
      liveEditedObjectIds: deps.liveEditedObjectIdsForFrame(frameIdx),
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
      onOverlayReady: (readyFrameIdx) => {
        if (
          readyFrameIdx !== frameIdx ||
          token !== state.frameLoadToken ||
          state.currentBaseImage !== image
        ) {
          return;
        }
        deps.draw(image, frameIdx);
      }
    });
  }
}
