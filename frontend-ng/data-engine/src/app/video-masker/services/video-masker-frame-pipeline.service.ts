import { ElementRef, Injectable } from '@angular/core';
import { VideoMaskObjectData } from '../../services/backend.service';
import { FrameRendererService } from './frame-renderer.service';

export interface FramePipelineState {
  frameLoadToken: number;
  pendingFrameIdx: number | null;
  frameLoadAnimationId: number | null;
  frameImageCache: Map<number, HTMLImageElement>;
  maskDataCache: Map<number, { [objId: string]: VideoMaskObjectData }>;
  maxFrameCacheSize: number;
  currentBaseImage: HTMLImageElement | null;
  currentMaskObjects: { [objId: string]: VideoMaskObjectData };
}

export interface FramePipelineDeps {
  canvasRef?: ElementRef<HTMLCanvasElement>;
  getVideoFrameUrl: (frameIdx: number) => string;
  getVideoMaskData: (frameIdx: number) => Promise<{ objects?: { [objId: string]: VideoMaskObjectData }; error?: unknown }>;
  hasManifestMasks: () => boolean;
  numFrames: () => number;
  draw: (image: HTMLImageElement, frameIdx: number) => void;
  onDisplayedFrame: (frameIdx: number) => void;
  setIsFrameLoading: (loading: boolean) => void;
}

@Injectable({ providedIn: 'root' })
export class VideoMaskerFramePipelineService {
  constructor(private frameRendererService: FrameRendererService) {}

  clearFrameCaches(state: FramePipelineState): void {
    state.frameImageCache.clear();
    state.maskDataCache.clear();
    state.currentBaseImage = null;
    state.currentMaskObjects = {};
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
    state.currentMaskObjects = {};
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

    await this.loadMaskDataForFrame(frameIdx, token, state, deps);
    if (token !== state.frameLoadToken) {
      return;
    }
    deps.draw(image, frameIdx);
  }

  private cacheFrameImage(frameIdx: number, image: HTMLImageElement, state: FramePipelineState): void {
    if (state.frameImageCache.has(frameIdx)) {
      state.frameImageCache.delete(frameIdx);
    }
    state.frameImageCache.set(frameIdx, image);
    this.frameRendererService.evictWithLimit(state.frameImageCache, state.maxFrameCacheSize, (oldestKey) => {
      state.maskDataCache.delete(oldestKey);
    });
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

  private async loadMaskDataForFrame(
    frameIdx: number,
    token: number,
    state: FramePipelineState,
    deps: FramePipelineDeps,
  ): Promise<void> {
    if (!deps.hasManifestMasks()) {
      state.currentMaskObjects = {};
      return;
    }
    const cachedMaskData = state.maskDataCache.get(frameIdx);
    if (cachedMaskData) {
      state.currentMaskObjects = cachedMaskData;
      return;
    }

    try {
      const response = await deps.getVideoMaskData(frameIdx);
      if (token !== state.frameLoadToken) {
        return;
      }
      if ((response as any)?.error) {
        state.currentMaskObjects = {};
        return;
      }
      state.currentMaskObjects = response.objects || {};
      state.maskDataCache.set(frameIdx, state.currentMaskObjects);
    } catch (error) {
      console.error(error);
      state.currentMaskObjects = {};
    }
  }
}
