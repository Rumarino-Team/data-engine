import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { BackendService, VideoMaskObjectData } from '../../services/backend.service';
import { DebugMaskSource } from '../state/video-masker-ui.types';
import { MaskObject, Point, TrackedPointSeries, VideoMaskerStateStore } from './video-masker-state.store';
import {
  evictWithLimit,
  hexToRgb,
  isObjectLiveEdited,
  maskHasForeground,
  normalizeMask2d,
} from '../video-masker.util';

interface FramePipelineState {
  frameLoadToken: number;
  pendingFrameIdx: number | null;
  frameLoadAnimationId: number | null;
  frameImageCache: Map<number, HTMLImageElement>;
  maskDataCache: Map<number, { [objId: string]: VideoMaskObjectData }>;
  maxFrameCacheSize: number;
  currentBaseImage: HTMLImageElement | null;
  currentMaskObjects: { [objId: string]: VideoMaskObjectData };
}

/**
 * Owns the <canvas> element plus the frame image/mask caches, and is the single place
 * that paints a frame (base image + manifest masks + live masks + prompt points +
 * tracking overlay). Reads everything else from {@link VideoMaskerStateStore}.
 */
@Injectable()
export class FrameCanvasService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);

  private canvas: HTMLCanvasElement | null = null;
  private readonly state: FramePipelineState = {
    frameLoadToken: 0,
    pendingFrameIdx: null,
    frameLoadAnimationId: null,
    frameImageCache: new Map<number, HTMLImageElement>(),
    maskDataCache: new Map<number, { [objId: string]: VideoMaskObjectData }>(),
    maxFrameCacheSize: 24,
    currentBaseImage: null,
    currentMaskObjects: {},
  };

  attachCanvas(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;
  }

  detach(): void {
    if (this.state.frameLoadAnimationId !== null) {
      cancelAnimationFrame(this.state.frameLoadAnimationId);
      this.state.frameLoadAnimationId = null;
    }
    this.canvas = null;
  }

  get currentBaseImage(): HTMLImageElement | null {
    return this.state.currentBaseImage;
  }

  clearMaskDataCache(): void {
    this.state.maskDataCache.clear();
  }

  clearFrameCaches(): void {
    this.state.frameImageCache.clear();
    this.state.maskDataCache.clear();
    this.state.currentBaseImage = null;
    this.state.currentMaskObjects = {};
    this.state.frameLoadToken++;
    if (this.state.frameLoadAnimationId !== null) {
      cancelAnimationFrame(this.state.frameLoadAnimationId);
      this.state.frameLoadAnimationId = null;
    }
  }

  scheduleFrameLoad(frameIdx: number): void {
    this.state.pendingFrameIdx = frameIdx;
    if (this.state.frameLoadAnimationId !== null) {
      return;
    }
    this.state.frameLoadAnimationId = requestAnimationFrame(() => {
      this.state.frameLoadAnimationId = null;
      const nextFrameIdx = this.state.pendingFrameIdx;
      this.state.pendingFrameIdx = null;
      if (nextFrameIdx !== null) {
        void this.loadFrame(nextFrameIdx);
      }
    });
  }

  async loadFrame(frameIdx: number): Promise<void> {
    if (!this.canvas) {
      return;
    }

    const token = ++this.state.frameLoadToken;
    this.state.currentMaskObjects = {};
    const cachedImage = this.state.frameImageCache.get(frameIdx);
    if (cachedImage?.complete) {
      await this.paintLoadedFrame(cachedImage, frameIdx, token);
      return;
    }

    this.store.isFrameLoading.set(true);
    const image = new Image();
    const frameUrl = this.backend.getVideoFrameUrl(frameIdx);

    image.onerror = () => {
      if (token !== this.state.frameLoadToken) {
        return;
      }
      console.error(`Failed to load frame image: ${frameUrl}`);
      this.store.isFrameLoading.set(false);
    };

    image.onload = async () => {
      if (token !== this.state.frameLoadToken) {
        return;
      }
      this.cacheFrameImage(frameIdx, image);
      await this.paintLoadedFrame(image, frameIdx, token);
    };

    image.src = frameUrl;
  }

  /** Repaint the frame that is currently displayed, if any. */
  redraw(): void {
    const frameIdx = this.store.displayedFrameIdx();
    if (frameIdx < 0 || !this.state.currentBaseImage || !this.canvas) {
      return;
    }
    this.draw(this.state.currentBaseImage, frameIdx);
  }

  draw(img: HTMLImageElement, frameIdx: number): void {
    const canvas = this.canvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const liveFrameMasks = this.store.masks().get(frameIdx);
    const liveEditedObjectIds = this.store.liveEditedObjectFrames().get(frameIdx) ?? new Set<number>();

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    if (this.store.hasManifestMasks() && Object.keys(this.state.currentMaskObjects).length > 0) {
      for (const [objIdStr, maskData] of Object.entries(this.state.currentMaskObjects)) {
        const objId = parseInt(objIdStr, 10);
        if (liveEditedObjectIds.has(objId)) {
          continue;
        }
        const obj = this.store.objects().find((candidate) => candidate.id === objId);
        this.drawMaskFromRle(ctx, maskData, obj?.color || '#ff9800');
      }
    }

    if (liveFrameMasks) {
      liveFrameMasks.forEach((mask, objId) => {
        const normalizedMask = normalizeMask2d(mask);
        if (!normalizedMask || !maskHasForeground(normalizedMask)) {
          return;
        }
        const obj = this.store.objects().find((candidate) => candidate.id === objId);
        if (obj) {
          this.drawMask(ctx, normalizedMask, obj.color);
        }
      });
    }

    const framePoints = this.store.points().get(frameIdx);
    if (framePoints) {
      framePoints.forEach((frameObjPoints) => {
        frameObjPoints.forEach((point) => this.drawPoint(ctx, point));
      });
    }

    const selectedObjectId = this.store.selectedObjectId();
    let maskSource: DebugMaskSource = 'none';
    if (selectedObjectId !== null) {
      const selectedLiveMask = liveFrameMasks?.get(selectedObjectId);
      const normalizedLiveMask = selectedLiveMask ? normalizeMask2d(selectedLiveMask) : null;
      const hasLiveMask = Boolean(normalizedLiveMask && maskHasForeground(normalizedLiveMask));
      if (hasLiveMask) {
        maskSource = 'live';
      } else if (
        !isObjectLiveEdited(this.store.liveEditedObjectFrames(), frameIdx, selectedObjectId) &&
        Boolean(this.state.currentMaskObjects[String(selectedObjectId)])
      ) {
        maskSource = 'manifest';
      }
    }
    this.store.lastMaskSource.set(maskSource);

    this.drawTrackingOverlay(
      ctx,
      frameIdx,
      this.store.trackingOverlayStyle(),
      this.store.trackedPoints(),
      this.store.objects(),
    );
  }

  private async paintLoadedFrame(
    image: HTMLImageElement,
    frameIdx: number,
    token: number,
  ): Promise<void> {
    this.state.currentBaseImage = image;
    this.ensureCanvasSize(image.width, image.height);
    if (this.canvas) {
      this.canvas.dataset['frameIdx'] = String(frameIdx);
    }
    this.draw(image, frameIdx);
    this.store.displayedFrameIdx.set(frameIdx);
    this.store.isFrameLoading.set(false);
    this.preloadNeighborFrames(frameIdx);

    await this.loadMaskDataForFrame(frameIdx, token);
    if (token !== this.state.frameLoadToken) {
      return;
    }
    this.draw(image, frameIdx);
  }

  private cacheFrameImage(frameIdx: number, image: HTMLImageElement): void {
    if (this.state.frameImageCache.has(frameIdx)) {
      this.state.frameImageCache.delete(frameIdx);
    }
    this.state.frameImageCache.set(frameIdx, image);
    evictWithLimit(this.state.frameImageCache, this.state.maxFrameCacheSize, (oldestKey) => {
      this.state.maskDataCache.delete(oldestKey);
    });
  }

  private preloadNeighborFrames(frameIdx: number): void {
    for (const neighborIdx of [frameIdx + 1, frameIdx - 1]) {
      if (
        neighborIdx < 0 ||
        neighborIdx >= this.store.numFrames() ||
        this.state.frameImageCache.has(neighborIdx)
      ) {
        continue;
      }
      const image = new Image();
      image.onload = () => this.cacheFrameImage(neighborIdx, image);
      image.src = this.backend.getVideoFrameUrl(neighborIdx);
    }
  }

  private ensureCanvasSize(width: number, height: number): void {
    if (!this.canvas) {
      return;
    }
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  private async loadMaskDataForFrame(frameIdx: number, token: number): Promise<void> {
    if (!this.store.hasManifestMasks()) {
      this.state.currentMaskObjects = {};
      return;
    }
    const cachedMaskData = this.state.maskDataCache.get(frameIdx);
    if (cachedMaskData) {
      this.state.currentMaskObjects = cachedMaskData;
      return;
    }

    try {
      const response = await firstValueFrom(this.backend.getVideoMaskData(frameIdx));
      if (token !== this.state.frameLoadToken) {
        return;
      }
      if ((response as any)?.error) {
        this.state.currentMaskObjects = {};
        return;
      }
      this.state.currentMaskObjects = response.objects || {};
      this.state.maskDataCache.set(frameIdx, this.state.currentMaskObjects);
    } catch (error) {
      console.error(error);
      this.state.currentMaskObjects = {};
    }
  }

  // --- primitive drawing helpers (from the former VideoMaskerRenderingService) ---

  private drawMask(ctx: CanvasRenderingContext2D, mask: boolean[][], color: string): void {
    const normalizedMask = normalizeMask2d(mask);
    if (!normalizedMask?.length || !normalizedMask[0]?.length) {
      return;
    }
    const width = normalizedMask[0].length;
    const height = normalizedMask.length;
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
    const [r, g, b] = hexToRgb(color);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (!normalizedMask[y][x]) {
          continue;
        }
        const index = (y * width + x) * 4;
        data[index] = r;
        data[index + 1] = g;
        data[index + 2] = b;
        data[index + 3] = 120;
      }
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
    ctx.drawImage(tempCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  private drawMaskFromRle(
    ctx: CanvasRenderingContext2D,
    maskData: VideoMaskObjectData,
    color: string,
  ): void {
    const size = maskData.size;
    if (!Array.isArray(size) || size.length !== 2) {
      return;
    }
    const height = Number(size[0]);
    const width = Number(size[1]);
    if (!Number.isFinite(height) || !Number.isFinite(width) || height <= 0 || width <= 0) {
      return;
    }

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
    const [r, g, b] = hexToRgb(color);

    for (const run of maskData.rle || []) {
      if (!Array.isArray(run) || run.length !== 2) {
        continue;
      }
      const start = Math.max(0, Number(run[0]) | 0);
      const length = Math.max(0, Number(run[1]) | 0);
      const end = Math.min(width * height, start + length);
      for (let index = start; index < end; index++) {
        const pixelOffset = index * 4;
        data[pixelOffset] = r;
        data[pixelOffset + 1] = g;
        data[pixelOffset + 2] = b;
        data[pixelOffset + 3] = 120;
      }
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
    ctx.drawImage(tempCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  private drawPoint(ctx: CanvasRenderingContext2D, point: Point): void {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = point.label === 1 ? '#00ff00' : '#ff0000';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  private drawTrackingOverlay(
    ctx: CanvasRenderingContext2D,
    frameIdx: number,
    style: 'point' | 'short' | 'full',
    trackedPoints: TrackedPointSeries[],
    objects: MaskObject[],
  ): void {
    if (!trackedPoints.length) {
      return;
    }

    for (const pointSeries of trackedPoints) {
      if (frameIdx < 0 || frameIdx >= pointSeries.tracks.length) {
        continue;
      }

      const obj = objects.find((candidate) => candidate.id === pointSeries.obj_id);
      const color = obj?.color || '#ffd54f';
      const visibleNow = pointSeries.visibility[frameIdx] !== false;

      if (style !== 'point') {
        const trailStart =
          style === 'short'
            ? Math.max(pointSeries.source_frame_idx, frameIdx - 20)
            : Math.max(pointSeries.source_frame_idx, 0);

        ctx.beginPath();
        let started = false;
        for (let idx = trailStart; idx <= frameIdx; idx++) {
          if (idx < 0 || idx >= pointSeries.tracks.length || pointSeries.visibility[idx] === false) {
            continue;
          }
          const [x, y] = pointSeries.tracks[idx];
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = style === 'short' ? 0.75 : 0.55;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      if (!visibleNow) {
        continue;
      }

      const [currentX, currentY] = pointSeries.tracks[frameIdx];
      ctx.beginPath();
      ctx.arc(currentX, currentY, 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}
