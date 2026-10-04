import { Injectable, inject, untracked } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { BackendService, VideoMaskObjectData } from '../../services/backend.service';
import { MaskOverlay } from './mask-texture';
import { CanvasViewportService } from './canvas-viewport.service';
import { DebugMaskSource } from '../state/video-masker-ui.types';
import { VideoMaskerStateStore } from './video-masker-state.store';
import {
  evictWithLimit,
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
 * Loads and caches frame data. The Konva viewport owns mask composition and
 * vector annotations; this service does not create canvases or draw pixels.
 */
@Injectable()
export class FramePipelineService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly viewport = inject(CanvasViewportService);

  private attached = false;
  private cacheGeneration = 0;
  private maskGeneration = 0;
  private readonly removedObjectIds = new Set<number>();
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

  attach(host: HTMLDivElement, onPoint: (point: { x: number; y: number }) => void): void {
    this.viewport.attach(host, onPoint);
    this.attached = true;
  }

  detach(): void {
    this.clearFrameCaches();
    this.viewport.detach();
    this.attached = false;
  }

  get currentBaseImage(): HTMLImageElement | null {
    return this.state.currentBaseImage;
  }

  clearMaskDataCache(): void {
    this.maskGeneration++;
    this.state.maskDataCache.clear();
    this.state.currentMaskObjects = {};
  }

  removeObject(objectId: number): void {
    // Saved manifests can still contain deleted objects, even after a refetch.
    this.removedObjectIds.add(objectId);
    this.clearMaskDataCache();
  }

  clearFrameCaches(): void {
    this.viewport.reset();
    this.cacheGeneration++;
    this.clearMaskDataCache();
    this.removedObjectIds.clear();
    this.state.frameImageCache.clear();
    this.state.currentBaseImage = null;
    this.state.currentMaskObjects = {};
    this.state.frameLoadToken++;
    this.state.pendingFrameIdx = null;
    this.store.isFrameLoading.set(false);
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
    if (!this.attached) {
      return;
    }

    const token = ++this.state.frameLoadToken;
    this.state.currentMaskObjects = {};
    const cachedImage = this.state.frameImageCache.get(frameIdx);
    if (cachedImage?.complete) {
      await this.displayLoadedFrame(cachedImage, frameIdx, token);
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
      await this.displayLoadedFrame(image, frameIdx, token);
    };

    image.src = frameUrl;
  }

  /** Repaint the frame that is currently displayed, if any. */
  redraw(): void {
    const frameIdx = this.store.displayedFrameIdx();
    if (frameIdx < 0 || !this.state.currentBaseImage || !this.attached) {
      return;
    }
    this.renderFrame(this.state.currentBaseImage, frameIdx);
  }

  private renderFrame(img: HTMLImageElement, frameIdx: number): void {
    const overlays: MaskOverlay[] = [];
    const liveFrameMasks = this.store.masks().get(frameIdx);
    const liveEditedObjectIds =
      this.store.liveEditedObjectFrames().get(frameIdx) ?? new Set<number>();

    if (this.store.hasManifestMasks() && Object.keys(this.state.currentMaskObjects).length > 0) {
      for (const [objIdStr, maskData] of Object.entries(this.state.currentMaskObjects)) {
        const objId = parseInt(objIdStr, 10);
        if (liveEditedObjectIds.has(objId) || this.removedObjectIds.has(objId)) {
          continue;
        }
        const obj = this.store.objects().find((candidate) => candidate.id === objId);
        overlays.push({ objectId: objId, source: maskData, color: obj?.color || '#ff9800' });
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
          overlays.push({ objectId: objId, source: mask, color: obj.color });
        }
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

    const points = Array.from(this.store.points().get(frameIdx)?.values() ?? []).flat();
    const tracks = this.store.trackedPoints();
    const objects = this.store.objects();
    const style = this.store.trackingOverlayStyle();
    untracked(() => this.viewport.render(img, overlays, points, tracks, objects, style, frameIdx));
  }

  private async displayLoadedFrame(
    image: HTMLImageElement,
    frameIdx: number,
    token: number,
  ): Promise<void> {
    this.state.currentBaseImage = image;
    this.renderFrame(image, frameIdx);
    this.store.displayedFrameIdx.set(frameIdx);
    this.store.isFrameLoading.set(false);
    this.preloadNeighborFrames(frameIdx);

    await this.loadMaskDataForFrame(frameIdx, token);
    if (token !== this.state.frameLoadToken) {
      return;
    }
    this.renderFrame(image, frameIdx);
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
    const generation = this.cacheGeneration;
    for (const neighborIdx of [frameIdx + 1, frameIdx - 1]) {
      if (
        neighborIdx < 0 ||
        neighborIdx >= this.store.numFrames() ||
        this.state.frameImageCache.has(neighborIdx)
      ) {
        continue;
      }
      const image = new Image();
      image.onload = () => {
        if (generation === this.cacheGeneration) {
          this.cacheFrameImage(neighborIdx, image);
        }
      };
      image.src = this.backend.getVideoFrameUrl(neighborIdx);
    }
  }

  private async loadMaskDataForFrame(frameIdx: number, token: number): Promise<void> {
    const generation = this.maskGeneration;
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
      if (token !== this.state.frameLoadToken || generation !== this.maskGeneration) {
        return;
      }
      if ((response as any)?.error) {
        this.state.currentMaskObjects = {};
        return;
      }
      this.state.currentMaskObjects = response.objects || {};
      this.state.maskDataCache.set(frameIdx, this.state.currentMaskObjects);
    } catch (error) {
      if (token !== this.state.frameLoadToken || generation !== this.maskGeneration) {
        return;
      }
      console.error(error);
      this.state.currentMaskObjects = {};
    }
  }
}
