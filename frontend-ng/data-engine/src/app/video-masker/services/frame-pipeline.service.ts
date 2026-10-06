import { Injectable, inject, untracked } from '@angular/core';
import { firstValueFrom, Subject, takeUntil } from 'rxjs';
import { BackendService, VideoMaskObjectData } from '../../services/backend.service';
import { MaskOverlay } from './mask-texture';
import { CanvasViewportService } from './canvas-viewport.service';
import { DebugMaskSource } from '../state/video-masker-ui.types';
import { VideoMaskerStateStore } from './video-masker-state.store';
import { evictWithLimit, isObjectLiveEdited, maskHasForeground } from '../video-masker.util';

interface FramePipelineState {
  frameLoadToken: number;
  pendingFrameIdx: number | null;
  frameLoadAnimationId: number | null;
  frameImageCache: Map<number, HTMLImageElement>;
  maskDataCache: Map<number, { [objId: string]: VideoMaskObjectData }>;
  maxFrameCacheSize: number;
  frameCacheBytes: number;
  currentBaseImage: HTMLImageElement | null;
  currentMaskObjects: { [objId: string]: VideoMaskObjectData };
}

type FrameMasks = { [objId: string]: VideoMaskObjectData };
interface PendingImage {
  image: HTMLImageElement;
  token: number | null;
}
interface PendingMasks {
  promise: Promise<FrameMasks | null>;
  cancel: () => void;
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
  private readonly pendingImages = new Map<number, PendingImage>();
  private readonly pendingMasks = new Map<number, PendingMasks>();
  // Estimated decoded RGBA pixels, excluding browser overhead and mask textures.
  private readonly maxFrameCacheBytes = 128 * 1024 * 1024;
  private readonly removedObjectIds = new Set<number>();
  private readonly state: FramePipelineState = {
    frameLoadToken: 0,
    pendingFrameIdx: null,
    frameLoadAnimationId: null,
    frameImageCache: new Map<number, HTMLImageElement>(),
    maskDataCache: new Map<number, { [objId: string]: VideoMaskObjectData }>(),
    maxFrameCacheSize: 24,
    frameCacheBytes: 0,
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
    for (const pending of this.pendingMasks.values()) pending.cancel();
    this.pendingMasks.clear();
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
    for (const [frameIdx] of this.pendingImages) this.cancelImage(frameIdx);
    this.state.frameImageCache.clear();
    this.state.frameCacheBytes = 0;
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
    // Keep only the requested image and useful immediate neighbors in flight.
    for (const [index, pending] of this.pendingImages) {
      if (Math.abs(index - frameIdx) > 1) this.cancelImage(index);
      else pending.token = null;
    }
    for (const [index, pending] of this.pendingMasks) {
      if (index !== frameIdx) {
        pending.cancel();
        this.pendingMasks.delete(index);
      }
    }
    const cachedImage = this.state.frameImageCache.get(frameIdx);
    if (cachedImage?.complete) {
      // Map insertion order is the eviction order; refresh it on a cache hit.
      this.state.frameImageCache.delete(frameIdx);
      this.state.frameImageCache.set(frameIdx, cachedImage);
      await this.displayLoadedFrame(cachedImage, frameIdx, token);
      return;
    }

    this.store.isFrameLoading.set(true);
    // Fetch masks alongside the image rather than starting another round trip after it.
    if (this.store.hasManifestMasks() && !this.state.maskDataCache.has(frameIdx)) {
      void this.loadMaskDataForFrame(frameIdx);
    }
    const pending = this.pendingImages.get(frameIdx);
    if (pending) pending.token = token;
    else this.requestImage(frameIdx, token);
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
        if (
          liveFrameMasks?.has(objId) ||
          liveEditedObjectIds.has(objId) ||
          this.removedObjectIds.has(objId)
        ) {
          continue;
        }
        const obj = this.store.objects().find((candidate) => candidate.id === objId);
        overlays.push({ objectId: objId, source: maskData, color: obj?.color || '#ff9800' });
      }
    }

    if (liveFrameMasks) {
      liveFrameMasks.forEach((mask, objId) => {
        if (!maskHasForeground(mask)) {
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
      const hasLiveMask = Boolean(selectedLiveMask && maskHasForeground(selectedLiveMask));
      if (hasLiveMask) {
        maskSource = 'live';
      } else if (
        !liveFrameMasks?.has(selectedObjectId) &&
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
    if (token !== this.state.frameLoadToken || !this.attached) return;
    this.state.currentBaseImage = image;
    const cachedMasks = this.state.maskDataCache.get(frameIdx);
    if (cachedMasks) {
      this.state.maskDataCache.delete(frameIdx);
      this.state.maskDataCache.set(frameIdx, cachedMasks);
    }
    this.state.currentMaskObjects = cachedMasks ?? {};
    this.renderFrame(image, frameIdx);
    this.store.displayedFrameIdx.set(frameIdx);
    this.store.isFrameLoading.set(false);
    this.preloadNeighborFrames(frameIdx);

    if (cachedMasks || !this.store.hasManifestMasks()) return;
    const generation = this.maskGeneration;
    const masks = await this.loadMaskDataForFrame(frameIdx);
    if (token !== this.state.frameLoadToken || generation !== this.maskGeneration || !masks) {
      return;
    }
    this.state.currentMaskObjects = masks;
    if (Object.keys(masks).length) this.renderFrame(image, frameIdx);
  }

  private cacheFrameImage(frameIdx: number, image: HTMLImageElement): void {
    if (this.state.frameImageCache.has(frameIdx)) {
      this.state.frameCacheBytes -= this.imageBytes(this.state.frameImageCache.get(frameIdx)!);
      this.state.frameImageCache.delete(frameIdx);
    }
    this.state.frameImageCache.set(frameIdx, image);
    this.state.frameCacheBytes += this.imageBytes(image);
    // Keep a single oversized image usable, but don't retain other images beside it.
    while (
      this.state.frameImageCache.size > 1 &&
      (this.state.frameImageCache.size > this.state.maxFrameCacheSize ||
        this.state.frameCacheBytes > this.maxFrameCacheBytes)
    ) {
      const oldestKey = this.state.frameImageCache.keys().next().value!;
      const oldest = this.state.frameImageCache.get(oldestKey)!;
      this.state.frameCacheBytes -= this.imageBytes(oldest);
      this.state.frameImageCache.delete(oldestKey);
      this.state.maskDataCache.delete(oldestKey);
    }
  }

  private imageBytes(image: HTMLImageElement): number {
    return (image.naturalWidth || image.width) * (image.naturalHeight || image.height) * 4;
  }

  private cancelImage(frameIdx: number): void {
    const pending = this.pendingImages.get(frameIdx);
    if (!pending) return;
    this.pendingImages.delete(frameIdx);
    pending.image.onload = pending.image.onerror = null;
    // Remove the resource to stop obsolete image loading where the browser supports it.
    pending.image.removeAttribute('src');
  }

  private requestImage(frameIdx: number, token: number | null): void {
    const image = new Image();
    const pending: PendingImage = { image, token };
    const generation = this.cacheGeneration;
    this.pendingImages.set(frameIdx, pending);
    image.onload = async () => {
      if (generation !== this.cacheGeneration || this.pendingImages.get(frameIdx) !== pending)
        return;
      this.pendingImages.delete(frameIdx);
      this.cacheFrameImage(frameIdx, image);
      if (pending.token !== null && pending.token === this.state.frameLoadToken) {
        await this.displayLoadedFrame(image, frameIdx, pending.token);
      }
    };
    image.onerror = () => {
      if (this.pendingImages.get(frameIdx) !== pending) return;
      this.pendingImages.delete(frameIdx);
      if (pending.token === this.state.frameLoadToken) {
        this.pendingMasks.get(frameIdx)?.cancel();
        this.pendingMasks.delete(frameIdx);
        console.error(`Failed to load frame image: ${image.src}`);
        this.store.isFrameLoading.set(false);
      }
    };
    image.src = this.backend.getVideoFrameUrl(frameIdx);
  }

  private preloadNeighborFrames(frameIdx: number): void {
    for (const neighborIdx of [frameIdx + 1, frameIdx - 1]) {
      if (
        neighborIdx < 0 ||
        neighborIdx >= this.store.numFrames() ||
        this.state.frameImageCache.has(neighborIdx) ||
        this.pendingImages.has(neighborIdx)
      ) {
        continue;
      }
      this.requestImage(neighborIdx, null);
    }
  }

  private loadMaskDataForFrame(frameIdx: number): Promise<FrameMasks | null> {
    const existing = this.pendingMasks.get(frameIdx);
    if (existing) return existing.promise;
    const generation = this.maskGeneration;
    const abort = new Subject<void>();
    let cancelled = false;
    const pending: PendingMasks = {
      cancel: () => {
        cancelled = true;
        abort.next();
        abort.complete();
      },
      promise: Promise.resolve(null),
    };
    this.pendingMasks.set(frameIdx, pending);
    pending.promise = (async () => {
      if (cancelled) return null;
      try {
        const response = await firstValueFrom(
          this.backend.getVideoMaskData(frameIdx).pipe(takeUntil(abort)),
        );
        if (cancelled || generation !== this.maskGeneration || (response as any)?.error)
          return null;
        const masks = response.objects || {};
        this.state.maskDataCache.set(frameIdx, masks);
        evictWithLimit(this.state.maskDataCache, this.state.maxFrameCacheSize);
        return masks;
      } catch (error) {
        if (!cancelled && generation === this.maskGeneration) console.error(error);
        return null;
      } finally {
        if (this.pendingMasks.get(frameIdx) === pending) this.pendingMasks.delete(frameIdx);
        abort.complete();
      }
    })();
    return pending.promise;
  }
}
