import { Injectable, inject, untracked } from '@angular/core';
import { firstValueFrom, Subject, takeUntil } from 'rxjs';
import { BackendService, VideoMaskDataResponse } from '../../services/backend.service';
import { MaskOverlay } from './mask-texture';
import { CanvasViewportService } from './canvas-viewport.service';
import { VideoMaskerStateStore } from './video-masker-state.store';
import { evictWithLimit, encodedMaskPixelCount } from '../video-masker.util';

interface FramePipelineState {
  frameLoadToken: number;
  pendingFrameIdx: number | null;
  frameLoadAnimationId: number | null;
  frameImageCache: Map<number, HTMLImageElement>;
  maskDataCache: Map<number, VideoMaskDataResponse['objects']>;
  maxFrameCacheSize: number;
  frameCacheBytes: number;
  currentBaseImage: HTMLImageElement | null;
  currentMaskObjects: VideoMaskDataResponse['objects'];
}

type FrameMasks = VideoMaskDataResponse['objects'];
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
  private maskContext = '';
  private readonly maskSources = new Map<number, VideoMaskDataResponse['sources']>();
  private readonly state: FramePipelineState = {
    frameLoadToken: 0,
    pendingFrameIdx: null,
    frameLoadAnimationId: null,
    frameImageCache: new Map<number, HTMLImageElement>(),
    maskDataCache: new Map<number, VideoMaskDataResponse['objects']>(),
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
    this.maskSources.clear();
    for (const pending of this.pendingMasks.values()) pending.cancel();
    this.pendingMasks.clear();
    this.state.maskDataCache.clear();
    this.state.currentMaskObjects = {};
  }

  private context(): string {
    return JSON.stringify([
      this.backend.getApiUrl(),
      this.store.sessionId(),
      this.store.annotationRevision(),
      this.store.stateEpoch(),
    ]);
  }

  private ensureMaskContext(): void {
    const context = this.context();
    if (context !== this.maskContext) {
      this.clearMaskDataCache();
      this.maskContext = context;
    }
  }

  acceptMaskData(response: VideoMaskDataResponse | undefined, frameIdx: number): boolean {
    this.ensureMaskContext();
    if (
      !response ||
      response.frame_idx !== frameIdx ||
      response.session_id !== this.store.sessionId() ||
      response.annotation_revision !== this.store.annotationRevision() ||
      response.state_epoch !== this.store.stateEpoch() ||
      this.backend.getApiUrl() !== this.store.annotationsBackendUrl() ||
      !response.objects ||
      !response.sources
    )
      return false;
    try {
      for (const [id, mask] of Object.entries(response.objects)) {
        if (
          !this.store.annotations()?.objects[id] ||
          !['live', 'manifest', 'override'].includes(response.sources[id])
        )
          return false;
        encodedMaskPixelCount(mask);
      }
    } catch {
      return false;
    }
    this.state.maskDataCache.set(frameIdx, response.objects);
    this.maskSources.set(frameIdx, response.sources);
    evictWithLimit(this.state.maskDataCache, this.state.maxFrameCacheSize);
    evictWithLimit(this.maskSources, this.state.maxFrameCacheSize);
    if (frameIdx === this.store.displayedFrameIdx())
      this.state.currentMaskObjects = response.objects;
    return true;
  }

  clearFrameCaches(): void {
    this.viewport.reset();
    this.cacheGeneration++;
    this.clearMaskDataCache();
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
    // Invalidate obsolete display callbacks immediately, before the next RAF.
    this.state.frameLoadToken++;
    this.retainUsefulRequests(frameIdx);
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

    this.ensureMaskContext();
    const token = ++this.state.frameLoadToken;
    this.retainUsefulRequests(frameIdx);
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
    if (!this.state.maskDataCache.has(frameIdx)) {
      void this.loadMaskDataForFrame(frameIdx);
    }
    const pending = this.pendingImages.get(frameIdx);
    if (pending) pending.token = token;
    else this.requestImage(frameIdx, token);
  }

  private retainUsefulRequests(frameIdx: number): void {
    // Keep the requested frame and immediate neighbors for both resource types.
    for (const [index, pending] of this.pendingImages) {
      if (Math.abs(index - frameIdx) > 1) this.cancelImage(index);
      else pending.token = null;
    }
    for (const [index, pending] of this.pendingMasks) {
      if (Math.abs(index - frameIdx) > 1) {
        pending.cancel();
        this.pendingMasks.delete(index);
      }
    }
  }

  /** Repaint the frame that is currently displayed, if any. */
  redraw(): void {
    this.ensureMaskContext();
    const frameIdx = this.store.displayedFrameIdx();
    if (frameIdx < 0 || !this.state.currentBaseImage || !this.attached) {
      return;
    }
    this.renderFrame(this.state.currentBaseImage, frameIdx);
  }

  private renderFrame(img: HTMLImageElement, frameIdx: number): void {
    this.ensureMaskContext();
    const overlays: MaskOverlay[] = [];
    for (const [id, mask] of Object.entries(this.state.currentMaskObjects)) {
      const object = this.store.objects().find((candidate) => candidate.id === Number(id));
      if (object) overlays.push({ objectId: object.id, source: mask, color: object.color });
    }
    const selected = this.store.selectedObjectId();
    this.store.lastMaskSource.set(
      selected === null ? 'none' : (this.maskSources.get(frameIdx)?.[selected] ?? 'none'),
    );

    const points = Array.from(this.store.previewPoints().get(frameIdx)?.values() ?? []).flat();
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
    this.ensureMaskContext();
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

    if (cachedMasks) {
      this.preloadNeighborMasks(frameIdx);
      return;
    }
    const generation = this.maskGeneration;
    const masks = await this.loadMaskDataForFrame(frameIdx);
    if (token !== this.state.frameLoadToken || generation !== this.maskGeneration || !masks) {
      return;
    }
    this.state.currentMaskObjects = masks;
    this.renderFrame(image, frameIdx);
    this.preloadNeighborMasks(frameIdx);
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

  private preloadNeighborMasks(frameIdx: number): void {
    // Start speculative mask work only after the displayed frame's image and
    // masks are ready. Navigation promotes an existing request instead of restarting it.
    for (const neighborIdx of [frameIdx + 1, frameIdx - 1]) {
      if (
        neighborIdx >= 0 &&
        neighborIdx < this.store.numFrames() &&
        !this.state.maskDataCache.has(neighborIdx) &&
        !this.pendingMasks.has(neighborIdx)
      )
        void this.loadMaskDataForFrame(neighborIdx);
    }
  }

  private loadMaskDataForFrame(frameIdx: number): Promise<FrameMasks | null> {
    this.ensureMaskContext();
    const existing = this.pendingMasks.get(frameIdx);
    if (existing) return existing.promise;
    const context = this.context();
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
        if (
          cancelled ||
          generation !== this.maskGeneration ||
          context !== this.context() ||
          !this.acceptMaskData(response, frameIdx)
        )
          return null;
        return response.objects;
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
