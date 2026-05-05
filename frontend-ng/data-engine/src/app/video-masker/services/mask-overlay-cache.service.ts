import { Injectable } from '@angular/core';
import {
  BackendService,
  VideoMaskDataWindowResponse,
  VideoMaskObjectData,
} from '../../services/backend.service';
import { MaskObject } from './video-masker-state.store';
import { firstValueFrom } from 'rxjs';

export interface PreparedMaskOverlay {
  frameIdx: number;
  objectKey: string;
  width: number;
  height: number;
  bitmap: ImageBitmap | HTMLCanvasElement;
}

export interface MaskOverlayPrefetchRequest {
  frameIdx: number;
  previousFrameIdx: number | null;
  numFrames: number;
  objects: MaskObject[];
  hasManifestMasks: boolean;
  liveEditedObjectIds: Set<number>;
  canvasWidth: number;
  canvasHeight: number;
  onOverlayReady: (frameIdx: number) => void;
}

@Injectable({ providedIn: 'root' })
export class MaskOverlayCacheService {
  private rawFrameMasks = new Map<number, { [objId: string]: VideoMaskObjectData }>();
  private preparedOverlays = new Map<string, PreparedMaskOverlay>();
  private inFlightWindows = new Map<string, Promise<void>>();
  private priorityQueue: Array<() => void> = [];
  private backgroundQueue: Array<() => void> = [];
  private idleScheduled = false;
  private maxPreparedFrames = 32;
  private maxRawFrames = 96;
  private cacheEpoch = 0;

  constructor(private backend?: BackendService) {}

  getRawCacheSize(): number {
    return this.rawFrameMasks.size;
  }

  getPreparedCacheSize(): number {
    return this.preparedOverlays.size;
  }

  clearAll(): void {
    this.cacheEpoch++;
    this.rawFrameMasks.clear();
    this.clearPrepared();
    this.inFlightWindows.clear();
    this.priorityQueue = [];
    this.backgroundQueue = [];
  }

  clearPrepared(): void {
    this.preparedOverlays.clear();
  }

  getPreparedOverlay(
    frameIdx: number,
    objects: MaskObject[],
    liveEditedObjectIds: Set<number>,
  ): PreparedMaskOverlay | null {
    const key = this.overlayKey(frameIdx, objects, liveEditedObjectIds);
    const overlay = this.preparedOverlays.get(key);
    if (!overlay) {
      return null;
    }
    this.preparedOverlays.delete(key);
    this.preparedOverlays.set(key, overlay);
    return overlay;
  }

  hasMaskForObject(frameIdx: number, objectId: number): boolean {
    return Boolean(this.rawFrameMasks.get(frameIdx)?.[String(objectId)]);
  }

  async prefetchAroundFrame(request: MaskOverlayPrefetchRequest): Promise<void> {
    if (
      !this.backend ||
      !request.hasManifestMasks ||
      request.numFrames <= 0 ||
      request.objects.length === 0 ||
      request.canvasWidth <= 0 ||
      request.canvasHeight <= 0
    ) {
      return;
    }
    const backend = this.backend;

    const objectIds = this.sortedObjectIds(request.objects);
    if (objectIds.length === 0) {
      return;
    }

    const [startFrameIdx, endFrameIdx] = this.prefetchWindow(
      request.frameIdx,
      request.previousFrameIdx,
      request.numFrames,
    );
    const requestEpoch = this.cacheEpoch;
    const windowKey = `${startFrameIdx}:${endFrameIdx}:${objectIds.join(',')}`;
    const existingRequest = this.inFlightWindows.get(windowKey);
    if (existingRequest) {
      await existingRequest;
      if (requestEpoch !== this.cacheEpoch) {
        return;
      }
    } else {
      const windowRequest = (async () => {
        const response = await firstValueFrom(
          backend.getVideoMaskDataWindow(startFrameIdx, endFrameIdx, objectIds),
        );
        if (requestEpoch !== this.cacheEpoch) {
          return;
        }
        this.storeWindowResponse(response);
      })();
      this.inFlightWindows.set(windowKey, windowRequest);
      try {
        await windowRequest;
      } finally {
        if (this.inFlightWindows.get(windowKey) === windowRequest) {
          this.inFlightWindows.delete(windowKey);
        }
      }
    }

    this.queueOverlayPreparation(
      request.frameIdx,
      request.objects,
      request.liveEditedObjectIds,
      request.canvasWidth,
      request.canvasHeight,
      true,
      request.onOverlayReady,
      requestEpoch,
    );

    for (let frameIdx = startFrameIdx; frameIdx <= endFrameIdx; frameIdx++) {
      if (frameIdx === request.frameIdx) {
        continue;
      }
      this.queueOverlayPreparation(
        frameIdx,
        request.objects,
        new Set<number>(),
        request.canvasWidth,
        request.canvasHeight,
        false,
        request.onOverlayReady,
        requestEpoch,
      );
    }
  }

  queueOverlayPreparation(
    frameIdx: number,
    objects: MaskObject[],
    liveEditedObjectIds: Set<number>,
    canvasWidth: number,
    canvasHeight: number,
    priority: boolean,
    onOverlayReady: (frameIdx: number) => void,
    cacheEpoch = this.cacheEpoch,
  ): void {
    if (objects.length === 0 || canvasWidth <= 0 || canvasHeight <= 0) {
      return;
    }
    const objectKey = this.overlayKey(frameIdx, objects, liveEditedObjectIds);
    if (this.preparedOverlays.has(objectKey)) {
      onOverlayReady(frameIdx);
      return;
    }
    const rawMasks = this.rawFrameMasks.get(frameIdx);
    if (!rawMasks) {
      return;
    }

    const task = () => {
      if (cacheEpoch !== this.cacheEpoch) {
        return;
      }
      if (this.preparedOverlays.has(objectKey)) {
        return;
      }
      const overlay = this.buildOverlay(
        frameIdx,
        objectKey,
        rawMasks,
        objects,
        liveEditedObjectIds,
        canvasWidth,
        canvasHeight,
      );
      if (!overlay) {
        return;
      }
      this.preparedOverlays.set(objectKey, overlay);
      this.evictPreparedOverlays();
      onOverlayReady(frameIdx);
    };

    if (priority) {
      this.priorityQueue.unshift(task);
    } else {
      this.backgroundQueue.push(task);
    }
    this.scheduleIdleWork();
  }

  private storeWindowResponse(response: VideoMaskDataWindowResponse): void {
    for (const [frameIdxRaw, framePayload] of Object.entries(response.frames || {})) {
      const frameIdx = Number(frameIdxRaw);
      if (!Number.isFinite(frameIdx)) {
        continue;
      }
      const normalizedFrameIdx = Math.trunc(frameIdx);
      const existingObjects = this.rawFrameMasks.get(normalizedFrameIdx) || {};
      this.rawFrameMasks.set(normalizedFrameIdx, {
        ...existingObjects,
        ...(framePayload.objects || {}),
      });
    }
    this.evictRawMasks();
  }

  private buildOverlay(
    frameIdx: number,
    objectKey: string,
    rawMasks: { [objId: string]: VideoMaskObjectData },
    objects: MaskObject[],
    liveEditedObjectIds: Set<number>,
    canvasWidth: number,
    canvasHeight: number,
  ): PreparedMaskOverlay | null {
    const canvas = document.createElement('canvas');
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return null;
    }

    const imageData = ctx.createImageData(canvasWidth, canvasHeight);
    const data = imageData.data;
    let hasForeground = false;
    for (const objectEntry of objects) {
      if (liveEditedObjectIds.has(objectEntry.id)) {
        continue;
      }
      const maskData = rawMasks[String(objectEntry.id)];
      if (!maskData) {
        continue;
      }
      const [r, g, b] = this.hexToRgb(objectEntry.color);
      const size = maskData.size;
      const sourceHeight = Number(size?.[0]);
      const sourceWidth = Number(size?.[1]);
      if (
        !Number.isFinite(sourceHeight) ||
        !Number.isFinite(sourceWidth) ||
        sourceHeight <= 0 ||
        sourceWidth <= 0
      ) {
        continue;
      }
      for (const run of maskData.rle || []) {
        if (!Array.isArray(run) || run.length !== 2) {
          continue;
        }
        const start = Math.max(0, Number(run[0]) | 0);
        const length = Math.max(0, Number(run[1]) | 0);
        const end = Math.min(sourceWidth * sourceHeight, start + length);
        for (let index = start; index < end; index++) {
          const sourceY = Math.floor(index / sourceWidth);
          const sourceX = index - sourceY * sourceWidth;
          const targetX = sourceWidth === canvasWidth
            ? sourceX
            : Math.min(canvasWidth - 1, Math.floor((sourceX / sourceWidth) * canvasWidth));
          const targetY = sourceHeight === canvasHeight
            ? sourceY
            : Math.min(canvasHeight - 1, Math.floor((sourceY / sourceHeight) * canvasHeight));
          const pixelOffset = (targetY * canvasWidth + targetX) * 4;
          data[pixelOffset] = r;
          data[pixelOffset + 1] = g;
          data[pixelOffset + 2] = b;
          data[pixelOffset + 3] = 120;
          hasForeground = true;
        }
      }
    }

    if (!hasForeground) {
      return null;
    }
    ctx.putImageData(imageData, 0, 0);
    return {
      frameIdx,
      objectKey,
      width: canvasWidth,
      height: canvasHeight,
      bitmap: canvas,
    };
  }

  private prefetchWindow(
    frameIdx: number,
    previousFrameIdx: number | null,
    numFrames: number,
  ): [number, number] {
    const clampedFrame = Math.min(Math.max(Math.trunc(frameIdx), 0), Math.max(numFrames - 1, 0));
    if (previousFrameIdx !== null && Math.abs(clampedFrame - previousFrameIdx) > 1) {
      if (clampedFrame > previousFrameIdx) {
        return [clampedFrame, Math.min(numFrames - 1, clampedFrame + 16)];
      }
      return [Math.max(0, clampedFrame - 16), clampedFrame];
    }
    return [Math.max(0, clampedFrame - 8), Math.min(numFrames - 1, clampedFrame + 8)];
  }

  private overlayKey(
    frameIdx: number,
    objects: MaskObject[],
    liveEditedObjectIds: Set<number>,
  ): string {
    const objectKey = objects
      .filter((entry) => !liveEditedObjectIds.has(entry.id))
      .slice()
      .sort((a, b) => a.id - b.id)
      .map((entry) => `${entry.id}:${entry.color}`)
      .join('|');
    return `${frameIdx}:${objectKey}`;
  }

  private sortedObjectIds(objects: MaskObject[]): number[] {
    return objects
      .map((entry) => Math.trunc(entry.id))
      .filter((id) => Number.isFinite(id) && id > 0)
      .sort((a, b) => a - b);
  }

  private scheduleIdleWork(): void {
    if (this.idleScheduled) {
      return;
    }
    this.idleScheduled = true;
    const callback = () => {
      this.idleScheduled = false;
      const task = this.priorityQueue.shift() || this.backgroundQueue.shift();
      task?.();
      if (this.priorityQueue.length > 0 || this.backgroundQueue.length > 0) {
        this.scheduleIdleWork();
      }
    };
    const requestIdle = (globalThis as any).requestIdleCallback as
      | ((handler: () => void) => number)
      | undefined;
    if (requestIdle) {
      requestIdle(callback);
      return;
    }
    setTimeout(callback, 0);
  }

  private evictRawMasks(): void {
    while (this.rawFrameMasks.size > this.maxRawFrames) {
      const oldestKey = this.rawFrameMasks.keys().next().value;
      if (oldestKey === undefined) {
        break;
      }
      this.rawFrameMasks.delete(oldestKey);
    }
  }

  private evictPreparedOverlays(): void {
    while (this.preparedOverlays.size > this.maxPreparedFrames) {
      const oldestKey = this.preparedOverlays.keys().next().value;
      if (oldestKey === undefined) {
        break;
      }
      this.preparedOverlays.delete(oldestKey);
    }
  }

  private hexToRgb(hex: string): [number, number, number] {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
      : [255, 152, 0];
  }
}
