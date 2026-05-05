import { describe, expect, it, vi } from 'vitest';
import { FrameRendererService } from './frame-renderer.service';
import { VideoMaskerFramePipelineService } from './video-masker-frame-pipeline.service';

describe('VideoMaskerFramePipelineService', () => {
  it('clears frame caches and resets state', () => {
    const maskOverlayCache = { prefetchAroundFrame: vi.fn() } as any;
    const service = new VideoMaskerFramePipelineService(new FrameRendererService(), maskOverlayCache);
    const state = {
      frameLoadToken: 1,
      pendingFrameIdx: 4,
      frameLoadAnimationId: null,
      frameImageCache: new Map([[1, {} as HTMLImageElement]]),
      maxFrameCacheSize: 24,
      currentBaseImage: {} as HTMLImageElement,
      previousFrameIdx: 3,
    };

    service.clearFrameCaches(state);

    expect(state.frameImageCache.size).toBe(0);
    expect(state.currentBaseImage).toBeNull();
    expect(state.previousFrameIdx).toBeNull();
    expect(state.frameLoadToken).toBe(2);
  });

  it('draws a cached frame immediately and schedules mask prefetch', async () => {
    const originalImage = (globalThis as any).Image;
    (globalThis as any).Image = class {
      onload: (() => void) | null = null;
      set src(_value: string) {}
    };
    const cachedImage = { complete: true, width: 320, height: 180 } as HTMLImageElement;
    const maskOverlayCache = { prefetchAroundFrame: vi.fn(async () => undefined) } as any;
    const service = new VideoMaskerFramePipelineService(new FrameRendererService(), maskOverlayCache);
    const state = {
      frameLoadToken: 0,
      pendingFrameIdx: null,
      frameLoadAnimationId: null,
      frameImageCache: new Map([[2, cachedImage]]),
      maxFrameCacheSize: 24,
      currentBaseImage: null,
      previousFrameIdx: null,
    };
    const canvas = { width: 0, height: 0, dataset: {} } as HTMLCanvasElement;
    const draw = vi.fn();

    try {
      await service.loadFrame(2, state, {
        canvasRef: { nativeElement: canvas },
        getVideoFrameUrl: vi.fn((frameIdx: number) => `/frame/${frameIdx}`),
        hasManifestMasks: () => true,
        numFrames: () => 10,
        objects: () => [{ id: 1, name: 'Object 1', color: '#ff0000' }],
        liveEditedObjectIdsForFrame: () => new Set<number>(),
        draw,
        onDisplayedFrame: vi.fn(),
        setIsFrameLoading: vi.fn(),
      });

      expect(draw).toHaveBeenCalledWith(cachedImage, 2);
      expect(maskOverlayCache.prefetchAroundFrame).toHaveBeenCalled();
      expect(canvas.width).toBe(320);
      expect(canvas.height).toBe(180);
    } finally {
      (globalThis as any).Image = originalImage;
    }
  });
});
