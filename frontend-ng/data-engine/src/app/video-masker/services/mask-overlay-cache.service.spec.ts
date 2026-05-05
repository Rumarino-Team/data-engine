import { of, Subject } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { BackendService } from '../../services/backend.service';
import { MaskOverlayCacheService } from './mask-overlay-cache.service';
import { MaskObject } from './video-masker-state.store';

describe('MaskOverlayCacheService', () => {
  const objects: MaskObject[] = [{ id: 1, name: 'Object 1', color: '#ff0000' }];
  const originalRequestIdleCallback = (globalThis as any).requestIdleCallback;
  const originalCreateElement = document.createElement.bind(document);
  let createElementSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    (globalThis as any).requestIdleCallback = (handler: () => void) => {
      handler();
      return 1;
    };
    createElementSpy = vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      if (tagName !== 'canvas') {
        return originalCreateElement(tagName);
      }
      return {
        width: 0,
        height: 0,
        getContext: vi.fn(() => ({
          createImageData: vi.fn((width: number, height: number) => ({
            data: new Uint8ClampedArray(width * height * 4),
            width,
            height,
          })),
          putImageData: vi.fn(),
        })),
      } as unknown as HTMLCanvasElement;
    });
  });

  afterEach(() => {
    createElementSpy.mockRestore();
    (globalThis as any).requestIdleCallback = originalRequestIdleCallback;
  });

  it('fetches a window and stores raw frame masks', async () => {
    const backend = {
      getVideoMaskDataWindow: vi.fn(() => of({
        start_frame_idx: 0,
        end_frame_idx: 0,
        frames: { '0': { objects: { '1': maskPayload() } } },
      })),
    } as unknown as BackendService;
    const service = new MaskOverlayCacheService(backend);

    await service.prefetchAroundFrame(baseRequest({ frameIdx: 0, objects }));

    expect(service.getRawCacheSize()).toBe(1);
    expect(service.hasMaskForObject(0, 1)).toBe(true);
  });

  it('does not duplicate an in-flight window request', async () => {
    const response$ = new Subject<any>();
    const backend = {
      getVideoMaskDataWindow: vi.fn(() => response$),
    } as unknown as BackendService;
    const service = new MaskOverlayCacheService(backend);
    const secondReady = vi.fn();

    const first = service.prefetchAroundFrame(baseRequest({ frameIdx: 0, objects }));
    const second = service.prefetchAroundFrame(baseRequest({
      frameIdx: 0,
      objects,
      onOverlayReady: secondReady,
    }));
    response$.next({
      start_frame_idx: 0,
      end_frame_idx: 0,
      frames: { '0': { objects: { '1': maskPayload() } } },
    });
    response$.complete();
    await Promise.all([first, second]);

    expect((backend.getVideoMaskDataWindow as any).mock.calls.length).toBe(1);
    expect(secondReady).toHaveBeenCalledWith(0);
  });

  it('ignores stale in-flight responses after invalidation', async () => {
    const response$ = new Subject<any>();
    const backend = {
      getVideoMaskDataWindow: vi.fn(() => response$),
    } as unknown as BackendService;
    const service = new MaskOverlayCacheService(backend);

    const pending = service.prefetchAroundFrame(baseRequest({ frameIdx: 0, objects }));
    service.clearAll();
    response$.next({
      start_frame_idx: 0,
      end_frame_idx: 0,
      frames: { '0': { objects: { '1': maskPayload() } } },
    });
    response$.complete();
    await pending;

    expect(service.getRawCacheSize()).toBe(0);
    expect(service.hasMaskForObject(0, 1)).toBe(false);
  });

  it('evicts prepared overlays after the cache budget', async () => {
    const backend = {
      getVideoMaskDataWindow: vi.fn((startFrameIdx: number) => of({
        start_frame_idx: startFrameIdx,
        end_frame_idx: startFrameIdx,
        frames: { [String(startFrameIdx)]: { objects: { '1': maskPayload() } } },
      })),
    } as unknown as BackendService;
    const service = new MaskOverlayCacheService(backend);

    for (let frameIdx = 0; frameIdx < 33; frameIdx++) {
      await service.prefetchAroundFrame(baseRequest({
        frameIdx,
        previousFrameIdx: frameIdx - 2,
        numFrames: 64,
        objects,
      }));
    }

    expect(service.getPreparedCacheSize()).toBe(32);
  });

  it('merges raw frame masks from overlapping object windows', async () => {
    const backend = {
      getVideoMaskDataWindow: vi
        .fn()
        .mockReturnValueOnce(of({
          start_frame_idx: 0,
          end_frame_idx: 0,
          frames: { '0': { objects: { '1': maskPayload() } } },
        }))
        .mockReturnValueOnce(of({
          start_frame_idx: 0,
          end_frame_idx: 0,
          frames: { '0': { objects: { '2': maskPayload() } } },
        })),
    } as unknown as BackendService;
    const service = new MaskOverlayCacheService(backend);

    await service.prefetchAroundFrame(baseRequest({
      frameIdx: 0,
      objects: [{ id: 1, name: 'Object 1', color: '#ff0000' }],
    }));
    await service.prefetchAroundFrame(baseRequest({
      frameIdx: 0,
      objects: [{ id: 2, name: 'Object 2', color: '#00ff00' }],
    }));

    expect(service.hasMaskForObject(0, 1)).toBe(true);
    expect(service.hasMaskForObject(0, 2)).toBe(true);
  });

  it('filters object IDs before requesting', async () => {
    const backend = {
      getVideoMaskDataWindow: vi.fn(() => of({ start_frame_idx: 0, end_frame_idx: 0, frames: {} })),
    } as unknown as BackendService;
    const service = new MaskOverlayCacheService(backend);

    await service.prefetchAroundFrame(baseRequest({
      frameIdx: 0,
      objects: [
        { id: 2, name: 'Object 2', color: '#00ff00' },
        { id: 1, name: 'Object 1', color: '#ff0000' },
        { id: -1, name: 'Invalid', color: '#000000' },
      ],
    }));

    expect((backend.getVideoMaskDataWindow as any).mock.calls[0][2]).toEqual([1, 2]);
  });

  it('prioritizes current frame overlay generation', async () => {
    const readyFrames: number[] = [];
    const backend = {
      getVideoMaskDataWindow: vi.fn(() => of({
        start_frame_idx: 4,
        end_frame_idx: 5,
        frames: {
          '4': { objects: { '1': maskPayload() } },
          '5': { objects: { '1': maskPayload() } },
        },
      })),
    } as unknown as BackendService;
    const service = new MaskOverlayCacheService(backend);

    await service.prefetchAroundFrame(baseRequest({
      frameIdx: 5,
      previousFrameIdx: 3,
      numFrames: 6,
      objects,
      onOverlayReady: (frameIdx) => readyFrames.push(frameIdx),
    }));

    expect(readyFrames[0]).toBe(5);
  });

  it('uses object colors in prepared overlay keys', async () => {
    const backend = {
      getVideoMaskDataWindow: vi.fn(() => of({
        start_frame_idx: 0,
        end_frame_idx: 0,
        frames: { '0': { objects: { '1': maskPayload() } } },
      })),
    } as unknown as BackendService;
    const service = new MaskOverlayCacheService(backend);

    await service.prefetchAroundFrame(baseRequest({ frameIdx: 0, objects }));

    expect(service.getPreparedOverlay(0, objects, new Set())).not.toBeNull();
    expect(service.getPreparedOverlay(
      0,
      [{ id: 1, name: 'Object 1', color: '#00ff00' }],
      new Set(),
    )).toBeNull();
  });

  function baseRequest(overrides: Partial<Parameters<MaskOverlayCacheService['prefetchAroundFrame']>[0]> = {}) {
    return {
      frameIdx: 0,
      previousFrameIdx: null,
      numFrames: 1,
      objects,
      hasManifestMasks: true,
      liveEditedObjectIds: new Set<number>(),
      canvasWidth: 1,
      canvasHeight: 1,
      onOverlayReady: vi.fn(),
      ...overrides,
    };
  }

  function maskPayload() {
    return { size: [1, 1] as [number, number], rle: [[0, 1]], bbox: [0, 0, 1, 1] as [number, number, number, number] };
  }
});
