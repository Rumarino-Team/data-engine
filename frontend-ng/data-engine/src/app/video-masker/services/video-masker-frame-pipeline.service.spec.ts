import { describe, expect, it } from 'vitest';
import { FrameRendererService } from './frame-renderer.service';
import { VideoMaskerFramePipelineService } from './video-masker-frame-pipeline.service';

describe('VideoMaskerFramePipelineService', () => {
  it('clears frame caches and resets state', () => {
    const service = new VideoMaskerFramePipelineService(new FrameRendererService());
    const state = {
      frameLoadToken: 1,
      pendingFrameIdx: 4,
      frameLoadAnimationId: null,
      frameImageCache: new Map([[1, {} as HTMLImageElement]]),
      maskDataCache: new Map([[1, {} as any]]),
      maxFrameCacheSize: 24,
      currentBaseImage: {} as HTMLImageElement,
      currentMaskObjects: { '1': {} as any },
    };

    service.clearFrameCaches(state);

    expect(state.frameImageCache.size).toBe(0);
    expect(state.maskDataCache.size).toBe(0);
    expect(state.currentBaseImage).toBeNull();
    expect(state.currentMaskObjects).toEqual({});
    expect(state.frameLoadToken).toBe(2);
  });
});
