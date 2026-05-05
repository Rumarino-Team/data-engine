import { describe, expect, it, vi } from 'vitest';
import { VideoMaskerRenderingService } from './video-masker-rendering.service';

describe('VideoMaskerRenderingService', () => {
  const service = new VideoMaskerRenderingService();

  it('normalizes nested mask arrays', () => {
    const mask = [[[ [true, false], [false, true] ]]];
    const out = service.normalizeMask2d(mask);
    expect(out).toEqual([[true, false], [false, true]]);
  });

  it('detects foreground pixels', () => {
    expect(service.maskHasForeground([[false], [false]])).toBe(false);
    expect(service.maskHasForeground([[false], [true]])).toBe(true);
  });

  it('draws prepared mask overlays without smoothing and restores the previous setting', () => {
    const ctx = {
      canvas: { width: 100, height: 50 },
      imageSmoothingEnabled: true,
      drawImage: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
    const overlay = {} as HTMLCanvasElement;

    service.drawPreparedMaskOverlay(ctx, overlay);

    expect((ctx.drawImage as any).mock.calls[0]).toEqual([overlay, 0, 0, 100, 50]);
    expect(ctx.imageSmoothingEnabled).toBe(true);
  });
});
