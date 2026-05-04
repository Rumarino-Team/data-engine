import { describe, expect, it } from 'vitest';
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
});
