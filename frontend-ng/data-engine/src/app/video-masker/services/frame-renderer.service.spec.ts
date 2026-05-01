import { describe, expect, it } from 'vitest';
import { FrameRendererService } from './frame-renderer.service';

describe('FrameRendererService', () => {
  it('evicts oldest entries when over max size', () => {
    const service = new FrameRendererService();
    const cache = new Map<number, string>([
      [1, 'a'],
      [2, 'b'],
      [3, 'c'],
    ]);
    const evicted: number[] = [];

    service.evictWithLimit(cache, 2, (key) => evicted.push(key));

    expect(cache.size).toBe(2);
    expect(cache.has(1)).toBe(false);
    expect(evicted).toEqual([1]);
  });

  it('clamps frame index to [0,max]', () => {
    const service = new FrameRendererService();
    expect(service.clampFrameIndex(-1, 10)).toBe(0);
    expect(service.clampFrameIndex(11, 10)).toBe(10);
  });
});
