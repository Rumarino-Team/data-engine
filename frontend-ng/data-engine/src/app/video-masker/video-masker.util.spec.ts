import { describe, expect, it } from 'vitest';
import {
  clampFrameIndex,
  evictWithLimit,
  markObjectLiveEdited,
  maskHasForeground,
  normalizeMask2d,
  resolveStateEpoch,
  unmarkObjectLiveEdited,
} from './video-masker.util';

describe('video-masker.util', () => {
  it('evicts oldest entries when over max size', () => {
    const cache = new Map<number, string>([
      [1, 'a'],
      [2, 'b'],
      [3, 'c'],
    ]);
    const evicted: number[] = [];

    evictWithLimit(cache, 2, (key) => evicted.push(key));

    expect(cache.size).toBe(2);
    expect(cache.has(1)).toBe(false);
    expect(evicted).toEqual([1]);
  });

  it('clamps frame index to [0,max]', () => {
    expect(clampFrameIndex(-1, 10)).toBe(0);
    expect(clampFrameIndex(11, 10)).toBe(10);
  });

  it('flags epoch mismatch for live state clear', () => {
    expect(resolveStateEpoch(3, 4)).toEqual({ normalizedEpoch: 4, shouldClearLiveState: true });
  });

  it('does not clear when epoch is unchanged', () => {
    expect(resolveStateEpoch(3, 3)).toEqual({ normalizedEpoch: 3, shouldClearLiveState: false });
  });

  it('ignores non-positive / non-finite epochs', () => {
    expect(resolveStateEpoch(3, 0)).toBeNull();
    expect(resolveStateEpoch(3, undefined)).toBeNull();
  });

  it('tracks live-edited mark/unmark correctly', () => {
    const marked = markObjectLiveEdited(new Map(), 2, 7);
    expect(marked.get(2)?.has(7)).toBe(true);

    const unmarked = unmarkObjectLiveEdited(marked, 2, 7);
    expect(unmarked.get(2)).toBeUndefined();
  });

  it('normalizes nested mask arrays', () => {
    const mask = [[[[true, false], [false, true]]]];
    expect(normalizeMask2d(mask)).toEqual([
      [true, false],
      [false, true],
    ]);
  });

  it('detects foreground pixels', () => {
    expect(maskHasForeground([[false], [false]])).toBe(false);
    expect(maskHasForeground([[false], [true]])).toBe(true);
  });
});
