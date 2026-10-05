import { describe, expect, it } from 'vitest';
import {
  buildInteractiveStateSnapshot,
  deserializeLiveMasks,
  encodeMaskToCounts,
  isMaskRle,
  packedMaskBytes,
  encodedMaskPixelCount,
  maskFromCounts,
  clampFrameIndex,
  evictWithLimit,
  markObjectLiveEdited,
  maskHasForeground,
  normalizeMask2d,
  resolveStateEpoch,
  unmarkObjectLiveEdited,
} from './video-masker.util';

import { VideoMaskObjectData } from '../services/backend-api.types';

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
    const mask = [
      [
        [
          [true, false],
          [false, true],
        ],
      ],
    ];
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

describe('encoded live masks', () => {
  const mask: VideoMaskObjectData = {
    size: [2, 3],
    rle: [
      [1, 3],
      [5, 1],
    ],
    bbox: [0, 0, 3, 2],
  };

  it('converts runs crossing row boundaries to saved counts and back without grids', () => {
    const encoded = encodeMaskToCounts(mask)!;
    expect(encoded).toEqual({ height: 2, width: 3, counts: [1, 3, 1, 1] });
    expect(maskFromCounts({ ...encoded, frame_idx: 4, obj_id: 2 })).toEqual(mask);
  });

  it('preserves empty and fully foreground masks', () => {
    const empty = maskFromCounts({
      height: 2160,
      width: 3840,
      counts: [8294400],
      frame_idx: 0,
      obj_id: 1,
    })!;
    expect(empty.rle).toEqual([]);
    expect(maskHasForeground(empty)).toBe(false);
    expect(encodeMaskToCounts(empty)?.counts).toEqual([8294400]);
    const full = maskFromCounts({ height: 2, width: 2, counts: [0, 4], frame_idx: 0, obj_id: 1 })!;
    expect(full).toEqual({ size: [2, 2], rle: [[0, 4]], bbox: [0, 0, 2, 2] });
    expect(maskHasForeground(full)).toBe(true);
  });

  it('retains encoded masks through save and restore', () => {
    const snapshot = buildInteractiveStateSnapshot({
      objects: [],
      selectedObjectId: 2,
      interactionMode: 'positive',
      currentFrameIdx: 4,
      pointsByFrame: new Map(),
      masksByFrame: new Map([[4, new Map([[2, mask]])]]),
    });
    const restored = deserializeLiveMasks(snapshot.live_masks);
    expect(restored.masks.get(4)?.get(2)).toEqual(mask);
    expect(restored.liveEditedFrames.get(4)?.has(2)).toBe(true);
  });

  it('still saves boolean masks from an older backend', () => {
    expect(
      encodeMaskToCounts([
        [false, true],
        [true, false],
      ]),
    ).toEqual({ height: 2, width: 2, counts: [1, 2, 1] });
  });

  it.each([
    { ...mask, size: [0, 3] },
    { ...mask, rle: [[-1, 2]] },
    {
      ...mask,
      rle: [
        [1, 3],
        [2, 1],
      ],
    },
    { ...mask, rle: [[5, 2]] },
    { ...mask, rle: [[0, 0]] },
    { ...mask, rle: [[0.5, 1]] },
    { ...mask, bbox: [0, 0, 4, 2] },
  ])('rejects invalid live RLE payloads %j', (invalid) => {
    expect(isMaskRle(invalid)).toBe(false);
  });

  it.each([[7], [1, -1, 6], [1.5, 4.5], [], [5]])(
    'rejects invalid saved counts %j',
    (...counts) => {
      expect(
        maskFromCounts({
          height: 2,
          width: 3,
          counts: counts as number[],
          frame_idx: 0,
          obj_id: 1,
        }),
      ).toBeNull();
    },
  );
});

describe('packed masks', () => {
  it('uses MSB-first bits across rows and saves identical counts', () => {
    const mask = {
      size: [3, 3] as [number, number],
      encoding: 'packed-bits' as const,
      data: 'qoA=',
    };
    const grid = [
      [true, false, true],
      [false, true, false],
      [true, false, true],
    ];
    expect(packedMaskBytes(mask)).toEqual(new Uint8Array([170, 128]));
    expect(packedMaskBytes(mask)).toBe(packedMaskBytes(mask));
    expect(encodedMaskPixelCount(mask)).toBe(5);
    expect(maskHasForeground(mask)).toBe(true);
    expect(encodeMaskToCounts(mask)).toEqual(encodeMaskToCounts(grid));
    const saved = encodeMaskToCounts(mask)!;
    const restored = maskFromCounts({ ...saved, frame_idx: 0, obj_id: 1 });
    expect(restored && encodeMaskToCounts(restored)).toEqual(saved);
  });
  it('decodes a full 1080p fragmented mask without a pixel grid', () => {
    const raw = String.fromCharCode(170).repeat((1920 * 1080) / 8);
    const mask = {
      size: [1080, 1920] as [number, number],
      encoding: 'packed-bits' as const,
      data: btoa(raw),
    };
    expect(packedMaskBytes(mask).length).toBe(259200);
    expect(encodedMaskPixelCount(mask)).toBe(1036800);
  });
  it('handles empty and full partial bytes', () => {
    expect(maskHasForeground({ size: [1, 3], encoding: 'packed-bits', data: 'AA==' })).toBe(false);
    expect(encodeMaskToCounts({ size: [1, 3], encoding: 'packed-bits', data: '4A==' })).toEqual({
      height: 1,
      width: 3,
      counts: [0, 3],
    });
  });
  it('rejects truncated bytes, noncanonical base64, invalid dimensions and nonzero padding', () => {
    for (const data of ['', '!!==', 'AQ==', 'AB==', 'AAA=']) {
      expect(() => packedMaskBytes({ size: [1, 3], encoding: 'packed-bits', data })).toThrow();
    }
    expect(() => packedMaskBytes({ size: [0, 3], encoding: 'packed-bits', data: '' })).toThrow();
  });
});
