import { describe, expect, it } from 'vitest';
import {
  isMaskRle,
  packedMaskBytes,
  encodedMaskPixelCount,
  clampFrameIndex,
  evictWithLimit,
  normalizeMask2d,
  resolveStateEpoch,
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
});

describe('packed masks', () => {
  it('uses MSB-first bits across rows and saves identical counts', () => {
    const mask = {
      size: [3, 3] as [number, number],
      encoding: 'packed-bits' as const,
      data: 'qoA=',
    };
    expect(packedMaskBytes(mask)).toEqual(new Uint8Array([170, 128]));
    expect(packedMaskBytes(mask)).toBe(packedMaskBytes(mask));
    expect(encodedMaskPixelCount(mask)).toBe(5);
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
  it('counts empty and full partial bytes', () => {
    expect(encodedMaskPixelCount({ size: [1, 3], encoding: 'packed-bits', data: 'AA==' })).toBe(0);
    expect(encodedMaskPixelCount({ size: [1, 3], encoding: 'packed-bits', data: '4A==' })).toBe(3);
  });
  it('rejects truncated bytes, noncanonical base64, invalid dimensions and nonzero padding', () => {
    for (const data of ['', '!!==', 'AQ==', 'AB==', 'AAA=']) {
      expect(() => packedMaskBytes({ size: [1, 3], encoding: 'packed-bits', data })).toThrow();
    }
    expect(() => packedMaskBytes({ size: [0, 3], encoding: 'packed-bits', data: '' })).toThrow();
  });
});
