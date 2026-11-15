import { describe, expect, it } from 'vitest';
import {
  isMaskRle,
  packedMaskBytes,
  encodedMaskPixelCount,
  clampFrameIndex,
  evictWithLimit,
  normalizeMask2d,
  frameLoadingHint,
  isFrameLoadingMode,
  outdatedTimelineSegments,
} from './video-masker.util';

import { VideoMaskObjectData } from '../services/backend-api.types';

describe('video-masker.util', () => {
  it('recognises exactly the frame loading modes the backend accepts', () => {
    for (const mode of ['lazy', 'preload', 'eager']) expect(isFrameLoadingMode(mode)).toBe(true);
    for (const other of ['', 'Lazy', 'streaming']) expect(isFrameLoadingMode(other)).toBe(false);
  });

  it('describes each frame loading mode differently', () => {
    const hints = new Set((['lazy', 'preload', 'eager'] as const).map(frameLoadingHint));
    expect(hints.size).toBe(3);
  });

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

describe('outdatedTimelineSegments', () => {
  const names: Record<number, string> = { 1: 'Boy', 2: 'Goal' };
  const labelOf = (id: number) => names[id];

  it('has nothing to tint without outdated masks or without a clip', () => {
    expect(outdatedTimelineSegments({}, labelOf, 101)).toEqual([]);
    expect(outdatedTimelineSegments({ '1': [[0, 5]] }, labelOf, 1)).toEqual([]);
    expect(outdatedTimelineSegments({ '1': [[0, 5]] }, labelOf, 0)).toEqual([]);
  });

  it('places a stretch with the same frame-to-percent mapping as the start and end markers', () => {
    const segments = outdatedTimelineSegments({ '1': [[0, 49], [60, 60]] }, labelOf, 101);
    expect(segments).toHaveLength(2);
    expect(segments[0]).toMatchObject({ first: 0, last: 49, leftPercent: 0, widthPercent: 49 });
    // A single frame has no width of its own; the stylesheet gives it a visible minimum.
    expect(segments[1]).toMatchObject({ first: 60, last: 60, leftPercent: 60, widthPercent: 0 });
  });

  it('merges objects into one strip and says which objects each stretch concerns', () => {
    const segments = outdatedTimelineSegments(
      { '1': [[0, 9]], '2': [[5, 14]] },
      labelOf,
      101,
    );
    expect(segments.map(({ first, last, labels }) => [first, last, labels])).toEqual([
      [0, 4, ['Boy']],
      [5, 9, ['Boy', 'Goal']],
      [10, 14, ['Goal']],
    ]);
  });

  it('joins touching runs of the same objects into one stretch', () => {
    const segments = outdatedTimelineSegments({ '1': [[0, 4], [5, 9]] }, labelOf, 101);
    expect(segments.map(({ first, last }) => [first, last])).toEqual([[0, 9]]);
  });

  it('never extends past the last frame', () => {
    const [segment] = outdatedTimelineSegments({ '1': [[8, 20]] }, labelOf, 11);
    expect(segment).toMatchObject({ first: 8, last: 10, leftPercent: 80, widthPercent: 20 });
  });

  it('describes a stretch plainly, as masks that are fine to keep and export', () => {
    const [segment] = outdatedTimelineSegments({ '1': [[0, 49]] }, labelOf, 101);
    expect(segment.tooltip).toBe(
      'Frames 0\u201349: masks for Boy come from an earlier propagation. They are still shown and included in exports.',
    );
    expect(segment.tooltip).not.toMatch(/outdated|invalid|stale|wrong|error/i);
    const [single] = outdatedTimelineSegments({ '1': [[60, 60]], '2': [[60, 60]] }, labelOf, 101);
    expect(single.tooltip).toContain('Frame 60: masks for Boy, Goal come from');
  });

  it('ends the last stretch exactly at the end of the track', () => {
    const [segment] = outdatedTimelineSegments({ '1': [[1, 199]] }, labelOf, 200);
    expect(segment.leftPercent + segment.widthPercent).toBeCloseTo(100, 6);
  });

  it('falls back to a generic name for an object it cannot name', () => {
    const [segment] = outdatedTimelineSegments({ '7': [[0, 3]] }, labelOf, 101);
    expect(segment.labels).toEqual(['Object 7']);
  });
});
