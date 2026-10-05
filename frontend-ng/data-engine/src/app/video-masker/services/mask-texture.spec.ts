import { vi } from 'vitest';
import { createMaskTexture } from './mask-texture';

describe('mask textures', () => {
  const putImageData = vi.fn();
  beforeEach(() => {
    putImageData.mockClear();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      createImageData: (width: number, height: number) => ({
        data: new Uint8ClampedArray(width * height * 4),
      }),
      putImageData,
    } as unknown as CanvasRenderingContext2D);
  });
  afterEach(() => vi.restoreAllMocks());

  it('produces identical transparent textures from live masks and manifest RLE', () => {
    const texture = createMaskTexture({
      objectId: 1,
      color: '#ff8800',
      source: [
        [true, false],
        [false, true],
      ],
    });
    const expected = new Uint8ClampedArray([
      255, 136, 0, 120, 0, 0, 0, 0, 0, 0, 0, 0, 255, 136, 0, 120,
    ]);
    expect(texture?.width).toBe(2);
    expect(texture?.height).toBe(2);
    expect(putImageData.mock.lastCall?.[0].data).toEqual(expected);
    createMaskTexture({
      objectId: 1,
      color: '#ff8800',
      source: {
        size: [2, 2],
        rle: [
          [0, 1],
          [3, 1],
        ],
        bbox: [0, 0, 2, 2],
      },
    });
    expect(putImageData.mock.lastCall?.[0].data).toEqual(expected);
  });

  it('rejects invalid dimensions and clips runs to the texture bounds', () => {
    expect(
      createMaskTexture({
        objectId: 1,
        color: '#ffffff',
        source: { size: [-1, 2], rle: [], bbox: [0, 0, 0, 0] },
      }),
    ).toBeNull();
    createMaskTexture({
      objectId: 1,
      color: '#ffffff',
      source: { size: [1, 2], rle: [[1, 100]], bbox: [0, 0, 2, 1] },
    });
    expect(putImageData.mock.lastCall?.[0].data).toEqual(
      new Uint8ClampedArray([0, 0, 0, 0, 255, 255, 255, 120]),
    );
  });

  it('clips negative and overlapping runs and ignores malformed or out-of-bounds runs', () => {
    createMaskTexture({
      objectId: 1,
      color: '#123456',
      source: {
        size: [2, 3],
        rle: [
          [-2, 3],
          [0, 2],
          [4, 20],
          [100, 1],
          [1.5, 1],
          [2, 0],
          [2, -1],
          [2, NaN],
        ],
        bbox: [0, 0, 3, 2],
      },
    });
    expect(putImageData.mock.lastCall?.[0].data).toEqual(
      new Uint8ClampedArray([
        18, 52, 86, 120, 18, 52, 86, 120, 0, 0, 0, 0, 0, 0, 0, 0, 18, 52, 86, 120, 18, 52, 86, 120,
      ]),
    );
  });
});
