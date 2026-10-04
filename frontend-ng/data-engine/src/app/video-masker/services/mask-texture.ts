import { VideoMaskObjectData } from '../../services/backend-api.types';
import { hexToRgb, normalizeMask2d } from '../video-masker.util';

export interface MaskOverlay {
  objectId: number;
  color: string;
  source: boolean[][] | VideoMaskObjectData;
}

/** Converts backend mask pixels to a native-resolution texture for a Konva Image.
 * No frame composition, scaling, or annotation drawing happens here.
 */
export function createMaskTexture({ source, color }: MaskOverlay): HTMLCanvasElement | null {
  const mask = Array.isArray(source) ? normalizeMask2d(source) : null;
  const runs = Array.isArray(source) ? null : source;
  const width = mask ? mask[0]?.length : runs?.size?.[1];
  const height = mask ? mask.length : runs?.size?.[0];
  if (
    !width ||
    !height ||
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 0 ||
    height < 0
  )
    return null;
  const texture = document.createElement('canvas');
  texture.width = width;
  texture.height = height;
  const context = texture.getContext('2d');
  if (!context) return null;
  const pixels = context.createImageData(width, height);
  const rgba = [...hexToRgb(color), 120];
  if (mask) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (mask[y][x]) pixels.data.set(rgba, (y * width + x) * 4);
      }
    }
  } else {
    for (const [start, length] of runs?.rle ?? []) {
      if (!Number.isInteger(start) || !Number.isInteger(length) || length <= 0) continue;
      for (
        let index = Math.max(0, start);
        index < Math.min(width * height, start + length);
        index++
      ) {
        pixels.data.set(rgba, index * 4);
      }
    }
  }
  context.putImageData(pixels, 0, 0);
  return texture;
}
