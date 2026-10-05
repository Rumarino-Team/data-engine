import { LiveMask } from '../../services/backend-api.types';
import { hexToRgb, normalizeMask2d, packedMaskBytes } from '../video-masker.util';

export interface MaskOverlay {
  objectId: number;
  color: string;
  source: LiveMask;
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
  // Read the RGBA bytes in native byte order, then write a whole pixel at once.
  // RLE runs can use the typed array's bulk fill instead of four writes per pixel.
  const rgba = new Uint8Array([...hexToRgb(color), 120]);
  const packedColor = new Uint32Array(rgba.buffer)[0];
  const packedPixels = new Uint32Array(
    pixels.data.buffer,
    pixels.data.byteOffset,
    pixels.data.byteLength / 4,
  );
  if (mask) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (mask[y][x]) packedPixels[y * width + x] = packedColor;
      }
    }
  } else if (runs && 'encoding' in runs) {
    const bytes = packedMaskBytes(runs);
    for (let i = 0; i < width * height; i++) {
      if (bytes[i >> 3] & (128 >> (i & 7))) packedPixels[i] = packedColor;
    }
  } else {
    for (const [start, length] of runs?.rle ?? []) {
      if (!Number.isInteger(start) || !Number.isInteger(length) || length <= 0) continue;
      const firstPixel = Math.max(0, start);
      const endPixel = Math.min(width * height, start + length);
      if (firstPixel < endPixel) packedPixels.fill(packedColor, firstPixel, endPixel);
    }
  }
  context.putImageData(pixels, 0, 0);
  return texture;
}
