import { LiveMask, PackedBitMask, VideoMaskObjectData } from '../services/backend.service';
import { LoadSourceMode } from './state/video-masker-ui.types';

// --- frame cache / index helpers (from FrameRendererService) ---

export function evictWithLimit<T>(
  cache: Map<number, T>,
  maxSize: number,
  onEvict?: (key: number) => void,
): void {
  while (cache.size > maxSize) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey === undefined) {
      break;
    }
    cache.delete(oldestKey);
    onEvict?.(oldestKey);
  }
}

export function clampFrameIndex(value: number, maxFrame: number): number {
  return Math.min(Math.max(Math.trunc(value), 0), Math.max(maxFrame, 0));
}

// --- epoch / live-edit helpers (from MaskStateService) ---

export function resolveStateEpoch(
  currentEpoch: number,
  nextEpoch: number | undefined,
): { normalizedEpoch: number; shouldClearLiveState: boolean } | null {
  if (typeof nextEpoch !== 'number' || !Number.isFinite(nextEpoch)) {
    return null;
  }
  const normalizedEpoch = Math.trunc(nextEpoch);
  if (normalizedEpoch <= 0) {
    return null;
  }
  return {
    normalizedEpoch,
    shouldClearLiveState: currentEpoch !== 0 && currentEpoch !== normalizedEpoch,
  };
}

export function normalizeMask2d(mask: unknown): boolean[][] | null {
  let candidate: unknown = mask;
  while (
    Array.isArray(candidate) &&
    candidate.length > 0 &&
    Array.isArray(candidate[0]) &&
    Array.isArray((candidate[0] as unknown[])[0])
  ) {
    candidate = candidate[0];
  }
  if (!Array.isArray(candidate) || candidate.length === 0 || !Array.isArray(candidate[0])) {
    return null;
  }
  return candidate as boolean[][];
}

export function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : [0, 0, 0];
}

// --- load-source copy helpers (from VideoSessionService) ---

export function loadPathPlaceholder(mode: LoadSourceMode): string {
  switch (mode) {
    case 'video_file':
      return 'Enter video file path (.mp4, .mov, .avi, .mkv, .webm, .m4v)';
    case 'saved_session_dir':
      return 'Enter saved session directory path (contains session.json, frames/, and masks/)';
    default:
      return 'Enter frames directory path';
  }
}

export function browseLabel(mode: LoadSourceMode): string {
  switch (mode) {
    case 'video_file':
      return 'Browse Video';
    case 'saved_session_dir':
      return 'Browse Saved Session';
    default:
      return 'Browse Frames';
  }
}

export function loadModeHint(mode: LoadSourceMode): string {
  if (mode === 'saved_session_dir') {
    return 'Choose a saved session directory containing session.json, frames/, and masks/.';
  }
  if (mode === 'video_file') {
    return 'Choose a video file; backend will extract frames and create a new session.';
  }
  return 'Choose a directory that already contains extracted video frames.';
}

// --- misc shared helpers ---

export function getMaskPixelCount(
  pixelCounts: Record<number, number> | undefined,
  objId: number,
): number | null {
  if (!pixelCounts) {
    return null;
  }
  const direct = (pixelCounts as Record<number, number>)[objId];
  if (typeof direct === 'number' && Number.isFinite(direct)) {
    return Math.trunc(direct);
  }
  const stringLookup = (pixelCounts as unknown as Record<string, number>)[String(objId)];
  if (typeof stringLookup === 'number' && Number.isFinite(stringLookup)) {
    return Math.trunc(stringLookup);
  }
  return null;
}

export function getErrorMessage(error: any, fallback: string): string {
  return error?.error?.detail || error?.message || fallback;
}

const packedBytesCache = new WeakMap<PackedBitMask, Uint8Array>();

/** Validate and decode once; keep packed bytes rather than allocating a boolean grid. */
export function packedMaskBytes(mask: PackedBitMask): Uint8Array {
  const cached = packedBytesCache.get(mask);
  if (cached) return cached;
  if (
    mask.encoding !== 'packed-bits' ||
    !Array.isArray(mask.size) ||
    mask.size.length !== 2 ||
    !mask.size.every((n) => Number.isSafeInteger(n) && n > 0) ||
    !Number.isSafeInteger(mask.size[0] * mask.size[1]) ||
    typeof mask.data !== 'string'
  )
    throw new Error('Invalid packed mask dimensions.');
  const total = mask.size[0] * mask.size[1];
  const length = Math.ceil(total / 8);
  if (
    mask.data.length !== 4 * Math.ceil(length / 3) ||
    !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(mask.data)
  )
    throw new Error('Invalid packed mask base64.');
  const raw = atob(mask.data);
  if (raw.length !== length || btoa(raw) !== mask.data)
    throw new Error('Invalid packed mask byte length.');
  const bytes = Uint8Array.from(raw, (char) => char.charCodeAt(0));
  if (total % 8 && bytes[length - 1] & ((1 << (8 - (total % 8))) - 1))
    throw new Error('Nonzero packed mask padding.');
  packedBytesCache.set(mask, bytes);
  return bytes;
}

export function encodedMaskPixelCount(mask: Exclude<LiveMask, boolean[][]>): number {
  if (!('encoding' in mask)) {
    if (!isMaskRle(mask)) throw new Error('Invalid RLE mask response.');
    return mask.rle.reduce((sum, [, length]) => sum + length, 0);
  }
  let count = 0;
  for (let byte of packedMaskBytes(mask)) {
    while (byte) {
      byte &= byte - 1;
      count++;
    }
  }
  return count;
}

/** Validate foreground runs before trusting an API response or saving a snapshot. */
export function isMaskRle(value: unknown): value is VideoMaskObjectData {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const mask = value as VideoMaskObjectData;
  if (
    !Array.isArray(mask.size) ||
    mask.size.length !== 2 ||
    !mask.size.every((size) => Number.isSafeInteger(size) && size > 0) ||
    !Number.isSafeInteger(mask.size[0] * mask.size[1]) ||
    !Array.isArray(mask.rle)
  )
    return false;
  let end = 0;
  for (const run of mask.rle) {
    if (
      !Array.isArray(run) ||
      run.length !== 2 ||
      !run.every(Number.isSafeInteger) ||
      run[0] < end ||
      run[1] <= 0 ||
      run[0] + run[1] > mask.size[0] * mask.size[1]
    )
      return false;
    end = run[0] + run[1];
  }
  return (
    Array.isArray(mask.bbox) &&
    mask.bbox.length === 4 &&
    mask.bbox.every((value) => Number.isSafeInteger(value) && value >= 0) &&
    mask.bbox[0] + mask.bbox[2] <= mask.size[1] &&
    mask.bbox[1] + mask.bbox[3] <= mask.size[0]
  );
}
