import {
  InteractiveMaskRle,
  InteractiveObject,
  InteractivePoint,
  VideoSaveInteractiveState,
  LiveMask,
  PackedBitMask,
  VideoMaskObjectData,
} from '../services/backend.service';
import { LoadSourceMode } from './state/video-masker-ui.types';
import { MaskObject, Point } from './services/video-masker-state.store';

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

export function markObjectLiveEdited(
  liveEditedObjectFrames: Map<number, Set<number>>,
  frameIdx: number,
  objId: number,
): Map<number, Set<number>> {
  const next = new Map(liveEditedObjectFrames);
  const existing = next.get(frameIdx);
  const nextSet = existing ? new Set(existing) : new Set<number>();
  nextSet.add(objId);
  next.set(frameIdx, nextSet);
  return next;
}

export function unmarkObjectLiveEdited(
  liveEditedObjectFrames: Map<number, Set<number>>,
  frameIdx: number,
  objId: number,
): Map<number, Set<number>> {
  const next = new Map(liveEditedObjectFrames);
  const existing = next.get(frameIdx);
  if (!existing) {
    return next;
  }
  const nextSet = new Set(existing);
  nextSet.delete(objId);
  if (nextSet.size === 0) {
    next.delete(frameIdx);
  } else {
    next.set(frameIdx, nextSet);
  }
  return next;
}

export function isObjectLiveEdited(
  liveEditedObjectFrames: Map<number, Set<number>>,
  frameIdx: number,
  objId: number,
): boolean {
  return Boolean(liveEditedObjectFrames.get(frameIdx)?.has(objId));
}

// --- mask geometry helpers ---

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

export function maskHasForeground(mask: LiveMask): boolean {
  if (!Array.isArray(mask)) {
    if ('encoding' in mask) return packedMaskBytes(mask).some((byte) => byte !== 0);
    return mask.rle.some(([, length]) => length > 0);
  }
  for (const row of normalizeMask2d(mask) ?? []) {
    for (const value of row) {
      if (value) {
        return true;
      }
    }
  }
  return false;
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

export function randomColor(): string {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}

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
  return error?.error?.detail || error?.error?.error || error?.message || fallback;
}

export function encodeMaskToCounts(
  mask: LiveMask,
): { width: number; height: number; counts: number[] } | null {
  if (!Array.isArray(mask)) {
    if ('encoding' in mask) {
      let bytes: Uint8Array;
      try {
        bytes = packedMaskBytes(mask);
      } catch {
        return null;
      }
      const [height, width] = mask.size;
      const counts: number[] = [];
      let current = false,
        length = 0;
      for (let i = 0; i < height * width; i++) {
        const value = (bytes[i >> 3] & (128 >> (i & 7))) !== 0;
        if (value !== current) {
          counts.push(length);
          length = 0;
          current = value;
        }
        length++;
      }
      counts.push(length);
      return { height, width, counts };
    }
    if (!isMaskRle(mask)) return null;
    const [height, width] = mask.size;
    const counts: number[] = [];
    let cursor = 0;
    for (const [start, length] of mask.rle) {
      counts.push(start - cursor, length);
      cursor = start + length;
    }
    if (cursor < width * height) counts.push(width * height - cursor);
    return { height, width, counts };
  }
  const normalizedMask = normalizeMask2d(mask);
  if (!normalizedMask || normalizedMask.length === 0 || normalizedMask[0].length === 0) {
    return null;
  }
  const height = normalizedMask.length;
  const width = normalizedMask[0].length;
  const counts: number[] = [];
  let currentValue = false;
  let currentRun = 0;
  for (let y = 0; y < height; y++) {
    const row = normalizedMask[y];
    if (!Array.isArray(row) || row.length !== width) {
      return null;
    }
    for (let x = 0; x < width; x++) {
      const value = Boolean(row[x]);
      if (value === currentValue) {
        currentRun += 1;
        continue;
      }
      counts.push(currentRun);
      currentRun = 1;
      currentValue = value;
    }
  }
  counts.push(currentRun);
  return { width, height, counts };
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

/** Convert saved alternating counts to foreground runs without allocating a pixel grid. */
export function maskFromCounts(entry: InteractiveMaskRle): VideoMaskObjectData | null {
  const { height, width, counts } = entry;
  if (
    !Number.isSafeInteger(height) ||
    !Number.isSafeInteger(width) ||
    height <= 0 ||
    width <= 0 ||
    !Number.isSafeInteger(width * height) ||
    !Array.isArray(counts) ||
    !counts.length
  )
    return null;
  const rle: number[][] = [];
  let cursor = 0;
  let foreground = false;
  let xMin = width,
    yMin = height,
    xMax = -1,
    yMax = -1;
  for (const count of counts) {
    if (!Number.isSafeInteger(count) || count < 0 || cursor + count > width * height) return null;
    if (foreground && count) {
      rle.push([cursor, count]);
      const firstRow = Math.floor(cursor / width);
      const lastRow = Math.floor((cursor + count - 1) / width);
      xMin = Math.min(xMin, firstRow === lastRow ? cursor % width : 0);
      xMax = Math.max(xMax, firstRow === lastRow ? (cursor + count - 1) % width : width - 1);
      yMin = Math.min(yMin, firstRow);
      yMax = Math.max(yMax, lastRow);
    }
    cursor += count;
    foreground = !foreground;
  }
  if (cursor !== width * height) return null;
  return {
    size: [height, width],
    rle,
    bbox: rle.length ? [xMin, yMin, xMax - xMin + 1, yMax - yMin + 1] : [0, 0, 0, 0],
  };
}

export function decodeMaskFromCounts(maskRle: InteractiveMaskRle): boolean[][] | null {
  const height = Math.trunc(maskRle.height);
  const width = Math.trunc(maskRle.width);
  if (height <= 0 || width <= 0) {
    return null;
  }
  if (!Array.isArray(maskRle.counts) || maskRle.counts.length === 0) {
    return null;
  }
  const totalPixels = width * height;
  const flatMask = new Array<boolean>(totalPixels).fill(false);
  let index = 0;
  let foreground = false;
  for (const rawCount of maskRle.counts) {
    const count = Math.trunc(rawCount);
    if (!Number.isFinite(count) || count < 0) {
      return null;
    }
    const end = index + count;
    if (end > totalPixels) {
      return null;
    }
    if (foreground) {
      for (let cursor = index; cursor < end; cursor++) {
        flatMask[cursor] = true;
      }
    }
    index = end;
    foreground = !foreground;
  }
  if (index !== totalPixels) {
    return null;
  }
  const mask2d: boolean[][] = [];
  for (let y = 0; y < height; y++) {
    const rowStart = y * width;
    mask2d.push(flatMask.slice(rowStart, rowStart + width));
  }
  return mask2d;
}

// --- interactive session (de)serialization (from VideoMaskerSessionStateService) ---

function normalizeObjectColor(color: string | undefined): string {
  if (typeof color === 'string' && /^#[0-9a-fA-F]{6}$/.test(color.trim())) {
    return color.trim();
  }
  return randomColor();
}

export function normalizeInteractiveObjects(objects: InteractiveObject[]): MaskObject[] {
  const normalized: MaskObject[] = [];
  const seen = new Set<number>();
  for (const candidate of objects) {
    if (!Number.isInteger(candidate.id) || candidate.id <= 0 || seen.has(candidate.id)) {
      continue;
    }
    seen.add(candidate.id);
    normalized.push({
      id: candidate.id,
      name: (candidate.name || '').trim() || `Object ${candidate.id}`,
      color: normalizeObjectColor(candidate.color),
    });
  }
  return normalized;
}

export function deserializePoints(points: InteractivePoint[]): Map<number, Map<number, Point[]>> {
  const pointsByFrame = new Map<number, Map<number, Point[]>>();
  for (const point of points) {
    if (
      !Number.isFinite(point.frame_idx) ||
      !Number.isFinite(point.obj_id) ||
      !Number.isFinite(point.x) ||
      !Number.isFinite(point.y) ||
      (point.label !== 0 && point.label !== 1)
    ) {
      continue;
    }
    const frameIdx = Math.trunc(point.frame_idx);
    const objId = Math.trunc(point.obj_id);
    if (frameIdx < 0 || objId <= 0) {
      continue;
    }
    let frameMap = pointsByFrame.get(frameIdx);
    if (!frameMap) {
      frameMap = new Map<number, Point[]>();
      pointsByFrame.set(frameIdx, frameMap);
    }
    const objPoints = frameMap.get(objId) || [];
    objPoints.push({ x: point.x, y: point.y, label: point.label });
    frameMap.set(objId, objPoints);
  }
  return pointsByFrame;
}

export function deserializeLiveMasks(liveMasks: InteractiveMaskRle[]): {
  masks: Map<number, Map<number, LiveMask>>;
  liveEditedFrames: Map<number, Set<number>>;
} {
  const masksByFrame = new Map<number, Map<number, LiveMask>>();
  const editedByFrame = new Map<number, Set<number>>();
  for (const entry of liveMasks) {
    const encodedMask = maskFromCounts(entry);
    if (!encodedMask) {
      continue;
    }
    const frameIdx = Math.trunc(entry.frame_idx);
    const objId = Math.trunc(entry.obj_id);
    if (frameIdx < 0 || objId <= 0) {
      continue;
    }
    let frameMasks = masksByFrame.get(frameIdx);
    if (!frameMasks) {
      frameMasks = new Map<number, LiveMask>();
      masksByFrame.set(frameIdx, frameMasks);
    }
    frameMasks.set(objId, encodedMask);

    const editedSet = editedByFrame.get(frameIdx) || new Set<number>();
    editedSet.add(objId);
    editedByFrame.set(frameIdx, editedSet);
  }
  return { masks: masksByFrame, liveEditedFrames: editedByFrame };
}

export function buildInteractiveStateSnapshot(args: {
  objects: MaskObject[];
  selectedObjectId: number | null;
  interactionMode: 'positive' | 'negative';
  currentFrameIdx: number;
  pointsByFrame: Map<number, Map<number, Point[]>>;
  masksByFrame: Map<number, Map<number, LiveMask>>;
}): VideoSaveInteractiveState {
  const objects = args.objects.map((entry) => ({
    id: entry.id,
    name: entry.name,
    color: entry.color,
  }));
  const points: InteractivePoint[] = [];
  args.pointsByFrame.forEach((framePoints, frameIdx) => {
    framePoints.forEach((objPoints, objId) => {
      for (const point of objPoints) {
        points.push({
          frame_idx: frameIdx,
          obj_id: objId,
          x: point.x,
          y: point.y,
          label: point.label === 1 ? 1 : 0,
        });
      }
    });
  });

  const liveMasks: InteractiveMaskRle[] = [];
  args.masksByFrame.forEach((frameMasks, frameIdx) => {
    frameMasks.forEach((mask, objId) => {
      const encoded = encodeMaskToCounts(mask);
      if (!encoded) return;
      liveMasks.push({
        frame_idx: frameIdx,
        obj_id: objId,
        height: encoded.height,
        width: encoded.width,
        counts: encoded.counts,
      });
    });
  });

  return {
    version: 1,
    objects,
    selected_object_id: args.selectedObjectId,
    interaction_mode: args.interactionMode,
    current_frame_idx: args.currentFrameIdx,
    points,
    live_masks: liveMasks,
  };
}
