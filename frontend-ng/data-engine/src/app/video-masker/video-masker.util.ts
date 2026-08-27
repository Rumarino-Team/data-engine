import {
  InteractiveMaskRle,
  InteractiveObject,
  InteractivePoint,
  VideoSaveInteractiveState,
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

// --- mask geometry helpers (from VideoMaskerRenderingService) ---

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

export function maskHasForeground(mask: boolean[][]): boolean {
  for (const row of mask) {
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
  mask: boolean[][],
): { width: number; height: number; counts: number[] } | null {
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

export function deserializeLiveMasks(
  liveMasks: InteractiveMaskRle[],
): { masks: Map<number, Map<number, boolean[][]>>; liveEditedFrames: Map<number, Set<number>> } {
  const masksByFrame = new Map<number, Map<number, boolean[][]>>();
  const editedByFrame = new Map<number, Set<number>>();
  for (const entry of liveMasks) {
    const decodedMask = decodeMaskFromCounts(entry);
    if (!decodedMask) {
      continue;
    }
    const frameIdx = Math.trunc(entry.frame_idx);
    const objId = Math.trunc(entry.obj_id);
    if (frameIdx < 0 || objId <= 0) {
      continue;
    }
    let frameMasks = masksByFrame.get(frameIdx);
    if (!frameMasks) {
      frameMasks = new Map<number, boolean[][]>();
      masksByFrame.set(frameIdx, frameMasks);
    }
    frameMasks.set(objId, decodedMask);

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
  masksByFrame: Map<number, Map<number, boolean[][]>>;
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
