import { Injectable } from '@angular/core';
import {
  InteractiveMaskRle,
  InteractiveObject,
  InteractivePoint,
  VideoSaveInteractiveState,
} from '../../services/backend.service';
import { MaskObject, Point } from './video-masker-state.store';

@Injectable({ providedIn: 'root' })
export class VideoMaskerSessionStateService {
  normalizeInteractiveObjects(
    objects: InteractiveObject[],
    randomColor: () => string,
  ): MaskObject[] {
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
        color: this.normalizeObjectColor(candidate.color, randomColor),
      });
    }
    return normalized;
  }

  deserializePoints(points: InteractivePoint[]): Map<number, Map<number, Point[]>> {
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

  deserializeLiveMasks(
    liveMasks: InteractiveMaskRle[],
    decodeMaskFromCounts: (maskRle: InteractiveMaskRle) => boolean[][] | null,
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

  buildInteractiveStateSnapshot(args: {
    objects: MaskObject[];
    selectedObjectId: number | null;
    interactionMode: 'positive' | 'negative';
    currentFrameIdx: number;
    pointsByFrame: Map<number, Map<number, Point[]>>;
    masksByFrame: Map<number, Map<number, boolean[][]>>;
    encodeMaskToCounts: (
      mask: boolean[][],
    ) => { width: number; height: number; counts: number[] } | null;
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
        const encoded = args.encodeMaskToCounts(mask);
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

  private normalizeObjectColor(color: string | undefined, randomColor: () => string): string {
    if (typeof color === 'string' && /^#[0-9a-fA-F]{6}$/.test(color.trim())) {
      return color.trim();
    }
    return randomColor();
  }
}
