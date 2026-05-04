import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class MaskStateService {
  updateStateEpoch(
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

  markObjectAsLiveEdited(
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

  unmarkObjectAsLiveEdited(
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

  isObjectLiveEdited(
    liveEditedObjectFrames: Map<number, Set<number>>,
    frameIdx: number,
    objId: number,
  ): boolean {
    return Boolean(liveEditedObjectFrames.get(frameIdx)?.has(objId));
  }
}
