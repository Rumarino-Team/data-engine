import { Injectable } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { BackendService, VideoAddPointsOrBoxRequest } from '../../services/backend.service';
import { Point } from './video-masker-state.store';

export interface ObjectCommandsDeps {
  backend: BackendService;
  getObjects: () => Array<{ id: number; name: string; color: string }>;
  setObjects: (objects: Array<{ id: number; name: string; color: string }>) => void;
  getSelectedObjectId: () => number | null;
  setSelectedObjectId: (value: number | null) => void;
  getMasksMap: () => Map<number, Map<number, boolean[][]>>;
  setMasksMap: (map: Map<number, Map<number, boolean[][]>>) => void;
  getPointsMap: () => Map<number, Map<number, Point[]>>;
  setPointsMap: (map: Map<number, Map<number, Point[]>>) => void;
  getTrackedPoints: () => any[];
  setTrackedPoints: (series: any[]) => void;
  setLiveEditedObjectFrames: (map: Map<number, Set<number>>) => void;
  randomColor: () => string;
  redraw: () => void;
  showError: (title: string, fallbackMessage: string, error: unknown) => void;
}

export interface AddPointCommandDeps {
  backend: BackendService;
  frameIdx: number;
  label: number;
  objId: number;
  x: number;
  y: number;
  expectedEpoch: number;
  displayedFrameIdx: () => number;
  getPointsMap: () => Map<number, Map<number, Point[]>>;
  setPointsMap: (map: Map<number, Map<number, Point[]>>) => void;
  getMasksMap: () => Map<number, Map<number, boolean[][]>>;
  setMasksMap: (map: Map<number, Map<number, boolean[][]>>) => void;
  setPointRequestInFlight: (loading: boolean) => void;
  onBeforeRequest: (requestFrameIdx: number, objId: number) => void;
  getMaskPixelCount: (pixelCounts: Record<number, number> | undefined, objId: number) => number | null;
  setResponseEpoch: (epoch: number) => void;
  setResponseFrame: (frameIdx: number, frameFile: string) => void;
  setMaskDebug: (pixelCount: number | null, fallbackUsed: boolean) => void;
  setDiscardReason: (reason: string | null) => void;
  onEpochMismatch: (responseEpoch: number) => void;
  markObjectAsLiveEdited: (frameIdx: number, objId: number) => void;
  wasLiveEditedBeforeRequest: (frameIdx: number, objId: number) => boolean;
  unmarkObjectAsLiveEdited: (frameIdx: number, objId: number) => void;
  redraw: () => void;
  onFailure: (error: unknown) => void;
}

@Injectable({ providedIn: 'root' })
export class VideoMaskerCommandsService {
  async addPoint(deps: AddPointCommandDeps): Promise<void> {
    const pointsMap = new Map(deps.getPointsMap());
    const framePointsMap = new Map(pointsMap.get(deps.frameIdx) || new Map<number, Point[]>());
    const previousObjectPoints = framePointsMap.get(deps.objId) || [];
    const objectPoints = [...previousObjectPoints, { x: deps.x, y: deps.y, label: deps.label }];
    framePointsMap.set(deps.objId, objectPoints);
    pointsMap.set(deps.frameIdx, framePointsMap);
    deps.setPointsMap(pointsMap);

    const requestFrameIdx = deps.frameIdx;
    deps.onBeforeRequest(requestFrameIdx, deps.objId);

    const request: VideoAddPointsOrBoxRequest = {
      frame_idx: requestFrameIdx,
      obj_id: deps.objId,
      points: objectPoints.map((point) => [point.x, point.y]),
      labels: objectPoints.map((point) => point.label),
      clear_old_points: true,
    };

    const liveEditedBeforeRequest = deps.wasLiveEditedBeforeRequest(deps.frameIdx, deps.objId);

    try {
      deps.setPointRequestInFlight(true);
      const response = await firstValueFrom(deps.backend.addNewPointsOrBox(request));
      if (
        (response as any)?.error ||
        typeof (response as any)?.request_frame_idx !== 'number' ||
        typeof (response as any)?.frame_idx !== 'number' ||
        typeof (response as any)?.frame_file !== 'string' ||
        typeof (response as any)?.state_epoch !== 'number' ||
        !Array.isArray((response as any)?.out_obj_ids) ||
        !Array.isArray((response as any)?.out_masks) ||
        typeof (response as any)?.mask_pixel_counts !== 'object'
      ) {
        throw new Error((response as any)?.error || 'Invalid mask response');
      }

      const responseStateEpoch = Math.trunc(response.state_epoch);
      deps.setResponseEpoch(responseStateEpoch);
      if (responseStateEpoch !== deps.expectedEpoch) {
        const reason = `Discarded stale response due to epoch mismatch (expected ${deps.expectedEpoch}, got ${responseStateEpoch}).`;
        deps.onEpochMismatch(responseStateEpoch);
        deps.setDiscardReason(reason);
        throw new Error(reason);
      }

      const responseRequestFrameIdx = Math.trunc(response.request_frame_idx);
      const responseFrameIdx = Math.trunc(response.frame_idx);
      deps.setResponseFrame(responseFrameIdx, response.frame_file || 'n/a');
      if (responseRequestFrameIdx !== requestFrameIdx || responseFrameIdx !== requestFrameIdx) {
        const reason = `Discarded response due to frame mismatch (request=${requestFrameIdx}, response_request=${responseRequestFrameIdx}, response_frame=${responseFrameIdx}).`;
        deps.setDiscardReason(reason);
        throw new Error(reason);
      }

      if (deps.displayedFrameIdx() !== requestFrameIdx) {
        const reason = `Discarded response because displayed frame moved from ${requestFrameIdx} to ${deps.displayedFrameIdx()}.`;
        deps.setDiscardReason(reason);
        throw new Error(reason);
      }

      const maskPixelCount = deps.getMaskPixelCount(response.mask_pixel_counts, deps.objId);
      deps.setMaskDebug(maskPixelCount, Boolean(response.single_frame_fallback_used));

      const masksMap = new Map(deps.getMasksMap());
      const frameMasksMap = new Map(masksMap.get(requestFrameIdx) || new Map<number, boolean[][]>());
      response.out_obj_ids.forEach((id, index) => {
        frameMasksMap.set(id, response.out_masks[index]);
      });
      masksMap.set(requestFrameIdx, frameMasksMap);
      deps.markObjectAsLiveEdited(requestFrameIdx, deps.objId);
      deps.setMasksMap(masksMap);
      deps.redraw();
    } catch (error) {
      deps.onFailure(error);

      const rollbackPointsMap = new Map(deps.getPointsMap());
      const rollbackFramePointsMap = new Map(
        rollbackPointsMap.get(deps.frameIdx) || new Map<number, Point[]>(),
      );
      if (previousObjectPoints.length > 0) {
        rollbackFramePointsMap.set(deps.objId, previousObjectPoints);
      } else {
        rollbackFramePointsMap.delete(deps.objId);
      }
      if (rollbackFramePointsMap.size === 0) {
        rollbackPointsMap.delete(deps.frameIdx);
      } else {
        rollbackPointsMap.set(deps.frameIdx, rollbackFramePointsMap);
      }
      if (!liveEditedBeforeRequest) {
        deps.unmarkObjectAsLiveEdited(deps.frameIdx, deps.objId);
      }
      deps.setPointsMap(rollbackPointsMap);
      deps.redraw();
    } finally {
      deps.setPointRequestInFlight(false);
    }
  }

  addObject(deps: ObjectCommandsDeps): void {
    const newId = deps.getObjects().length + 1;
    deps.setObjects([...deps.getObjects(), { id: newId, name: `Object ${newId}`, color: deps.randomColor() }]);
    deps.setSelectedObjectId(newId);
  }

  removeObject(deps: ObjectCommandsDeps): void {
    const id = deps.getSelectedObjectId();
    if (id === null) return;

    deps.backend.removeObject(id).subscribe({
      next: () => {
        deps.setObjects(deps.getObjects().filter((entry) => entry.id !== id));
        this.removeObjectFromFrameMaps(id, deps);
        const nextObjects = deps.getObjects();
        deps.setSelectedObjectId(nextObjects.length > 0 ? nextObjects[0].id : null);
        deps.redraw();
      },
      error: (error) => deps.showError('Remove failed', 'Failed to remove object.', error),
    });
  }

  async removeAllObjects(deps: ObjectCommandsDeps): Promise<void> {
    const objectIds = deps.getObjects().map((entry) => entry.id);
    if (objectIds.length === 0) return;

    try {
      await Promise.all(objectIds.map((id) => firstValueFrom(deps.backend.removeObject(id))));
      deps.setObjects([]);
      deps.setSelectedObjectId(null);
      deps.setMasksMap(new Map());
      deps.setPointsMap(new Map());
      deps.setLiveEditedObjectFrames(new Map());
      deps.setTrackedPoints([]);
      deps.redraw();
    } catch (error) {
      deps.showError('Remove all failed', 'Failed to remove all objects.', error);
    }
  }

  private removeObjectFromFrameMaps(objectId: number, deps: ObjectCommandsDeps): void {
    const nextMasks = new Map(deps.getMasksMap());
    nextMasks.forEach((frameMap) => frameMap.delete(objectId));
    deps.setMasksMap(nextMasks);

    const nextPoints = new Map(deps.getPointsMap());
    nextPoints.forEach((frameMap) => frameMap.delete(objectId));
    deps.setPointsMap(nextPoints);

    deps.setTrackedPoints(deps.getTrackedPoints().filter((series: any) => series.obj_id !== objectId));
  }
}
