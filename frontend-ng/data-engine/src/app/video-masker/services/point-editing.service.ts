import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import {
  BackendService,
  LiveMask,
  VideoAddPointsOrBoxRequest,
} from '../../services/backend.service';
import { VideoMaskerStateStore, Point } from './video-masker-state.store';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import {
  getErrorMessage,
  getMaskPixelCount,
  isObjectLiveEdited,
  encodedMaskPixelCount,
  markObjectLiveEdited,
  unmarkObjectLiveEdited,
} from '../video-masker.util';

/** Applies prompt edits and validates responses before committing or rolling back. */
@Injectable()
export class PointEditingService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly framePipeline = inject(FramePipelineService);
  private readonly toast = inject(ToastService);
  private get isBusy(): boolean {
    return this.store.isInteractionBusy();
  }

  private markObjectAsLiveEdited(frameIdx: number, objId: number): void {
    this.store.liveEditedObjectFrames.set(
      markObjectLiveEdited(this.store.liveEditedObjectFrames(), frameIdx, objId),
    );
  }

  private unmarkObjectAsLiveEdited(frameIdx: number, objId: number): void {
    this.store.liveEditedObjectFrames.set(
      unmarkObjectLiveEdited(this.store.liveEditedObjectFrames(), frameIdx, objId),
    );
  }

  private isObjectLiveEdited(frameIdx: number, objId: number): boolean {
    return isObjectLiveEdited(this.store.liveEditedObjectFrames(), frameIdx, objId);
  }

  private showError(title: string, fallbackMessage: string, error: unknown): void {
    console.error(error);
    this.toast.show('error', title, getErrorMessage(error, fallbackMessage));
  }

  async addPoint(x: number, y: number, label: number, frameIdx: number): Promise<void> {
    if (this.isBusy) return;
    const objId = this.store.selectedObjectId();
    if (objId === null || !this.store.objects().some((object) => object.id === objId)) return;

    const previousObjectPoints = this.store.points().get(frameIdx)?.get(objId) || [];
    await this.submitObjectPoints(objId, frameIdx, previousObjectPoints, [
      ...previousObjectPoints,
      { x, y, label },
    ]);
  }

  async removePoint(objId: number, frameIdx: number, pointIndex: number): Promise<void> {
    if (this.isBusy) return;
    const previousObjectPoints = this.store.points().get(frameIdx)?.get(objId) || [];
    if (pointIndex < 0 || pointIndex >= previousObjectPoints.length) return;
    const objectPoints = previousObjectPoints.filter((_point, index) => index !== pointIndex);
    if (objectPoints.length > 0) {
      await this.submitObjectPoints(objId, frameIdx, previousObjectPoints, objectPoints, false);
      return;
    }

    this.store.isPointRequestInFlight.set(true);
    try {
      const response = await firstValueFrom(this.backend.clearAllPromptsInFrame(frameIdx, objId));
      if (response?.error) throw new Error(response.error);
      this.setObjectPoints(frameIdx, objId, []);
      const masksMap = new Map(this.store.masks());
      const frameMasksMap = new Map(masksMap.get(frameIdx) || new Map<number, LiveMask>());
      frameMasksMap.delete(objId);
      if (frameMasksMap.size) masksMap.set(frameIdx, frameMasksMap);
      else masksMap.delete(frameIdx);
      this.store.masks.set(masksMap);
      this.unmarkObjectAsLiveEdited(frameIdx, objId);
      this.store.invalidateTracking();
      this.framePipeline.redraw();
    } catch (error) {
      this.showError('Point removal failed', 'Unable to remove point.', error);
    } finally {
      this.store.isPointRequestInFlight.set(false);
    }
  }

  private setObjectPoints(frameIdx: number, objId: number, objectPoints: Point[]): void {
    const pointsMap = new Map(this.store.points());
    const framePointsMap = new Map(pointsMap.get(frameIdx) || new Map<number, Point[]>());
    if (objectPoints.length > 0) framePointsMap.set(objId, objectPoints);
    else framePointsMap.delete(objId);
    if (framePointsMap.size > 0) pointsMap.set(frameIdx, framePointsMap);
    else pointsMap.delete(frameIdx);
    this.store.points.set(pointsMap);
  }

  private async submitObjectPoints(
    objId: number,
    frameIdx: number,
    previousObjectPoints: Point[],
    objectPoints: Point[],
    requireDisplayedFrame = true,
  ): Promise<void> {
    const expectedEpoch = this.store.stateEpoch();
    this.setObjectPoints(frameIdx, objId, objectPoints);

    const requestFrameIdx = frameIdx;
    this.store.lastClickRequestFrameIdx.set(requestFrameIdx);
    this.store.lastDebugObjectId.set(objId);
    this.store.lastMaskPixelCount.set(null);
    this.store.lastBackendResponseFrameIdx.set(null);
    this.store.lastBackendResponseFrameFile.set('n/a');
    this.store.lastBackendResponseStateEpoch.set(null);
    this.store.lastFallbackUsed.set(false);
    this.store.lastDiscardReason.set(null);

    const request: VideoAddPointsOrBoxRequest = {
      frame_idx: requestFrameIdx,
      obj_id: objId,
      points: objectPoints.map((point) => [point.x, point.y]),
      labels: objectPoints.map((point) => point.label),
      clear_old_points: true,
    };

    const liveEditedBeforeRequest = this.isObjectLiveEdited(frameIdx, objId);
    let responseChangedEpoch = false;
    const ownsPointUpdate = () =>
      this.store.stateEpoch() === expectedEpoch &&
      this.store.points().get(frameIdx)?.get(objId) === objectPoints &&
      this.store.objects().some((object) => object.id === objId);

    try {
      this.store.isPointRequestInFlight.set(true);
      const response = await firstValueFrom(this.backend.addNewPointsOrBox(request));
      if (!ownsPointUpdate()) {
        this.store.lastDiscardReason.set(
          'Discarded point response because the editing state changed.',
        );
        return;
      }
      if (
        (response as any)?.error ||
        typeof (response as any)?.request_frame_idx !== 'number' ||
        typeof (response as any)?.frame_idx !== 'number' ||
        typeof (response as any)?.frame_file !== 'string' ||
        !Number.isInteger(response?.state_epoch) ||
        !Array.isArray((response as any)?.out_obj_ids) ||
        !Array.isArray((response as any)?.out_masks) ||
        typeof (response as any)?.mask_pixel_counts !== 'object'
      ) {
        throw new Error((response as any)?.error || 'Invalid mask response');
      }

      const responseStateEpoch = Math.trunc(response.state_epoch);
      this.store.lastBackendResponseStateEpoch.set(responseStateEpoch);
      if (responseStateEpoch !== expectedEpoch) {
        const reason = `Discarded stale response due to epoch mismatch (expected ${expectedEpoch}, got ${responseStateEpoch}).`;
        if (responseStateEpoch > expectedEpoch) {
          this.store.updateStateEpoch(
            responseStateEpoch,
            'add_new_points_or_box mismatch response',
          );
          responseChangedEpoch = true;
        }
        this.store.lastDiscardReason.set(reason);
        throw new Error(reason);
      }

      const responseRequestFrameIdx = Math.trunc(response.request_frame_idx);
      const responseFrameIdx = Math.trunc(response.frame_idx);
      this.store.lastBackendResponseFrameIdx.set(responseFrameIdx);
      this.store.lastBackendResponseFrameFile.set(response.frame_file || 'n/a');
      if (responseRequestFrameIdx !== requestFrameIdx || responseFrameIdx !== requestFrameIdx) {
        const reason = `Discarded response due to frame mismatch (request=${requestFrameIdx}, response_request=${responseRequestFrameIdx}, response_frame=${responseFrameIdx}).`;
        this.store.lastDiscardReason.set(reason);
        throw new Error(reason);
      }

      if (requireDisplayedFrame && this.store.displayedFrameIdx() !== requestFrameIdx) {
        const reason = `Discarded response because displayed frame moved from ${requestFrameIdx} to ${this.store.displayedFrameIdx()}.`;
        this.store.lastDiscardReason.set(reason);
        throw new Error(reason);
      }

      if (
        response.out_masks.length !== response.out_obj_ids.length ||
        new Set(response.out_obj_ids).size !== response.out_obj_ids.length
      ) {
        throw new Error('Object IDs and masks do not match.');
      }
      response.out_obj_ids.forEach((id, index) => {
        const mask = response.out_masks[index];
        if (!Number.isSafeInteger(id) || id <= 0) throw new Error('Invalid mask object ID.');
        if (Array.isArray(mask)) {
          if (response.mask_encoding) throw new Error('Expected encoded mask.');
          return;
        }
        const shape = response.mask_shapes?.[id];
        const pixels = encodedMaskPixelCount(mask);
        if (
          !shape ||
          shape[0] !== mask.size[0] ||
          shape[1] !== mask.size[1] ||
          response.mask_pixel_counts[id] !== pixels
        ) {
          throw new Error('Encoded mask metadata does not match its pixels.');
        }
      });

      const maskPixelCount = getMaskPixelCount(response.mask_pixel_counts, objId);
      this.store.lastMaskPixelCount.set(maskPixelCount);
      this.store.lastFallbackUsed.set(Boolean(response.single_frame_fallback_used));

      const masksMap = new Map(this.store.masks());
      const frameMasksMap = new Map(masksMap.get(requestFrameIdx) || new Map<number, LiveMask>());
      response.out_obj_ids.forEach((id, index) => {
        frameMasksMap.set(id, response.out_masks[index]);
      });
      masksMap.set(requestFrameIdx, frameMasksMap);
      this.markObjectAsLiveEdited(requestFrameIdx, objId);
      this.store.masks.set(masksMap);
      this.store.invalidateTracking();
      this.framePipeline.redraw();
    } catch (error) {
      if (!responseChangedEpoch && !ownsPointUpdate()) {
        this.store.lastDiscardReason.set(
          'Discarded point failure because the editing state changed.',
        );
        return;
      }
      console.error(error);
      this.toast.show(
        'error',
        'Point update failed',
        getErrorMessage(error, 'Unable to update point mask.'),
      );

      // Never restore prompts into a replacement session or overwrite a newer edit.
      if (
        this.store.points().get(frameIdx)?.get(objId) !== objectPoints ||
        !this.store.objects().some((object) => object.id === objId)
      )
        return;
      if (!liveEditedBeforeRequest) {
        this.unmarkObjectAsLiveEdited(frameIdx, objId);
      }
      this.setObjectPoints(frameIdx, objId, previousObjectPoints);
      this.framePipeline.redraw();
    } finally {
      this.store.isPointRequestInFlight.set(false);
    }
  }
}
