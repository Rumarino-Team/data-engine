import { DestroyRef, Injectable, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { firstValueFrom } from 'rxjs';
import { BackendService, LiveMask, VideoAddPointsResponse } from '../../services/backend.service';
import { VideoMaskerStateStore, Point } from './video-masker-state.store';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import {
  getErrorMessage,
  getMaskPixelCount,
  encodedMaskPixelCount,
  markObjectLiveEdited,
  unmarkObjectLiveEdited,
} from '../video-masker.util';

/** Keeps pending previews separate from backend-accepted prompts. */
@Injectable()
export class PointEditingService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly framePipeline = inject(FramePipelineService);
  private readonly toast = inject(ToastService);
  private readonly destroyRef = inject(DestroyRef);

  async addPoint(x: number, y: number, label: number, frameIdx: number): Promise<void> {
    const objId = this.store.selectedObjectId();
    if (objId === null) return;
    await this.submitObjectPoints(objId, frameIdx, [
      ...(this.store.points().get(frameIdx)?.get(objId) ?? []),
      { x, y, label },
    ]);
  }

  async removePoint(objId: number, frameIdx: number, pointIndex: number): Promise<void> {
    const points = this.store.points().get(frameIdx)?.get(objId) ?? [];
    if (pointIndex < 0 || pointIndex >= points.length) return;
    await this.submitObjectPoints(
      objId,
      frameIdx,
      points.filter((_, index) => index !== pointIndex),
    );
  }

  private async submitObjectPoints(
    objId: number,
    frameIdx: number,
    objectPoints: Point[],
  ): Promise<void> {
    if (
      this.store.isInteractionBusy() ||
      !this.store.objects().some((object) => object.id === objId)
    )
      return;
    const sessionId = this.store.sessionId();
    const revision = this.store.annotationRevision();
    const backendUrl = this.backend.getApiUrl();
    if (!sessionId || revision === null || backendUrl !== this.store.annotationsBackendUrl())
      return;
    const expectedEpoch = this.store.stateEpoch();
    const precondition = { session_id: sessionId, expected_revision: revision };
    const preview = { objId, frameIdx, points: objectPoints };
    const ownsContext = () =>
      !this.destroyRef.destroyed &&
      sessionId === this.store.sessionId() &&
      backendUrl === this.backend.getApiUrl() &&
      backendUrl === this.store.annotationsBackendUrl();
    const ownsUpdate = () =>
      ownsContext() &&
      this.store.stateEpoch() === expectedEpoch &&
      this.store.annotationRevision() === revision &&
      this.store.pendingPointEdit() === preview;
    this.store.pendingPointEdit.set(preview);
    this.store.isPointRequestInFlight.set(true);
    const requestFrameIdx = frameIdx;
    this.store.lastClickRequestFrameIdx.set(frameIdx);
    this.store.lastDebugObjectId.set(objId);
    this.store.lastMaskPixelCount.set(null);
    this.store.lastBackendResponseFrameIdx.set(null);
    this.store.lastBackendResponseFrameFile.set('n/a');
    this.store.lastBackendResponseStateEpoch.set(null);
    this.store.lastFallbackUsed.set(false);
    this.store.lastDiscardReason.set(null);
    this.framePipeline.redraw();
    let responseReceived = false;
    try {
      const request = objectPoints.length
        ? this.backend.addNewPointsOrBox(
            {
              frame_idx: frameIdx,
              obj_id: objId,
              points: objectPoints.map(({ x, y }) => [x, y]),
              labels: objectPoints.map(({ label }) => label),
              clear_old_points: true,
            },
            precondition,
          )
        : this.backend.clearAllPromptsInFrame(frameIdx, objId, precondition);
      const result = await firstValueFrom(request.pipe(takeUntilDestroyed(this.destroyRef)));
      if (!ownsUpdate()) {
        this.store.lastDiscardReason.set(
          'Discarded point response because the editing state changed.',
        );
        return;
      }
      responseReceived = true;
      if (!objectPoints.length) {
        if (!this.store.applyAnnotationSnapshot(result, backendUrl)) {
          this.store.annotationRecoveryRequested.set(true);
          return;
        }
        const masks = new Map(this.store.masks());
        const frame = new Map(masks.get(frameIdx));
        frame.delete(objId);
        if (frame.size) masks.set(frameIdx, frame);
        else masks.delete(frameIdx);
        this.store.masks.set(masks);
        this.store.liveEditedObjectFrames.set(
          unmarkObjectLiveEdited(this.store.liveEditedObjectFrames(), frameIdx, objId),
        );
        this.store.invalidateTracking();
        return;
      }
      const response = result as VideoAddPointsResponse;
      if (
        typeof (response as any)?.request_frame_idx !== 'number' ||
        typeof (response as any)?.frame_idx !== 'number' ||
        typeof (response as any)?.frame_file !== 'string' ||
        !Number.isInteger(response?.state_epoch) ||
        !Array.isArray((response as any)?.out_obj_ids) ||
        !Array.isArray((response as any)?.out_masks) ||
        typeof (response as any)?.mask_pixel_counts !== 'object'
      ) {
        throw new Error('Invalid mask response');
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

      if (!this.store.applyAnnotationSnapshot(response, backendUrl)) {
        this.store.annotationRecoveryRequested.set(true);
        return;
      }

      const maskPixelCount = getMaskPixelCount(response.mask_pixel_counts, objId);
      this.store.lastMaskPixelCount.set(maskPixelCount);
      this.store.lastFallbackUsed.set(Boolean(response.single_frame_fallback_used));

      const masksMap = new Map(this.store.masks());
      const frameMasksMap = new Map(masksMap.get(requestFrameIdx) || new Map<number, LiveMask>());
      response.out_obj_ids.forEach((id, index) => {
        frameMasksMap.set(id, response.out_masks[index]);
      });
      masksMap.set(requestFrameIdx, frameMasksMap);
      this.store.liveEditedObjectFrames.set(
        markObjectLiveEdited(this.store.liveEditedObjectFrames(), requestFrameIdx, objId),
      );
      this.store.masks.set(masksMap);
      this.store.invalidateTracking();
      this.framePipeline.redraw();
    } catch (error) {
      if (!ownsContext()) return;
      if (!responseReceived && !ownsUpdate()) {
        this.store.lastDiscardReason.set(
          'Discarded point failure because the editing state changed.',
        );
        return;
      }
      const status = (error as { status?: number })?.status;
      if (
        responseReceived ||
        status === undefined ||
        status === 0 ||
        status === 409 ||
        status >= 500
      )
        this.store.annotationRecoveryRequested.set(true);
      this.toast.show(
        'error',
        objectPoints.length ? 'Point update failed' : 'Point removal failed',
        getErrorMessage(error, 'Unable to update prompts.'),
      );
    } finally {
      if (this.store.pendingPointEdit() === preview) this.store.pendingPointEdit.set(null);
      this.store.isPointRequestInFlight.set(false);
      if (!this.destroyRef.destroyed) this.framePipeline.redraw();
    }
  }
}
