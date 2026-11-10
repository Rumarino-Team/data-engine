import { DestroyRef, Injectable, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { firstValueFrom } from 'rxjs';
import { AnnotationUpdate, BackendService } from '../../services/backend.service';
import { VideoMaskerStateStore, Point } from './video-masker-state.store';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import { getErrorMessage } from '../video-masker.util';

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
    if (!sessionId || revision === null) return;
    const precondition = { session_id: sessionId, expected_revision: revision };
    this.store.pendingPointEdit.set({ objId, frameIdx, points: objectPoints });
    this.store.isPointRequestInFlight.set(true);
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
      const result: AnnotationUpdate = await firstValueFrom(
        request.pipe(takeUntilDestroyed(this.destroyRef)),
      );
      responseReceived = true;
      if (!this.store.applyAnnotationUpdate(result)) {
        this.store.annotationRecoveryRequested.set(true);
        return;
      }
      if (!this.framePipeline.acceptMaskData(result.mask_data, frameIdx))
        throw new Error('Invalid effective mask response.');
      if (frameIdx !== this.store.displayedFrameIdx())
        this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
    } catch (error) {
      if (this.destroyRef.destroyed) return;
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
      this.store.pendingPointEdit.set(null);
      this.store.isPointRequestInFlight.set(false);
      if (!this.destroyRef.destroyed) this.framePipeline.redraw();
    }
  }
}
