import { Injectable, inject } from '@angular/core';
import { finalize, firstValueFrom } from 'rxjs';
import { BackendService } from '../../services/backend.service';
import { getErrorMessage, randomColor } from '../video-masker.util';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import { MaskObject, VideoMaskerStateStore } from './video-masker-state.store';

/** Owns object labels, ID allocation, and removal from the backend and local state. */
@Injectable()
export class ObjectOperationsService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly framePipeline = inject(FramePipelineService);
  private readonly toast = inject(ToastService);

  private get isBusy(): boolean {
    return this.store.isInteractionBusy();
  }

  private showError(title: string, fallback: string, error: unknown): void {
    console.error(error);
    this.toast.show('error', title, getErrorMessage(error, fallback));
  }

  // --- objects -----------------------------------------------------------

  renameObject(id: number, value: string): void {
    if (this.isBusy) return;
    const label = value.trim().slice(0, 200);
    const previous = this.store.objects().find((object) => object.id === id);
    if (!label || !previous || previous.label === label) return;
    const next = { ...previous, label };
    this.replaceObject(next);
    this.syncObject(next, previous);
  }

  addObject(): void {
    if (this.isBusy) return;
    const newId = this.reserveNextObjectId();
    this.store.nextObjectId = newId + 1;
    const object = { id: newId, label: `Object ${newId}`, color: randomColor() };
    this.store.objects.set([...this.store.objects(), object]);
    this.store.selectedObjectId.set(newId);
    this.syncObject(object);
  }

  /** Persists object metadata to the backend; a failed relabel reverts to `previous`. */
  syncObject(object: MaskObject, previous?: MaskObject): void {
    this.backend.upsertObject(object).subscribe({
      error: (error) => {
        if (previous) this.replaceObject(previous);
        this.showError('Object update failed', 'Failed to save object label.', error);
      },
    });
  }

  removeObject(): void {
    if (this.isBusy) return;
    const id = this.store.selectedObjectId();
    if (id === null) return;

    this.reserveNextObjectId();
    this.store.isLoading.set(true);
    this.backend
      .removeObject(id)
      .pipe(finalize(() => this.store.isLoading.set(false)))
      .subscribe({
        next: () => {
          this.applyObjectRemoval(id);
        },
        error: (error) => this.showError('Remove failed', 'Failed to remove object.', error),
      });
  }

  async removeAllObjects(): Promise<void> {
    if (this.isBusy) return;
    const objectIds = this.store.objects().map((entry) => entry.id);
    if (objectIds.length === 0) return;

    this.reserveNextObjectId();
    this.store.isLoading.set(true);
    try {
      for (const id of objectIds) {
        await firstValueFrom(this.backend.removeObject(id));
        this.applyObjectRemoval(id);
      }
    } catch (error) {
      this.showError('Remove all failed', 'Failed to remove all objects.', error);
    } finally {
      this.store.isLoading.set(false);
    }
  }

  private replaceObject(next: MaskObject): void {
    this.store.objects.update((objects) =>
      objects.map((object) => (object.id === next.id ? next : object)),
    );
    this.framePipeline.redraw();
  }

  private reserveNextObjectId(): number {
    this.store.nextObjectId = this.store
      .objects()
      .reduce((next, object) => Math.max(next, object.id + 1), this.store.nextObjectId);
    return this.store.nextObjectId;
  }

  private applyObjectRemoval(id: number): void {
    this.store.objects.set(this.store.objects().filter((entry) => entry.id !== id));
    this.removeObjectFromFrameMaps(id);
    if (this.store.selectedObjectId() === id) {
      this.store.selectedObjectId.set(this.store.objects()[0]?.id ?? null);
    }
    this.store.invalidateTracking();
    this.framePipeline.removeObject(id);
    this.framePipeline.redraw();
    this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
  }

  private removeObjectFromFrameMaps(objectId: number): void {
    const nextMasks = new Map(this.store.masks());
    nextMasks.forEach((frameMap, frameIdx) => {
      const next = new Map(frameMap);
      next.delete(objectId);
      if (next.size) nextMasks.set(frameIdx, next);
      else nextMasks.delete(frameIdx);
    });
    this.store.masks.set(nextMasks);

    const nextPoints = new Map(this.store.points());
    nextPoints.forEach((frameMap, frameIdx) => {
      const next = new Map(frameMap);
      next.delete(objectId);
      if (next.size) nextPoints.set(frameIdx, next);
      else nextPoints.delete(frameIdx);
    });
    this.store.points.set(nextPoints);

    const nextEdited = new Map(this.store.liveEditedObjectFrames());
    nextEdited.forEach((ids, frameIdx) => {
      const next = new Set(ids);
      next.delete(objectId);
      if (next.size) nextEdited.set(frameIdx, next);
      else nextEdited.delete(frameIdx);
    });
    this.store.liveEditedObjectFrames.set(nextEdited);
  }
}
