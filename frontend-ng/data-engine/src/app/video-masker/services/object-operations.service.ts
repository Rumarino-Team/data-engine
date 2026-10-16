import { DestroyRef, Injectable, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Observable, defer, finalize, take } from 'rxjs';
import { AnnotationSnapshot, BackendService } from '../../services/backend.service';
import { getErrorMessage, randomColor } from '../video-masker.util';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

type SessionContext = { backendUrl: string; sessionId: string };

/** Sends object edits and applies accepted backend annotations. ID allocation moves next. */
@Injectable()
export class ObjectOperationsService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly framePipeline = inject(FramePipelineService);
  private readonly toast = inject(ToastService);
  private readonly destroyRef = inject(DestroyRef);

  renameObject(id: number, value: string): void {
    if (this.store.isInteractionBusy()) return;
    const label = value.trim().slice(0, 200);
    const object = this.store.objects().find((entry) => entry.id === id);
    if (!label || !object || object.label === label) return;
    void this.mutate(() => this.backend.upsertObject({ ...object, label }), 'Object update failed');
  }

  addObject(): void {
    if (this.store.isInteractionBusy()) return;
    const id = this.reserveNextObjectId();
    this.store.nextObjectId = id + 1;
    const object = { id, label: `Object ${id}`, color: randomColor() };
    void this.mutate(() => this.backend.upsertObject(object), 'Object creation failed', id);
  }

  removeObject(): void {
    if (this.store.isInteractionBusy()) return;
    const id = this.store.selectedObjectId();
    if (id !== null) void this.mutate(() => this.backend.removeObject(id), 'Remove failed');
  }

  async removeAllObjects(): Promise<void> {
    if (this.store.isInteractionBusy()) return;
    const context = this.sessionContext();
    if (!context) return;
    const ids = this.store.objects().map((object) => object.id);
    this.store.pendingObjectUpdates.update((count) => count + 1);
    try {
      for (const id of ids) {
        if (
          !(await this.mutate(
            () => this.backend.removeObject(id),
            'Remove all failed',
            undefined,
            context,
          ))
        )
          break;
      }
    } finally {
      this.store.pendingObjectUpdates.update((count) => count - 1);
    }
  }

  private sessionContext(): SessionContext | null {
    const sessionId = this.store.sessionId();
    const backendUrl = this.backend.getApiUrl();
    return sessionId && backendUrl === this.store.annotationsBackendUrl()
      ? { sessionId, backendUrl }
      : null;
  }

  private ownsSession(context: SessionContext): boolean {
    return (
      !this.destroyRef.destroyed &&
      context.sessionId === this.store.sessionId() &&
      context.backendUrl === this.backend.getApiUrl() &&
      context.backendUrl === this.store.annotationsBackendUrl()
    );
  }

  private mutate(
    send: () => Observable<AnnotationSnapshot>,
    errorTitle: string,
    selectId?: number,
    context = this.sessionContext(),
  ): Promise<boolean> {
    if (!context || !this.ownsSession(context)) return Promise.resolve(false);
    this.store.pendingObjectUpdates.update((count) => count + 1);
    return new Promise((resolve) => {
      defer(send)
        .pipe(
          take(1),
          takeUntilDestroyed(this.destroyRef),
          finalize(() => this.store.pendingObjectUpdates.update((count) => count - 1)),
        )
        .subscribe({
          next: (snapshot) => {
            if (!this.ownsSession(context)) {
              resolve(false);
              return;
            }
            const previousIds = this.store.objects().map((object) => object.id);
            if (!this.store.applyAnnotationSnapshot(snapshot, context.backendUrl)) {
              resolve(false);
              return;
            }
            const removed = previousIds.filter(
              (id) => !this.store.objects().some((object) => object.id === id),
            );
            if (removed.length) this.clearRemovedRenderCaches(removed);
            if (
              selectId !== undefined &&
              this.store.objects().some((object) => object.id === selectId)
            ) {
              this.store.selectedObjectId.set(selectId);
            }
            this.framePipeline.redraw();
            resolve(true);
          },
          error: (error) => {
            if (this.ownsSession(context))
              this.toast.show(
                'error',
                errorTitle,
                getErrorMessage(error, 'Unable to update objects.'),
              );
            resolve(false);
          },
          complete: () => resolve(false),
        });
    });
  }

  private reserveNextObjectId(): number {
    return this.store
      .objects()
      .reduce((next, object) => Math.max(next, object.id + 1), this.store.nextObjectId);
  }

  /** Generated masks and textures are view caches; accepted prompts came from the snapshot. */
  private clearRemovedRenderCaches(ids: number[]): void {
    const masks = new Map(this.store.masks());
    for (const [frame, objects] of masks) {
      const retained = new Map([...objects].filter(([id]) => !ids.includes(id)));
      if (retained.size) masks.set(frame, retained);
      else masks.delete(frame);
    }
    this.store.masks.set(masks);
    const edited = new Map(this.store.liveEditedObjectFrames());
    for (const [frame, objects] of edited) {
      const retained = new Set([...objects].filter((id) => !ids.includes(id)));
      if (retained.size) edited.set(frame, retained);
      else edited.delete(frame);
    }
    this.store.liveEditedObjectFrames.set(edited);
    this.store.invalidateTracking();
    ids.forEach((id) => this.framePipeline.removeObject(id));
    this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
  }
}
