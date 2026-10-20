import { DestroyRef, Injectable, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Observable, defer, finalize, take } from 'rxjs';
import {
  AnnotationUpdate,
  AnnotationPrecondition,
  BackendService,
} from '../../services/backend.service';
import { getErrorMessage } from '../video-masker.util';
import { FramePipelineService } from './frame-pipeline.service';
import { ToastService } from './toast.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

type SessionContext = { backendUrl: string; sessionId: string };

/** Sends object intent and applies accepted backend annotations. */
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
    void this.mutate(
      (precondition) => this.backend.updateObject({ ...object, label }, precondition),
      'Object update failed',
    );
  }

  addObject(): void {
    if (this.store.isInteractionBusy()) return;
    void this.mutate(
      (precondition) => this.backend.createObject(precondition),
      'Object creation failed',
      (snapshot) => snapshot.created_object_id,
    );
  }

  removeObject(): void {
    if (this.store.isInteractionBusy()) return;
    const id = this.store.selectedObjectId();
    if (id !== null)
      void this.mutate(
        (precondition) => this.backend.removeObject(id, precondition),
        'Remove failed',
      );
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
            (precondition) => this.backend.removeObject(id, precondition),
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

  private mutate<T extends AnnotationUpdate>(
    send: (precondition: AnnotationPrecondition) => Observable<T>,
    errorTitle: string,
    selectCreated?: (snapshot: T) => number,
    context = this.sessionContext(),
  ): Promise<boolean> {
    if (!context || !this.ownsSession(context)) return Promise.resolve(false);
    this.store.pendingObjectUpdates.update((count) => count + 1);
    return new Promise((resolve) => {
      defer(() =>
        send({
          session_id: context.sessionId,
          expected_revision: this.store.annotationRevision()!,
        }),
      )
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
            if (!this.store.applyAnnotationUpdate(snapshot, context.backendUrl)) {
              this.store.annotationRecoveryRequested.set(true);
              resolve(false);
              return;
            }
            const removed = previousIds.filter(
              (id) => !this.store.objects().some((object) => object.id === id),
            );
            if (removed.length) this.clearRemovedRenderCaches();
            const selectId = selectCreated?.(snapshot);
            if (
              selectId !== undefined &&
              this.store.objects().some((object) => object.id === selectId)
            ) {
              this.store.selectedObjectId.set(selectId);
            }
            this.framePipeline.redraw();
            this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
            resolve(true);
          },
          error: (error) => {
            if (this.ownsSession(context)) {
              const status = (error as { status?: number })?.status;
              if (status === undefined || status === 0 || status === 409 || status >= 500)
                this.store.annotationRecoveryRequested.set(true);
              this.toast.show(
                'error',
                errorTitle,
                getErrorMessage(error, 'Unable to update objects.'),
              );
            }
            resolve(false);
          },
          complete: () => resolve(false),
        });
    });
  }

  /** Generated masks and textures are view caches; accepted prompts came from the snapshot. */
  private clearRemovedRenderCaches(): void {
    this.store.invalidateTracking();
    this.framePipeline.clearMaskDataCache();
    this.framePipeline.scheduleFrameLoad(this.store.targetFrameIdx());
  }
}
