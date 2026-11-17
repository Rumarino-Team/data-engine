import { DestroyRef, Injectable, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, catchError, defer, exhaustMap, map, of, startWith, switchMap, timer } from 'rxjs';
import { BackendService } from '../../services/backend.service';
import { DEFAULT_API_URL, normalizeApiUrl } from '../../services/backend-api-url';
import { FramePipelineService } from './frame-pipeline.service';
import { SessionLeaveService } from './session-leave.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

/** Owns the backend URL and connectivity status. */
@Injectable()
export class BackendConnectionService {
  private readonly destroyRef = inject(DestroyRef);
  private readonly refreshHealth = new Subject<void>();
  private monitoringHealth = false;
  private readonly framePipeline = inject(FramePipelineService);
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);
  private readonly leave = inject(SessionLeaveService);

  initApiUrlFromBackend(): void {
    this.store.apiUrlInput.set(this.backend.getApiUrl());
  }

  checkApiHealth(showChecking = false): void {
    if (this.destroyRef.destroyed) return;
    if (showChecking || this.store.apiHealthStatus() === 'checking') {
      this.store.apiHealthStatus.set('checking');
    }
    if (this.monitoringHealth) {
      // An explicit refresh cancels the old URL's request and restarts the timer.
      this.refreshHealth.next();
      return;
    }
    this.monitoringHealth = true;
    this.refreshHealth
      .pipe(
        startWith(undefined),
        switchMap(() =>
          timer(0, 3000).pipe(
            // Periodic ticks never overlap a health request that is still running.
            exhaustMap(() =>
              defer(() => this.backend.health()).pipe(
                map(() => 'online' as const),
                catchError(() => of('offline' as const)),
              ),
            ),
          ),
        ),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((status) => this.store.apiHealthStatus.set(status));
  }

  applyApiUrl(): void {
    if (this.store.isInteractionBusy()) return;
    const mayChange = normalizeApiUrl(this.store.apiUrlInput()) !== this.backend.getApiUrl();
    this.switchBackend(mayChange, () => this.backend.setApiUrl(this.store.apiUrlInput()));
  }

  resetApiUrl(): void {
    if (this.store.isInteractionBusy()) return;
    this.switchBackend(this.backend.getApiUrl() !== DEFAULT_API_URL, () =>
      this.backend.resetApiUrl(),
    );
  }

  /** A switch drops the open session, so unsaved changes are saved or discarded first, on the old backend. */
  private switchBackend(mayChange: boolean, change: () => string): void {
    const run = () => {
      const previous = this.backend.getApiUrl();
      this.finishBackendSwitch(previous, change());
    };
    if (!mayChange || !this.store.hasUnsavedChanges()) {
      run();
      return;
    }
    void this.leave.confirmLeave('switch backends').then((proceed) => {
      if (proceed && !this.destroyRef.destroyed && !this.store.isInteractionBusy()) run();
    });
  }

  private finishBackendSwitch(previous: string, next: string): void {
    this.store.apiUrlInput.set(next);
    if (previous !== next) {
      this.store.resetSession();
      this.framePipeline.clearFrameCaches();
    }
    this.checkApiHealth(true);
  }

  isApiUrlDirty(): boolean {
    return this.store.apiUrlInput().trim() !== this.backend.getApiUrl();
  }
}
