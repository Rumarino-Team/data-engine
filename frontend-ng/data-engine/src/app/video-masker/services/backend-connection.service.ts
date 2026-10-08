import { DestroyRef, Injectable, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, catchError, defer, exhaustMap, map, of, startWith, switchMap, timer } from 'rxjs';
import { BackendService } from '../../services/backend.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

/** Owns the backend URL and connectivity status. */
@Injectable()
export class BackendConnectionService {
  private readonly destroyRef = inject(DestroyRef);
  private readonly refreshHealth = new Subject<void>();
  private monitoringHealth = false;
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);

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
    this.store.apiUrlInput.set(this.backend.setApiUrl(this.store.apiUrlInput()));
    this.checkApiHealth(true);
  }

  resetApiUrl(): void {
    this.store.apiUrlInput.set(this.backend.resetApiUrl());
    this.checkApiHealth(true);
  }

  isApiUrlDirty(): boolean {
    return this.store.apiUrlInput().trim() !== this.backend.getApiUrl();
  }
}
