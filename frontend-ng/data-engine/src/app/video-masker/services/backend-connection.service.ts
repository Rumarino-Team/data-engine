import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { BackendService } from '../../services/backend.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

/** Owns the backend URL and connectivity status. */
@Injectable()
export class BackendConnectionService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly backend = inject(BackendService);

  initApiUrlFromBackend(): void {
    this.store.apiUrlInput.set(this.backend.getApiUrl());
  }

  async checkApiHealth(showChecking = false): Promise<void> {
    if (showChecking || this.store.apiHealthStatus() === 'checking') {
      this.store.apiHealthStatus.set('checking');
    }
    try {
      await firstValueFrom(this.backend.health());
      this.store.apiHealthStatus.set('online');
    } catch {
      this.store.apiHealthStatus.set('offline');
    }
  }

  applyApiUrl(): void {
    this.store.apiUrlInput.set(this.backend.setApiUrl(this.store.apiUrlInput()));
    void this.checkApiHealth(true);
  }

  resetApiUrl(): void {
    this.store.apiUrlInput.set(this.backend.resetApiUrl());
    void this.checkApiHealth(true);
  }

  isApiUrlDirty(): boolean {
    return this.store.apiUrlInput().trim() !== this.backend.getApiUrl();
  }
}
