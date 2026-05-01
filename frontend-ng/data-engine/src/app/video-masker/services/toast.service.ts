import { Injectable, signal } from '@angular/core';
import { AppToast, ToastSeverity } from '../state/video-masker-ui.types';

@Injectable({ providedIn: 'root' })
export class ToastService {
  readonly toasts = signal<AppToast[]>([]);
  private nextToastId = 1;

  show(severity: ToastSeverity, title: string, message: string): void {
    const toast: AppToast = {
      id: this.nextToastId++,
      severity,
      title,
      message,
      createdAt: Date.now(),
    };
    this.toasts.update((existing) => [toast, ...existing].slice(0, 6));
    if (severity === 'info' || severity === 'success') {
      setTimeout(() => this.dismiss(toast.id), 5000);
    }
  }

  dismiss(id: number): void {
    this.toasts.update((existing) => existing.filter((toast) => toast.id !== id));
  }
}
