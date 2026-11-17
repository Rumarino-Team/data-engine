import { Injectable, signal } from '@angular/core';

export interface ConfirmAction {
  value: string;
  label: string;
  tone?: 'primary' | 'danger' | 'neutral';
}

export interface ConfirmRequest {
  title: string;
  message: string;
  actions: ConfirmAction[];
}

/** In-app replacement for window.confirm, which desktop webviews do not reliably provide. */
@Injectable()
export class ConfirmService {
  readonly request = signal<ConfirmRequest | null>(null);
  private resolve: ((value: string | null) => void) | null = null;

  /** Resolves with the chosen action's value, or null when dismissed. A newer question dismisses the older one. */
  ask<T extends string>(request: {
    title: string;
    message: string;
    actions: (ConfirmAction & { value: T })[];
  }): Promise<T | null> {
    this.answer(null);
    return new Promise<T | null>((resolve) => {
      this.resolve = resolve as (value: string | null) => void;
      this.request.set(request);
    });
  }

  answer(value: string | null): void {
    const resolve = this.resolve;
    this.resolve = null;
    this.request.set(null);
    resolve?.(value);
  }
}
