import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class FrameRendererService {
  evictWithLimit<T>(cache: Map<number, T>, maxSize: number, onEvict?: (key: number) => void): void {
    while (cache.size > maxSize) {
      const oldestKey = cache.keys().next().value;
      if (oldestKey === undefined) {
        break;
      }
      cache.delete(oldestKey);
      onEvict?.(oldestKey);
    }
  }

  clampFrameIndex(value: number, maxFrame: number): number {
    return Math.min(Math.max(Math.trunc(value), 0), Math.max(maxFrame, 0));
  }
}
