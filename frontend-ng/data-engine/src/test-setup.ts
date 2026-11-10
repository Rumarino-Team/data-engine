import { afterEach, vi } from 'vitest';

// Every spec starts clean: spies restored, stubbed globals undone and real timers back.
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});
