import { afterEach, vi } from 'vitest';

// Node 25+ defines its own `localStorage` global (undefined unless --localstorage-file is given).
// Vitest's jsdom environment does not overwrite pre-existing globals that are not on its key list,
// so jsdom's real localStorage is shadowed. Point the global back at jsdom's.
const jsdomWindow = (globalThis as { jsdom?: { window: Window } }).jsdom?.window;
if (jsdomWindow && typeof globalThis.localStorage === 'undefined') {
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    get: () => jsdomWindow.localStorage,
  });
}

// Every spec starts clean: spies restored, stubbed globals undone and real timers back.
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});
