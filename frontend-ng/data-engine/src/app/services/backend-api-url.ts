export const API_URL_STORAGE_KEY = 'dataEngineApiUrl';
export const DEFAULT_API_URL = 'http://127.0.0.1:8000';

export function readStoredApiUrl(): string | null {
  // Reading `localStorage` itself throws when storage is blocked (e.g. a SecurityError), so the
  // whole access is guarded, not just the typeof check.
  try {
    return localStorage.getItem(API_URL_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function writeStoredApiUrl(value: string | null): void {
  try {
    if (value === null) {
      localStorage.removeItem(API_URL_STORAGE_KEY);
    } else {
      localStorage.setItem(API_URL_STORAGE_KEY, value);
    }
  } catch {
    // Storage is blocked or full: the URL still applies for this session, it is just not remembered.
  }
}

export function normalizeApiUrl(value: string): string {
  const normalized = value.trim().replace(/^['"]|['"]$/g, '');
  if (!normalized) {
    return DEFAULT_API_URL;
  }
  return normalized.replace(/\/+$/, '');
}
