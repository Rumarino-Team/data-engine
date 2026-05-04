export const API_URL_STORAGE_KEY = 'dataEngineApiUrl';
export const DEFAULT_API_URL = 'http://127.0.0.1:8000';

export function readStoredApiUrl(): string | null {
  if (typeof localStorage === 'undefined') {
    return null;
  }
  return localStorage.getItem(API_URL_STORAGE_KEY);
}

export function writeStoredApiUrl(value: string | null): void {
  if (typeof localStorage === 'undefined') {
    return;
  }
  if (value === null) {
    localStorage.removeItem(API_URL_STORAGE_KEY);
    return;
  }
  localStorage.setItem(API_URL_STORAGE_KEY, value);
}

export function normalizeApiUrl(value: string): string {
  const normalized = value.trim().replace(/^['"]|['"]$/g, '');
  if (!normalized) {
    return DEFAULT_API_URL;
  }
  return normalized.replace(/\/+$/, '');
}
