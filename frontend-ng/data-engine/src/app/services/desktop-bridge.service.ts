import { Injectable } from '@angular/core';

const SUPPORTED_VIDEO_EXTENSIONS = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'];

@Injectable({
	providedIn: 'root'
})
export class DesktopBridgeService {
	isTauri(): boolean {
		if (typeof window === 'undefined') {
			return false;
		}
		return '__TAURI_INTERNALS__' in window || '__TAURI__' in window;
	}

	async pickVideoFile(): Promise<string | null> {
		return this.openPathPicker({
			multiple: false,
			directory: false,
			filters: [
				{
					name: 'Video',
					extensions: SUPPORTED_VIDEO_EXTENSIONS,
				},
			],
		});
	}

	async pickFramesDirectory(): Promise<string | null> {
		return this.openPathPicker({
			multiple: false,
			directory: true,
		});
	}

	private async openPathPicker(options: Record<string, unknown>): Promise<string | null> {
		if (!this.isTauri()) {
			return null;
		}

		try {
			const dialog = await import('@tauri-apps/plugin-dialog');
			const selected = await dialog.open(options);
			if (typeof selected !== 'string' || !selected.trim()) {
				return null;
			}
			return this.normalizeNativePath(selected);
		} catch (error) {
			console.error('Failed to open native picker', error);
			return null;
		}
	}

	private normalizeNativePath(value: string): string {
		let normalized = value.trim();
		if (!normalized.startsWith('file://')) {
			return normalized;
		}

		try {
			const parsed = new URL(normalized);
			normalized = decodeURIComponent(parsed.pathname || '');
			if (parsed.host && parsed.host !== 'localhost') {
				normalized = `//${parsed.host}${normalized}`;
			}
			if (/^\/[A-Za-z]:\//.test(normalized)) {
				normalized = normalized.slice(1);
			}
		} catch {
			normalized = normalized.replace(/^file:\/\//, '');
		}

		return normalized;
	}
}
