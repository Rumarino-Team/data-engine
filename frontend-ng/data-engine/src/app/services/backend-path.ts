function safeDecodeURIComponent(value: string): string {
	try {
		return decodeURIComponent(value);
	} catch {
		return value;
	}
}


export function normalizeBackendPath(path: string): string {
	let normalized = path.trim().replace(/^['\"]|['\"]$/g, '');
	if (!normalized) {
		return normalized;
	}

	if (normalized.startsWith('file://')) {
		try {
			const parsed = new URL(normalized);
			normalized = parsed.pathname || '';
			if (parsed.host && parsed.host !== 'localhost') {
				normalized = `//${parsed.host}${normalized}`;
			}
			if (/^\/[A-Za-z]:\//.test(normalized)) {
				normalized = normalized.slice(1);
			}
		} catch {
			// Keep the original input if URL parsing fails.
		}
	}

	normalized = safeDecodeURIComponent(normalized);
	normalized = normalized.replace(/\\ /g, ' ');
	return normalized;
}
