import { Component, ElementRef, OnDestroy, ViewChild, effect, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';
import {
	ApiHealthStatus,
	BackendJob,
	BackendService,
	InteractiveMaskRle,
	InteractiveObject,
	InteractivePoint,
	JobStageHistoryEntry,
	RestoredSessionPayload,
	TrackPromptPointMetadata,
	TrackPromptPointsResponse,
	VideoAddPointsOrBoxRequest,
	VideoInitStateResponse,
	VideoMaskObjectData,
	VideoPropagateResponse,
	VideoSaveInteractiveState,
	VideoSaveResponse
} from '../services/backend.service';
import { DesktopBridgeService } from '../services/desktop-bridge.service';

interface MaskObject {
	id: number;
	name: string;
	color: string;
}

interface Point {
	x: number;
	y: number;
	label: number; // 1 for positive, 0 for negative
}

interface TrackedPointSeries extends TrackPromptPointMetadata {
	tracks: number[][];
	visibility: boolean[];
}

type TrackingOverlayStyle = 'point' | 'short' | 'full';
type DebugMaskSource = 'live' | 'manifest' | 'none';
type ToastSeverity = 'error' | 'warning' | 'info' | 'success';
type LoadSourceMode = 'frames_dir' | 'video_file' | 'saved_session_dir';

interface AppToast {
	id: number;
	severity: ToastSeverity;
	title: string;
	message: string;
	createdAt: number;
}

@Component({
	selector: 'app-video-masker',
	standalone: true,
	imports: [CommonModule, FormsModule],
	templateUrl: './video-masker.component.html',
	styleUrls: ['./video-masker.component.css']
})
export class VideoMaskerComponent implements OnDestroy {
	@ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;
	@ViewChild('videoFileInput') videoFileInputRef?: ElementRef<HTMLInputElement>;
	@ViewChild('framesDirInput') framesDirInputRef?: ElementRef<HTMLInputElement>;

	videoDir = signal<string>('');
	loadSourceMode = signal<LoadSourceMode>('frames_dir');
	apiUrlInput = signal<string>('');
	isInitialized = signal<boolean>(false);
	numFrames = signal<number>(0);
	targetFrameIdx = signal<number>(0);
	displayedFrameIdx = signal<number>(-1);
	stateEpoch = signal<number>(0);

	objects = signal<MaskObject[]>([]);
	selectedObjectId = signal<number | null>(null);
	interactionMode = signal<'positive' | 'negative'>('positive');

	masks = signal<Map<number, Map<number, boolean[][]>>>(new Map());
	points = signal<Map<number, Map<number, Point[]>>>(new Map());
	liveEditedObjectFrames = signal<Map<number, Set<number>>>(new Map());
	hasManifestMasks = signal<boolean>(false);
	saveName = signal<string>('');

	trackingOverlayStyle = signal<TrackingOverlayStyle>('short');
	trackingUseSupportGrid = signal<boolean>(false);
	trackedPoints = signal<TrackedPointSeries[]>([]);

	isLoading = signal<boolean>(false);
	isFrameLoading = signal<boolean>(false);
	isPointRequestInFlight = signal<boolean>(false);
	apiHealthStatus = signal<ApiHealthStatus>('checking');
	activeJob = signal<BackendJob | null>(null);
	activeJobTitle = signal<string>('');
	toasts = signal<AppToast[]>([]);
	lastClickRequestFrameIdx = signal<number | null>(null);
	lastBackendResponseFrameIdx = signal<number | null>(null);
	lastBackendResponseFrameFile = signal<string>('n/a');
	lastBackendResponseStateEpoch = signal<number | null>(null);
	lastDebugObjectId = signal<number | null>(null);
	lastMaskPixelCount = signal<number | null>(null);
	lastFallbackUsed = signal<boolean>(false);
	lastMaskSource = signal<DebugMaskSource>('none');
	lastDiscardReason = signal<string | null>(null);

	private frameLoadToken = 0;
	private pendingFrameIdx: number | null = null;
	private frameLoadAnimationId: number | null = null;
	private frameImageCache = new Map<number, HTMLImageElement>();
	private maskDataCache = new Map<number, { [objId: string]: VideoMaskObjectData }>();
	private readonly maxFrameCacheSize = 24;
	private currentBaseImage: HTMLImageElement | null = null;
	private currentMaskObjects: { [objId: string]: VideoMaskObjectData } = {};
	private healthTimerId: ReturnType<typeof setInterval> | null = null;
	private nextToastId = 1;

	constructor(
		private backend: BackendService,
		private desktopBridge: DesktopBridgeService,
	) {
		this.apiUrlInput.set(this.backend.getApiUrl());

		effect(() => {
			if (this.isInitialized()) {
				this.scheduleFrameLoad(this.targetFrameIdx());
			}
		});

		effect(() => {
			this.trackingOverlayStyle();
			this.trackedPoints();
			if (this.currentBaseImage) {
				this.drawCurrentFrame();
			}
		});

		this.checkApiHealth(true);
		this.healthTimerId = setInterval(() => this.checkApiHealth(), 3000);
	}

	ngOnDestroy(): void {
		if (this.healthTimerId !== null) {
			clearInterval(this.healthTimerId);
		}
		if (this.frameLoadAnimationId !== null) {
			cancelAnimationFrame(this.frameLoadAnimationId);
		}
	}

	private async checkApiHealth(showChecking = false): Promise<void> {
		if (showChecking || this.apiHealthStatus() === 'checking') {
			this.apiHealthStatus.set('checking');
		}
		try {
			await firstValueFrom(this.backend.health());
			this.apiHealthStatus.set('online');
		} catch {
			this.apiHealthStatus.set('offline');
		}
	}

	private showToast(severity: ToastSeverity, title: string, message: string): void {
		const toast: AppToast = {
			id: this.nextToastId++,
			severity,
			title,
			message,
			createdAt: Date.now(),
		};
		this.toasts.update((existing) => [toast, ...existing].slice(0, 6));
		if (severity === 'info' || severity === 'success') {
			setTimeout(() => this.dismissToast(toast.id), 5000);
		}
	}

	dismissToast(id: number): void {
		this.toasts.update((existing) => existing.filter((toast) => toast.id !== id));
	}

	recentJobHistory(): JobStageHistoryEntry[] {
		const history = this.activeJob()?.stage_history || [];
		return history.slice(-3);
	}

	private getErrorMessage(error: any, fallback: string): string {
		return error?.error?.detail || error?.error?.error || error?.message || fallback;
	}

	private delay(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}

	private async runBackendJob<T>(
		title: string,
		startJob: () => Promise<{ job_id: string }>,
	): Promise<T | null> {
		this.isLoading.set(true);
		this.activeJobTitle.set(title);
		this.activeJob.set(null);
		try {
			const started = await startJob();
			while (true) {
				const response = await firstValueFrom(this.backend.getJob<T>(started.job_id));
				this.activeJob.set(response.job);
				if (response.job.status === 'completed') {
					return response.job.result as T;
				}
				if (response.job.status === 'failed') {
					const message = response.job.error?.message || response.job.message || `${title} failed`;
					this.showToast('error', title, message);
					return null;
				}
				await this.delay(500);
			}
		} catch (error: any) {
			console.error(error);
			this.apiHealthStatus.set('offline');
			this.showToast('error', title, this.getErrorMessage(error, `${title} failed`));
			return null;
		} finally {
			this.activeJob.set(null);
			this.activeJobTitle.set('');
			this.isLoading.set(false);
		}
	}

	private updateStateEpoch(nextEpoch: number | undefined, source: string): void {
		if (typeof nextEpoch !== 'number' || !Number.isFinite(nextEpoch)) {
			return;
		}
		const normalizedEpoch = Math.trunc(nextEpoch);
		if (normalizedEpoch <= 0) {
			return;
		}
		const previousEpoch = this.stateEpoch();
		if (previousEpoch !== 0 && previousEpoch !== normalizedEpoch) {
			this.masks.set(new Map());
			this.liveEditedObjectFrames.set(new Map());
			this.lastDiscardReason.set(`State epoch changed (${previousEpoch} -> ${normalizedEpoch}) during ${source}; cleared live masks.`);
		}
		this.stateEpoch.set(normalizedEpoch);
	}

	private markObjectAsLiveEdited(frameIdx: number, objId: number): void {
		const next = new Map(this.liveEditedObjectFrames());
		const existing = next.get(frameIdx);
		const nextSet = existing ? new Set(existing) : new Set<number>();
		nextSet.add(objId);
		next.set(frameIdx, nextSet);
		this.liveEditedObjectFrames.set(next);
	}

	private unmarkObjectAsLiveEdited(frameIdx: number, objId: number): void {
		const next = new Map(this.liveEditedObjectFrames());
		const existing = next.get(frameIdx);
		if (!existing) {
			return;
		}
		const nextSet = new Set(existing);
		nextSet.delete(objId);
		if (nextSet.size === 0) {
			next.delete(frameIdx);
		} else {
			next.set(frameIdx, nextSet);
		}
		this.liveEditedObjectFrames.set(next);
	}

	private isObjectLiveEdited(frameIdx: number, objId: number): boolean {
		const frameSet = this.liveEditedObjectFrames().get(frameIdx);
		return Boolean(frameSet?.has(objId));
	}

	private resetDebugState(): void {
		this.lastClickRequestFrameIdx.set(null);
		this.lastBackendResponseFrameIdx.set(null);
		this.lastBackendResponseFrameFile.set('n/a');
		this.lastBackendResponseStateEpoch.set(null);
		this.lastDebugObjectId.set(null);
		this.lastMaskPixelCount.set(null);
		this.lastFallbackUsed.set(false);
		this.lastMaskSource.set('none');
		this.lastDiscardReason.set(null);
	}

	private getMaskPixelCount(pixelCounts: Record<number, number> | undefined, objId: number): number | null {
		if (!pixelCounts) {
			return null;
		}
		const direct = (pixelCounts as Record<number, number>)[objId];
		if (typeof direct === 'number' && Number.isFinite(direct)) {
			return Math.trunc(direct);
		}
		const stringLookup = (pixelCounts as unknown as Record<string, number>)[String(objId)];
		if (typeof stringLookup === 'number' && Number.isFinite(stringLookup)) {
			return Math.trunc(stringLookup);
		}
		return null;
	}

	openVideoFilePicker() {
		this.videoFileInputRef?.nativeElement.click();
	}

	openFramesDirPicker() {
		this.framesDirInputRef?.nativeElement.click();
	}

	onLoadSourceModeChange(value: string) {
		const nextMode: LoadSourceMode =
			value === 'video_file' || value === 'saved_session_dir' ? value : 'frames_dir';
		this.loadSourceMode.set(nextMode);
	}

	onVideoDirChange(value: string) {
		this.videoDir.set(value);
	}

	onApiUrlChange(value: string) {
		this.apiUrlInput.set(value);
	}

	onSaveNameChange(value: string) {
		this.saveName.set(value);
	}

	getLoadPathPlaceholder(): string {
		switch (this.loadSourceMode()) {
			case 'video_file':
				return 'Enter video file path (.mp4, .mov, .avi, .mkv, .webm, .m4v)';
			case 'saved_session_dir':
				return 'Enter saved session directory path (contains session.json, frames/, and masks/)';
			default:
				return 'Enter frames directory path';
		}
	}

	getBrowseLabel(): string {
		switch (this.loadSourceMode()) {
			case 'video_file':
				return 'Browse Video';
			case 'saved_session_dir':
				return 'Browse Saved Session';
			default:
				return 'Browse Frames';
		}
	}

	getLoadModeHint(): string {
		if (this.loadSourceMode() === 'saved_session_dir') {
			return 'Saved session directory should include session.json, frames/, and masks/.';
		}
		return '';
	}

	applyApiUrl() {
		this.apiUrlInput.set(this.backend.setApiUrl(this.apiUrlInput()));
		this.checkApiHealth(true);
	}

	resetApiUrl() {
		this.apiUrlInput.set(this.backend.resetApiUrl());
		this.checkApiHealth(true);
	}

	async browseSelectedSource() {
		const mode = this.loadSourceMode();
		if (this.desktopBridge.isTauri()) {
			if (mode === 'video_file') {
				const selectedPath = await this.desktopBridge.pickVideoFile();
				if (selectedPath) {
					this.videoDir.set(selectedPath);
				}
				return;
			}
			const selectedPath = await this.desktopBridge.pickFramesDirectory();
			if (selectedPath) {
				this.videoDir.set(selectedPath);
			}
			return;
		}
		if (mode === 'video_file') {
			this.openVideoFilePicker();
			return;
		}
		this.openFramesDirPicker();
	}

	async browseVideo() {
		this.loadSourceMode.set('video_file');
		await this.browseSelectedSource();
	}

	async browseFramesDirectory() {
		this.loadSourceMode.set('frames_dir');
		await this.browseSelectedSource();
	}

	onVideoFileSelected(event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) {
			return;
		}

		const nativePath = this.getNativeFilePath(file);
		if (nativePath) {
			this.videoDir.set(nativePath);
		} else {
			this.videoDir.set(file.name);
			this.showPathUnavailableMessage('video');
		}

		input.value = '';
	}

	onFramesDirSelected(event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) {
			return;
		}

		const nativePath = this.getNativeFilePath(file);
		if (nativePath) {
			this.videoDir.set(this.getParentDirectory(nativePath));
		} else {
			this.showPathUnavailableMessage('directory');
		}

		input.value = '';
	}

	private getNativeFilePath(file: File): string | null {
		const fileWithPath = file as File & { path?: string };
		if (typeof fileWithPath.path === 'string' && fileWithPath.path.trim()) {
			return fileWithPath.path.trim();
		}
		return null;
	}

	private getParentDirectory(filePath: string): string {
		const separatorIndex = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'));
		if (separatorIndex <= 0) {
			return filePath;
		}
		return filePath.slice(0, separatorIndex);
	}

	private showPathUnavailableMessage(target: 'video' | 'directory') {
		if (target === 'video') {
			this.showToast('warning', 'Path unavailable', 'Selected video file name is available, but this browser does not expose the full local path. Paste the full video path manually.');
			return;
		}
		if (this.loadSourceMode() === 'saved_session_dir') {
			this.showToast('warning', 'Path unavailable', 'Selected folder contents are available, but this browser does not expose the full local directory path. Paste the full saved session directory path manually.');
			return;
		}
		this.showToast('warning', 'Path unavailable', 'Selected folder contents are available, but this browser does not expose the full local directory path. Paste the full frames directory path manually.');
	}

	isApiUrlDirty(): boolean {
		return this.apiUrlInput().trim() !== this.backend.getApiUrl();
	}

	async initVideo() {
		const enteredPath = this.videoDir().trim().replace(/^['\"]|['\"]$/g, '');
		if (!enteredPath) {
			if (this.loadSourceMode() === 'saved_session_dir') {
				this.showToast('warning', 'Missing session path', 'Enter a saved session directory path or browse for one.');
				return;
			}
			if (this.loadSourceMode() === 'video_file') {
				this.showToast('warning', 'Missing video path', 'Enter a video file path or browse for one.');
				return;
			}
			this.showToast('warning', 'Missing frames path', 'Enter a frames directory path or browse for one.');
			return;
		}

		this.videoDir.set(enteredPath);
		const res = await this.runBackendJob<VideoInitStateResponse>(
			'Loading video',
			() => firstValueFrom(this.backend.initVideoState(enteredPath)),
		);
		if (!res) {
			return;
		}
		this.numFrames.set(res.num_frames);
		this.targetFrameIdx.set(0);
		this.displayedFrameIdx.set(-1);
		this.saveName.set('');
		this.trackedPoints.set([]);
		this.masks.set(new Map());
		this.points.set(new Map());
		this.liveEditedObjectFrames.set(new Map());
		this.clearFrameCaches();
		this.objects.set([{ id: 1, name: 'Object 1', color: this.getRandomColor() }]);
		this.selectedObjectId.set(1);
		this.hasManifestMasks.set(Boolean(res.restored_session?.has_mask_manifest));
		this.updateStateEpoch(res.state_epoch, 'video init');
		this.restoreInteractiveSessionState(res.restored_session);
		this.resetDebugState();
		this.isInitialized.set(true);
		const restoredWarnings = res.restored_session?.interactive_state_warnings || [];
		if (restoredWarnings.length > 0) {
			this.showToast('warning', 'Session partially restored', restoredWarnings.slice(0, 2).join(' '));
		}
	}

	private restoreInteractiveSessionState(restored: RestoredSessionPayload | null | undefined): void {
		if (!restored?.interactive_state) {
			return;
		}
		const interactive = restored.interactive_state;
		const restoredObjects = this.normalizeInteractiveObjects(interactive.objects || []);
		if (restoredObjects.length > 0) {
			this.objects.set(restoredObjects);
		}

		const restoredPoints = this.deserializePoints(interactive.points || []);
		if (restoredPoints.size > 0) {
			this.points.set(restoredPoints);
		}

		const { masks, liveEditedFrames } = this.deserializeLiveMasks(interactive.live_masks || []);
		if (masks.size > 0) {
			this.masks.set(masks);
			this.liveEditedObjectFrames.set(liveEditedFrames);
		}
		this.ensureObjectsForRestoredState();

		const availableObjectIds = new Set(this.objects().map((objectEntry) => objectEntry.id));
		const requestedObjectId = interactive.selected_object_id ?? null;
		if (requestedObjectId !== null && availableObjectIds.has(requestedObjectId)) {
			this.selectedObjectId.set(requestedObjectId);
		} else if (this.objects().length > 0) {
			this.selectedObjectId.set(this.objects()[0].id);
		}

		if (interactive.interaction_mode === 'positive' || interactive.interaction_mode === 'negative') {
			this.interactionMode.set(interactive.interaction_mode);
		}
		if (
			typeof interactive.current_frame_idx === 'number'
			&& Number.isFinite(interactive.current_frame_idx)
			&& interactive.current_frame_idx >= 0
			&& interactive.current_frame_idx < this.numFrames()
		) {
			this.targetFrameIdx.set(Math.trunc(interactive.current_frame_idx));
		}
	}

	private ensureObjectsForRestoredState(): void {
		const existing = new Map(this.objects().map((entry) => [entry.id, entry]));
		const requiredObjectIds = new Set<number>();
		this.points().forEach((framePoints) => {
			framePoints.forEach((_objPoints, objId) => requiredObjectIds.add(objId));
		});
		this.masks().forEach((frameMasks) => {
			frameMasks.forEach((_mask, objId) => requiredObjectIds.add(objId));
		});
		let changed = false;
		for (const objId of requiredObjectIds) {
			if (existing.has(objId)) {
				continue;
			}
			existing.set(objId, {
				id: objId,
				name: `Object ${objId}`,
				color: this.getRandomColor(),
			});
			changed = true;
		}
		if (changed) {
			this.objects.set(Array.from(existing.values()).sort((a, b) => a.id - b.id));
		}
	}

	private normalizeInteractiveObjects(objects: InteractiveObject[]): MaskObject[] {
		const normalized: MaskObject[] = [];
		const seen = new Set<number>();
		for (const candidate of objects) {
			if (!Number.isInteger(candidate.id) || candidate.id <= 0 || seen.has(candidate.id)) {
				continue;
			}
			seen.add(candidate.id);
			normalized.push({
				id: candidate.id,
				name: (candidate.name || '').trim() || `Object ${candidate.id}`,
				color: this.normalizeObjectColor(candidate.color),
			});
		}
		return normalized;
	}

	private normalizeObjectColor(color: string | undefined): string {
		if (typeof color === 'string' && /^#[0-9a-fA-F]{6}$/.test(color.trim())) {
			return color.trim();
		}
		return this.getRandomColor();
	}

	private deserializePoints(points: InteractivePoint[]): Map<number, Map<number, Point[]>> {
		const pointsByFrame = new Map<number, Map<number, Point[]>>();
		for (const point of points) {
			if (
				!Number.isFinite(point.frame_idx)
				|| !Number.isFinite(point.obj_id)
				|| !Number.isFinite(point.x)
				|| !Number.isFinite(point.y)
				|| (point.label !== 0 && point.label !== 1)
			) {
				continue;
			}
			const frameIdx = Math.trunc(point.frame_idx);
			const objId = Math.trunc(point.obj_id);
			if (frameIdx < 0 || objId <= 0) {
				continue;
			}
			let frameMap = pointsByFrame.get(frameIdx);
			if (!frameMap) {
				frameMap = new Map<number, Point[]>();
				pointsByFrame.set(frameIdx, frameMap);
			}
			const objPoints = frameMap.get(objId) || [];
			objPoints.push({
				x: point.x,
				y: point.y,
				label: point.label,
			});
			frameMap.set(objId, objPoints);
		}
		return pointsByFrame;
	}

	private deserializeLiveMasks(
		liveMasks: InteractiveMaskRle[],
	): { masks: Map<number, Map<number, boolean[][]>>; liveEditedFrames: Map<number, Set<number>> } {
		const masksByFrame = new Map<number, Map<number, boolean[][]>>();
		const editedByFrame = new Map<number, Set<number>>();
		for (const entry of liveMasks) {
			const decodedMask = this.decodeMaskFromCounts(entry);
			if (!decodedMask) {
				continue;
			}
			const frameIdx = Math.trunc(entry.frame_idx);
			const objId = Math.trunc(entry.obj_id);
			if (frameIdx < 0 || objId <= 0) {
				continue;
			}
			let frameMasks = masksByFrame.get(frameIdx);
			if (!frameMasks) {
				frameMasks = new Map<number, boolean[][]>();
				masksByFrame.set(frameIdx, frameMasks);
			}
			frameMasks.set(objId, decodedMask);

			const editedSet = editedByFrame.get(frameIdx) || new Set<number>();
			editedSet.add(objId);
			editedByFrame.set(frameIdx, editedSet);
		}
		return { masks: masksByFrame, liveEditedFrames: editedByFrame };
	}

	private buildInteractiveStateSnapshot(): VideoSaveInteractiveState {
		const objects = this.objects().map((entry) => ({
			id: entry.id,
			name: entry.name,
			color: entry.color,
		}));
		const points: InteractivePoint[] = [];
		this.points().forEach((framePoints, frameIdx) => {
			framePoints.forEach((objPoints, objId) => {
				for (const point of objPoints) {
					points.push({
						frame_idx: frameIdx,
						obj_id: objId,
						x: point.x,
						y: point.y,
						label: point.label === 1 ? 1 : 0,
					});
				}
			});
		});

		const liveMasks: InteractiveMaskRle[] = [];
		this.masks().forEach((frameMasks, frameIdx) => {
			frameMasks.forEach((mask, objId) => {
				const encoded = this.encodeMaskToCounts(mask);
				if (!encoded) {
					return;
				}
				liveMasks.push({
					frame_idx: frameIdx,
					obj_id: objId,
					height: encoded.height,
					width: encoded.width,
					counts: encoded.counts,
				});
			});
		});

		return {
			version: 1,
			objects,
			selected_object_id: this.selectedObjectId(),
			interaction_mode: this.interactionMode(),
			current_frame_idx: this.targetFrameIdx(),
			points,
			live_masks: liveMasks,
		};
	}

	private encodeMaskToCounts(mask: boolean[][]): { width: number; height: number; counts: number[] } | null {
		const normalizedMask = this.normalizeMask2d(mask);
		if (!normalizedMask || normalizedMask.length === 0 || normalizedMask[0].length === 0) {
			return null;
		}
		const height = normalizedMask.length;
		const width = normalizedMask[0].length;
		const counts: number[] = [];
		let currentValue = false;
		let currentRun = 0;
		for (let y = 0; y < height; y++) {
			const row = normalizedMask[y];
			if (!Array.isArray(row) || row.length !== width) {
				return null;
			}
			for (let x = 0; x < width; x++) {
				const value = Boolean(row[x]);
				if (value === currentValue) {
					currentRun += 1;
					continue;
				}
				counts.push(currentRun);
				currentRun = 1;
				currentValue = value;
			}
		}
		counts.push(currentRun);
		return { width, height, counts };
	}

	private decodeMaskFromCounts(maskRle: InteractiveMaskRle): boolean[][] | null {
		const height = Math.trunc(maskRle.height);
		const width = Math.trunc(maskRle.width);
		if (height <= 0 || width <= 0) {
			return null;
		}
		if (!Array.isArray(maskRle.counts) || maskRle.counts.length === 0) {
			return null;
		}
		const totalPixels = width * height;
		const flatMask = new Array<boolean>(totalPixels).fill(false);
		let index = 0;
		let foreground = false;
		for (const rawCount of maskRle.counts) {
			const count = Math.trunc(rawCount);
			if (!Number.isFinite(count) || count < 0) {
				return null;
			}
			const end = index + count;
			if (end > totalPixels) {
				return null;
			}
			if (foreground) {
				for (let cursor = index; cursor < end; cursor++) {
					flatMask[cursor] = true;
				}
			}
			index = end;
			foreground = !foreground;
		}
		if (index !== totalPixels) {
			return null;
		}
		const mask2d: boolean[][] = [];
		for (let y = 0; y < height; y++) {
			const rowStart = y * width;
			mask2d.push(flatMask.slice(rowStart, rowStart + width));
		}
		return mask2d;
	}

	private clearFrameCaches(): void {
		this.frameImageCache.clear();
		this.maskDataCache.clear();
		this.currentBaseImage = null;
		this.currentMaskObjects = {};
		this.frameLoadToken++;
		if (this.frameLoadAnimationId !== null) {
			cancelAnimationFrame(this.frameLoadAnimationId);
			this.frameLoadAnimationId = null;
		}
	}

	private scheduleFrameLoad(frameIdx: number) {
		this.pendingFrameIdx = frameIdx;
		if (this.frameLoadAnimationId !== null) {
			return;
		}
		this.frameLoadAnimationId = requestAnimationFrame(() => {
			this.frameLoadAnimationId = null;
			const nextFrameIdx = this.pendingFrameIdx;
			this.pendingFrameIdx = null;
			if (nextFrameIdx !== null) {
				this.loadFrame(nextFrameIdx);
			}
		});
	}

	loadFrame(frameIdx: number) {
		if (!this.canvasRef?.nativeElement) {
			return;
		}

		const token = ++this.frameLoadToken;
		this.currentMaskObjects = {};
		const cachedImage = this.frameImageCache.get(frameIdx);
		if (cachedImage?.complete) {
			this.paintLoadedFrame(cachedImage, frameIdx, token);
			return;
		}

		this.isFrameLoading.set(true);
		const image = new Image();
		const frameUrl = this.backend.getVideoFrameUrl(frameIdx);

		image.onerror = () => {
			if (token !== this.frameLoadToken) {
				return;
			}
			console.error(`Failed to load frame image: ${frameUrl}`);
			this.isFrameLoading.set(false);
		};

		image.onload = async () => {
			if (token !== this.frameLoadToken) {
				return;
			}
			this.cacheFrameImage(frameIdx, image);
			this.paintLoadedFrame(image, frameIdx, token);
		};

		image.src = frameUrl;
	}

	private async paintLoadedFrame(image: HTMLImageElement, frameIdx: number, token: number) {
		this.currentBaseImage = image;
		this.ensureCanvasSize(image.width, image.height);
		this.draw(image, frameIdx);
		this.displayedFrameIdx.set(frameIdx);
		this.isFrameLoading.set(false);
		this.preloadNeighborFrames(frameIdx);

		await this.loadMaskDataForFrame(frameIdx, token);
		if (token !== this.frameLoadToken) {
			return;
		}
		this.draw(image, frameIdx);
	}

	private cacheFrameImage(frameIdx: number, image: HTMLImageElement) {
		if (this.frameImageCache.has(frameIdx)) {
			this.frameImageCache.delete(frameIdx);
		}
		this.frameImageCache.set(frameIdx, image);
		while (this.frameImageCache.size > this.maxFrameCacheSize) {
			const oldestKey = this.frameImageCache.keys().next().value;
			if (oldestKey === undefined) {
				break;
			}
			this.frameImageCache.delete(oldestKey);
			this.maskDataCache.delete(oldestKey);
		}
	}

	private preloadNeighborFrames(frameIdx: number) {
		for (const neighborIdx of [frameIdx + 1, frameIdx - 1]) {
			if (neighborIdx < 0 || neighborIdx >= this.numFrames() || this.frameImageCache.has(neighborIdx)) {
				continue;
			}
			const image = new Image();
			image.onload = () => this.cacheFrameImage(neighborIdx, image);
			image.src = this.backend.getVideoFrameUrl(neighborIdx);
		}
	}

	private ensureCanvasSize(width: number, height: number) {
		if (!this.canvasRef?.nativeElement) {
			return;
		}
		const canvas = this.canvasRef.nativeElement;
		if (canvas.width !== width || canvas.height !== height) {
			canvas.width = width;
			canvas.height = height;
		}
	}

	private async loadMaskDataForFrame(frameIdx: number, token: number) {
		if (!this.hasManifestMasks()) {
			this.currentMaskObjects = {};
			return;
		}
		const cachedMaskData = this.maskDataCache.get(frameIdx);
		if (cachedMaskData) {
			this.currentMaskObjects = cachedMaskData;
			return;
		}

		try {
			const response = await firstValueFrom(this.backend.getVideoMaskData(frameIdx));
			if (token !== this.frameLoadToken) {
				return;
			}

			if ((response as any)?.error) {
				this.currentMaskObjects = {};
				return;
			}
			this.currentMaskObjects = response.objects || {};
			this.maskDataCache.set(frameIdx, this.currentMaskObjects);
		} catch (error) {
			console.error(error);
			this.currentMaskObjects = {};
		}
	}

	private drawCurrentFrame() {
		if (!this.currentBaseImage || !this.canvasRef?.nativeElement) {
			return;
		}
		const frameIdx = this.displayedFrameIdx();
		if (frameIdx < 0) {
			return;
		}
		this.draw(this.currentBaseImage, frameIdx);
	}

	draw(img: HTMLImageElement, frameIdx: number) {
		const canvas = this.canvasRef.nativeElement;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const liveFrameMasks = this.masks().get(frameIdx);
		const liveEditedObjectIds = this.liveEditedObjectFrames().get(frameIdx) ?? new Set<number>();

		ctx.clearRect(0, 0, canvas.width, canvas.height);
		ctx.drawImage(img, 0, 0);

		if (this.hasManifestMasks() && Object.keys(this.currentMaskObjects).length > 0) {
			for (const [objIdStr, maskData] of Object.entries(this.currentMaskObjects)) {
				const objId = parseInt(objIdStr, 10);
				if (liveEditedObjectIds.has(objId)) {
					continue;
				}
				const obj = this.objects().find((candidate) => candidate.id === objId);
				this.drawMaskFromRle(ctx, maskData, obj?.color || '#ff9800');
			}
		}

		if (liveFrameMasks) {
			liveFrameMasks.forEach((mask, objId) => {
				const normalizedMask = this.normalizeMask2d(mask);
				if (!normalizedMask || !this.maskHasForeground(normalizedMask)) {
					return;
				}
				const obj = this.objects().find((candidate) => candidate.id === objId);
				if (obj) {
					this.drawMask(ctx, normalizedMask, obj.color);
				}
			});
		}

		const framePoints = this.points().get(frameIdx);
		if (framePoints) {
			framePoints.forEach((frameObjPoints) => {
				frameObjPoints.forEach((point) => this.drawPoint(ctx, point));
			});
		}

		const selectedObjectId = this.selectedObjectId();
		let maskSource: DebugMaskSource = 'none';
		if (selectedObjectId !== null) {
			const selectedLiveMask = liveFrameMasks?.get(selectedObjectId);
			const normalizedLiveMask = selectedLiveMask ? this.normalizeMask2d(selectedLiveMask) : null;
			const hasLiveMask = Boolean(normalizedLiveMask && this.maskHasForeground(normalizedLiveMask));
			if (hasLiveMask) {
				maskSource = 'live';
			} else if (!this.isObjectLiveEdited(frameIdx, selectedObjectId) && Boolean(this.currentMaskObjects[String(selectedObjectId)])) {
				maskSource = 'manifest';
			}
		}
		this.lastMaskSource.set(maskSource);

		this.drawTrackingOverlay(ctx, frameIdx);
	}

	drawMask(ctx: CanvasRenderingContext2D, mask: boolean[][], color: string) {
		const normalizedMask = this.normalizeMask2d(mask);
		if (!normalizedMask?.length || !normalizedMask[0]?.length) {
			return;
		}
		const width = normalizedMask[0].length;
		const height = normalizedMask.length;
		const imageData = ctx.createImageData(width, height);
		const data = imageData.data;
		const [r, g, b] = this.hexToRgb(color);

		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				if (!normalizedMask[y][x]) {
					continue;
				}
				const index = (y * width + x) * 4;
				data[index] = r;
				data[index + 1] = g;
				data[index + 2] = b;
				data[index + 3] = 120;
			}
		}

		const tempCanvas = document.createElement('canvas');
		tempCanvas.width = width;
		tempCanvas.height = height;
		tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
		ctx.drawImage(tempCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
	}

	private normalizeMask2d(mask: unknown): boolean[][] | null {
		let candidate: any = mask;
		while (Array.isArray(candidate) && candidate.length > 0 && Array.isArray(candidate[0]) && Array.isArray(candidate[0][0])) {
			candidate = candidate[0];
		}
		if (!Array.isArray(candidate) || candidate.length === 0 || !Array.isArray(candidate[0])) {
			return null;
		}
		return candidate as boolean[][];
	}

	private maskHasForeground(mask: boolean[][]): boolean {
		for (const row of mask) {
			for (const value of row) {
				if (value) {
					return true;
				}
			}
		}
		return false;
	}

	private drawMaskFromRle(ctx: CanvasRenderingContext2D, maskData: VideoMaskObjectData, color: string) {
		const size = maskData.size;
		if (!Array.isArray(size) || size.length !== 2) {
			return;
		}
		const height = Number(size[0]);
		const width = Number(size[1]);
		if (!Number.isFinite(height) || !Number.isFinite(width) || height <= 0 || width <= 0) {
			return;
		}

		const imageData = ctx.createImageData(width, height);
		const data = imageData.data;
		const [r, g, b] = this.hexToRgb(color);

		for (const run of maskData.rle || []) {
			if (!Array.isArray(run) || run.length !== 2) {
				continue;
			}
			const start = Math.max(0, Number(run[0]) | 0);
			const length = Math.max(0, Number(run[1]) | 0);
			const end = Math.min(width * height, start + length);
			for (let index = start; index < end; index++) {
				const pixelOffset = index * 4;
				data[pixelOffset] = r;
				data[pixelOffset + 1] = g;
				data[pixelOffset + 2] = b;
				data[pixelOffset + 3] = 120;
			}
		}

		const tempCanvas = document.createElement('canvas');
		tempCanvas.width = width;
		tempCanvas.height = height;
		tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
		ctx.drawImage(tempCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
	}

	drawPoint(ctx: CanvasRenderingContext2D, point: Point) {
		ctx.beginPath();
		ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
		ctx.fillStyle = point.label === 1 ? '#00ff00' : '#ff0000';
		ctx.fill();
		ctx.strokeStyle = 'white';
		ctx.lineWidth = 2;
		ctx.stroke();
	}

	private drawTrackingOverlay(ctx: CanvasRenderingContext2D, frameIdx: number) {
		const style = this.trackingOverlayStyle();
		if (!this.trackedPoints().length) {
			return;
		}

		for (const pointSeries of this.trackedPoints()) {
			if (frameIdx < 0 || frameIdx >= pointSeries.tracks.length) {
				continue;
			}

			const obj = this.objects().find((candidate) => candidate.id === pointSeries.obj_id);
			const color = obj?.color || '#ffd54f';
			const visibleNow = pointSeries.visibility[frameIdx] !== false;

			if (style !== 'point') {
				const trailStart = style === 'short'
					? Math.max(pointSeries.source_frame_idx, frameIdx - 20)
					: Math.max(pointSeries.source_frame_idx, 0);

				ctx.beginPath();
				let started = false;
				for (let idx = trailStart; idx <= frameIdx; idx++) {
					if (idx < 0 || idx >= pointSeries.tracks.length || pointSeries.visibility[idx] === false) {
						continue;
					}
					const [x, y] = pointSeries.tracks[idx];
					if (!started) {
						ctx.moveTo(x, y);
						started = true;
					} else {
						ctx.lineTo(x, y);
					}
				}
				ctx.strokeStyle = color;
				ctx.lineWidth = 2;
				ctx.globalAlpha = style === 'short' ? 0.75 : 0.55;
				ctx.stroke();
				ctx.globalAlpha = 1;
			}

			if (!visibleNow) {
				continue;
			}

			const [currentX, currentY] = pointSeries.tracks[frameIdx];
			ctx.beginPath();
			ctx.arc(currentX, currentY, 4.5, 0, 2 * Math.PI);
			ctx.fillStyle = color;
			ctx.fill();
			ctx.strokeStyle = '#ffffff';
			ctx.lineWidth = 1.5;
			ctx.stroke();
		}
	}

	onCanvasClick(event: MouseEvent) {
		if (!this.isInitialized() || this.selectedObjectId() === null || this.isFrameLoading() || this.isPointRequestInFlight() || !this.currentBaseImage) {
			return;
		}

		const rect = this.canvasRef.nativeElement.getBoundingClientRect();
		const scaleX = this.canvasRef.nativeElement.width / rect.width;
		const scaleY = this.canvasRef.nativeElement.height / rect.height;

		const x = (event.clientX - rect.left) * scaleX;
		const y = (event.clientY - rect.top) * scaleY;
		const label = this.interactionMode() === 'positive' ? 1 : 0;
		const frameIdx = this.displayedFrameIdx();
		if (frameIdx < 0) {
			return;
		}
		this.addPoint(x, y, label, frameIdx);
	}

	async addPoint(x: number, y: number, label: number, frameIdx: number) {
		const objId = this.selectedObjectId();
		if (objId === null) return;

		const wasLiveEditedBeforeRequest = this.isObjectLiveEdited(frameIdx, objId);
		const pointsMap = new Map(this.points());
		const framePointsMap = new Map(pointsMap.get(frameIdx) || new Map<number, Point[]>());
		const previousObjectPoints = framePointsMap.get(objId) || [];
		const objectPoints = [...previousObjectPoints, { x, y, label }];
		framePointsMap.set(objId, objectPoints);
		pointsMap.set(frameIdx, framePointsMap);
		this.points.set(pointsMap);
		const frameObjectPoints = objectPoints.map((point) => [point.x, point.y]);
		const frameObjectLabels = objectPoints.map((point) => point.label);
		const requestFrameIdx = frameIdx;
		const expectedEpoch = this.stateEpoch();
		this.lastClickRequestFrameIdx.set(requestFrameIdx);
		this.lastDebugObjectId.set(objId);
		this.lastMaskPixelCount.set(null);
		this.lastBackendResponseFrameIdx.set(null);
		this.lastBackendResponseFrameFile.set('n/a');
		this.lastBackendResponseStateEpoch.set(null);
		this.lastFallbackUsed.set(false);
		this.lastDiscardReason.set(null);

		const request: VideoAddPointsOrBoxRequest = {
			frame_idx: requestFrameIdx,
			obj_id: objId,
			points: frameObjectPoints,
			labels: frameObjectLabels,
			clear_old_points: true
		};

		try {
			this.isPointRequestInFlight.set(true);
			const response = await firstValueFrom(this.backend.addNewPointsOrBox(request));
			if (
				(response as any)?.error ||
				typeof (response as any)?.request_frame_idx !== 'number' ||
				typeof (response as any)?.frame_idx !== 'number' ||
				typeof (response as any)?.frame_file !== 'string' ||
				typeof (response as any)?.state_epoch !== 'number' ||
				!Array.isArray((response as any)?.out_obj_ids) ||
				!Array.isArray((response as any)?.out_masks) ||
				typeof (response as any)?.mask_pixel_counts !== 'object'
			) {
				throw new Error((response as any)?.error || 'Invalid mask response');
			}
			const responseStateEpoch = Math.trunc(response.state_epoch);
			this.lastBackendResponseStateEpoch.set(responseStateEpoch);
			if (responseStateEpoch !== expectedEpoch) {
				this.updateStateEpoch(responseStateEpoch, 'add_new_points_or_box mismatch response');
				this.lastDiscardReason.set(
					`Discarded stale response due to epoch mismatch (expected ${expectedEpoch}, got ${responseStateEpoch}).`
				);
				throw new Error(this.lastDiscardReason() || 'Discarded stale response');
			}
			const responseRequestFrameIdx = Math.trunc(response.request_frame_idx);
			const responseFrameIdx = Math.trunc(response.frame_idx);
			this.lastBackendResponseFrameIdx.set(responseFrameIdx);
			this.lastBackendResponseFrameFile.set(response.frame_file || 'n/a');
			if (responseRequestFrameIdx !== requestFrameIdx || responseFrameIdx !== requestFrameIdx) {
				this.lastDiscardReason.set(
					`Discarded response due to frame mismatch (request=${requestFrameIdx}, response_request=${responseRequestFrameIdx}, response_frame=${responseFrameIdx}).`
				);
				throw new Error(this.lastDiscardReason() || 'Discarded mismatched response frame');
			}
			if (this.displayedFrameIdx() !== requestFrameIdx) {
				this.lastDiscardReason.set(
					`Discarded response because displayed frame moved from ${requestFrameIdx} to ${this.displayedFrameIdx()}.`
				);
				throw new Error(this.lastDiscardReason() || 'Displayed frame changed during request');
			}

			const maskPixelCount = this.getMaskPixelCount(response.mask_pixel_counts, objId);
			this.lastMaskPixelCount.set(maskPixelCount);
			this.lastFallbackUsed.set(Boolean(response.single_frame_fallback_used));

			const masksMap = new Map(this.masks());
			const frameMasksMap = new Map(masksMap.get(requestFrameIdx) || new Map<number, boolean[][]>());
			response.out_obj_ids.forEach((id, index) => {
				frameMasksMap.set(id, response.out_masks[index]);
			});
			masksMap.set(requestFrameIdx, frameMasksMap);
			this.markObjectAsLiveEdited(requestFrameIdx, objId);
			this.masks.set(masksMap);
			this.drawCurrentFrame();
		} catch (error) {
			console.error(error);
			const rollbackPointsMap = new Map(this.points());
			const rollbackFramePointsMap = new Map(rollbackPointsMap.get(frameIdx) || new Map<number, Point[]>());
			if (previousObjectPoints.length > 0) {
				rollbackFramePointsMap.set(objId, previousObjectPoints);
			} else {
				rollbackFramePointsMap.delete(objId);
			}
			if (rollbackFramePointsMap.size === 0) {
				rollbackPointsMap.delete(frameIdx);
			} else {
				rollbackPointsMap.set(frameIdx, rollbackFramePointsMap);
			}
			if (!wasLiveEditedBeforeRequest) {
				this.unmarkObjectAsLiveEdited(frameIdx, objId);
			}
			this.points.set(rollbackPointsMap);
			this.drawCurrentFrame();
		} finally {
			this.isPointRequestInFlight.set(false);
		}
	}

	onScrubberFrameChange(value: number | string) {
		const parsed = Number(value);
		if (!Number.isFinite(parsed)) {
			return;
		}
		const frameIdx = Math.trunc(parsed);
		const maxFrame = Math.max(this.numFrames() - 1, 0);
		this.targetFrameIdx.set(Math.min(Math.max(frameIdx, 0), maxFrame));
	}

	addObject() {
		const newId = this.objects().length + 1;
		const newObject: MaskObject = {
			id: newId,
			name: `Object ${newId}`,
			color: this.getRandomColor()
		};
		this.objects.update((existing) => [...existing, newObject]);
		this.selectedObjectId.set(newId);
	}

	removeObject() {
		const id = this.selectedObjectId();
		if (id === null) return;

		this.backend.removeObject(id).subscribe(() => {
			this.objects.update((existing) => existing.filter((entry) => entry.id !== id));
			this.removeObjectFromFrameMaps(id);
			this.selectedObjectId.set(this.objects().length > 0 ? this.objects()[0].id : null);
			this.drawCurrentFrame();
		});
	}

	async removeAllObjects() {
		const objectIds = this.objects().map((entry) => entry.id);
		if (objectIds.length === 0) {
			return;
		}

		try {
			await Promise.all(objectIds.map((id) => firstValueFrom(this.backend.removeObject(id))));
			this.objects.set([]);
			this.selectedObjectId.set(null);
			this.masks.set(new Map());
			this.points.set(new Map());
			this.liveEditedObjectFrames.set(new Map());
			this.trackedPoints.set([]);
			this.drawCurrentFrame();
		} catch (error) {
			console.error(error);
			this.showToast('error', 'Remove all failed', this.getErrorMessage(error, 'Failed to remove all objects.'));
		}
	}

	private removeObjectFromFrameMaps(objectId: number) {
		const nextMasks = new Map(this.masks());
		nextMasks.forEach((frameMap) => frameMap.delete(objectId));
		this.masks.set(nextMasks);

		const nextPoints = new Map(this.points());
		nextPoints.forEach((frameMap) => frameMap.delete(objectId));
		this.points.set(nextPoints);

		this.trackedPoints.set(this.trackedPoints().filter((series) => series.obj_id !== objectId));
	}

	async propagate() {
		const response = await this.runBackendJob<VideoPropagateResponse>(
			'Propagating masks',
			() => firstValueFrom(this.backend.propagateInVideo({
				include_masks_in_response: false,
				include_saved_mask_paths: false
			})),
		);
		if (!response) {
			return;
		}
		this.updateStateEpoch(response.state_epoch, 'propagation');
		this.hasManifestMasks.set(Boolean(response.mask_manifest_path));
		this.maskDataCache.clear();
		this.scheduleFrameLoad(this.targetFrameIdx());
	}

	async runTracking() {
		const response = await this.runBackendJob<TrackPromptPointsResponse>(
			'Tracking prompt points',
			() => firstValueFrom(this.backend.trackPromptPoints({
				add_support_grid: this.trackingUseSupportGrid()
			})),
		);
		if (!response) {
			return;
		}
		this.updateStateEpoch(response.state_epoch, 'tracking restore');

		const trackedSeries: TrackedPointSeries[] = response.points.map((point, index) => ({
			...point,
			tracks: response.tracks[index] || [],
			visibility: response.visibility[index] || []
		}));
		this.trackedPoints.set(trackedSeries);
		this.drawCurrentFrame();
	}

	clearMasks() {
		this.backend.resetVideoState().subscribe((response) => {
			this.updateStateEpoch(response?.state_epoch, 'reset');
			this.hasManifestMasks.set(false);
			this.trackedPoints.set([]);
			this.masks.set(new Map());
			this.points.set(new Map());
			this.liveEditedObjectFrames.set(new Map());
			this.maskDataCache.clear();
			this.scheduleFrameLoad(this.targetFrameIdx());
		});
	}

	save() {
		const name = this.saveName().trim();
		if (!name) {
			this.showToast('warning', 'Missing save name', 'Enter a name for this saved session.');
			return;
		}
		const interactiveState = this.buildInteractiveStateSnapshot();
		this.isLoading.set(true);
		firstValueFrom(this.backend.saveVideoSession(name, interactiveState))
			.then((response: VideoSaveResponse) => {
				this.updateStateEpoch(response.state_epoch, 'save');
				this.saveName.set(response.name);
				this.showToast('success', 'Session saved', `Saved to ${response.saved_path}`);
			})
			.catch((error: any) => {
				console.error(error);
				this.showToast('error', 'Save failed', this.getErrorMessage(error, 'Session save failed'));
			})
			.finally(() => {
				this.isLoading.set(false);
			});
	}

	getRandomColor() {
		const letters = '0123456789ABCDEF';
		let color = '#';
		for (let i = 0; i < 6; i++) {
			color += letters[Math.floor(Math.random() * 16)];
		}
		return color;
	}

	hexToRgb(hex: string) {
		const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
		return result ? [
			parseInt(result[1], 16),
			parseInt(result[2], 16),
			parseInt(result[3], 16)
		] : [0, 0, 0];
	}
}
