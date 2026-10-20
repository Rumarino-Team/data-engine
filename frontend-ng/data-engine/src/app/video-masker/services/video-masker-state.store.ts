import { Injectable, computed, signal, inject } from '@angular/core';
import {
  ApiHealthStatus,
  BackendJob,
  TrackPromptPointMetadata,
  VideoAnnotations,
  AnnotationSnapshot,
} from '../../services/backend.service';
import {
  DebugMaskSource,
  LoadSourceMode,
  TrackingOverlayStyle,
} from '../state/video-masker-ui.types';

import { ToastService } from './toast.service';
import { resolveStateEpoch } from '../video-masker.util';

export interface MaskObject {
  id: number;
  label: string;
  color: string;
}

export interface Point {
  x: number;
  y: number;
  label: number;
}

export interface TrackedPointSeries extends TrackPromptPointMetadata {
  tracks: number[][];
  visibility: boolean[];
}

@Injectable({ providedIn: 'root' })
export class VideoMaskerStateStore {
  videoDir = signal<string>('');
  loadSourceMode = signal<LoadSourceMode>('frames_dir');
  apiUrlInput = signal<string>('');
  isInitialized = signal<boolean>(false);
  numFrames = signal<number>(0);
  targetFrameIdx = signal<number>(0);
  displayedFrameIdx = signal<number>(-1);
  propagationStartFrameIdx = signal<number>(0);
  propagationEndFrameIdx = signal<number>(0);
  stateEpoch = signal<number>(0);

  sessionId = signal<string | null>(null);
  annotationsBackendUrl = signal<string | null>(null);
  annotations = signal<VideoAnnotations | null>(null);
  annotationRevision = computed(() => this.annotations()?.revision ?? null);
  acceptedObjects = computed(() =>
    Object.values(this.annotations()?.objects ?? {}).sort((a, b) => a.id - b.id),
  );
  acceptedPoints = computed(() => {
    const frames = new Map<number, Map<number, Point[]>>();
    for (const prompt of this.annotations()?.prompts ?? []) {
      let objects = frames.get(prompt.frame_idx);
      if (!objects) frames.set(prompt.frame_idx, (objects = new Map()));
      const points = prompt.clear_old_points === false ? (objects.get(prompt.obj_id) ?? []) : [];
      objects.set(prompt.obj_id, [
        ...points,
        ...(prompt.points ?? []).map(([x, y], index) => ({
          x,
          y,
          label: prompt.labels[index],
        })),
      ]);
    }
    for (const [frameIdx, objects] of frames) {
      for (const [objId, points] of objects) if (!points.length) objects.delete(objId);
      if (!objects.size) frames.delete(frameIdx);
    }
    return frames;
  });
  pendingObjectUpdates = signal(0);
  annotationRecoveryRequested = signal(false);
  isRecoveringAnnotations = signal(false);

  /** Replaces accepted inputs without changing navigation or other presentation state. */
  applyAnnotationSnapshot(
    info: AnnotationSnapshot,
    backendUrl: string,
    replaceSession = false,
  ): boolean {
    if (!info.session_id || !info.annotations) return false;
    if (
      !replaceSession &&
      (this.sessionId() !== info.session_id || this.annotationsBackendUrl() !== backendUrl)
    )
      return false;
    if (
      !replaceSession &&
      (info.annotations.revision < (this.annotationRevision() ?? 0) ||
        info.state_epoch < this.stateEpoch())
    )
      return false;

    this.sessionId.set(info.session_id);
    this.annotationsBackendUrl.set(backendUrl);
    this.annotations.set(structuredClone(info.annotations));
    this.objects.set(this.acceptedObjects());
    this.pendingPointEdit.set(null);
    this.updateStateEpoch(info.state_epoch, 'annotation snapshot');
    if (!this.objects().some((object) => object.id === this.selectedObjectId())) {
      this.selectedObjectId.set(this.objects()[0]?.id ?? null);
    }
    return true;
  }

  // Session-scoped bookkeeping shared by object and tracking workflows.
  trackingRevision = 0;

  invalidateTracking(): void {
    this.trackingRevision++;
    this.trackedPoints.set([]);
  }

  objects = signal<MaskObject[]>([]);
  selectedObjectId = signal<number | null>(null);
  interactionMode = signal<'positive' | 'negative'>('positive');

  readonly points = this.acceptedPoints;
  pendingPointEdit = signal<{ frameIdx: number; objId: number; points: Point[] } | null>(null);
  previewPoints = computed(() => {
    const pending = this.pendingPointEdit();
    if (!pending) return this.points();
    const frames = new Map(this.points());
    const objects = new Map(frames.get(pending.frameIdx));
    if (pending.points.length) objects.set(pending.objId, pending.points);
    else objects.delete(pending.objId);
    if (objects.size) frames.set(pending.frameIdx, objects);
    else frames.delete(pending.frameIdx);
    return frames;
  });
  objectPointGroups = computed(() => {
    const frames = Array.from(this.previewPoints().entries()).sort(([a], [b]) => a - b);
    return this.objects().map((object) => ({
      ...object,
      children: frames.flatMap(([frameIdx, objects]) =>
        (objects.get(object.id) ?? []).map((point, index) => ({
          ...point,
          frameIdx,
          number: index + 1,
        })),
      ),
    }));
  });
  hasManifestMasks = signal<boolean>(false);
  saveName = signal<string>('');

  trackingOverlayStyle = signal<TrackingOverlayStyle>('short');
  trackingUseSupportGrid = signal<boolean>(false);
  trackedPoints = signal<TrackedPointSeries[]>([]);

  isLoading = signal<boolean>(false);
  isFrameLoading = signal<boolean>(false);
  isPointRequestInFlight = signal<boolean>(false);
  isInteractionBusy = computed(
    () =>
      this.isLoading() ||
      this.isPointRequestInFlight() ||
      this.pendingObjectUpdates() > 0 ||
      this.isRecoveringAnnotations(),
  );
  apiHealthStatus = signal<ApiHealthStatus>('checking');
  activeJob = signal<BackendJob | null>(null);
  activeJobTitle = signal<string>('');
  readonly toasts = inject(ToastService).toasts;
  lastClickRequestFrameIdx = signal<number | null>(null);
  lastBackendResponseFrameIdx = signal<number | null>(null);
  lastBackendResponseFrameFile = signal<string>('n/a');
  lastBackendResponseStateEpoch = signal<number | null>(null);
  lastDebugObjectId = signal<number | null>(null);
  lastMaskPixelCount = signal<number | null>(null);
  lastFallbackUsed = signal<boolean>(false);
  lastMaskSource = signal<DebugMaskSource>('none');
  lastDiscardReason = signal<string | null>(null);

  updateStateEpoch(nextEpoch: number | undefined, source: string): void {
    const resolution = resolveStateEpoch(this.stateEpoch(), nextEpoch);
    if (!resolution) {
      return;
    }
    if (resolution.shouldClearLiveState) {
      this.lastDiscardReason.set(
        `State epoch changed (${this.stateEpoch()} -> ${resolution.normalizedEpoch}) during ${source}; invalidated mask cache.`,
      );
    }
    this.stateEpoch.set(resolution.normalizedEpoch);
  }

  resetInteractiveMaps(): void {
    this.pendingPointEdit.set(null);
  }

  resetDebugState(): void {
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
}
