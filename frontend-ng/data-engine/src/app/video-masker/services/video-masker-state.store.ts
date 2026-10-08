import { Injectable, computed, signal, inject } from '@angular/core';
import {
  ApiHealthStatus,
  BackendJob,
  TrackPromptPointMetadata,
  LiveMask,
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
  name: string;
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

  // Session-scoped bookkeeping shared by object and tracking workflows.
  nextObjectId = 1;
  trackingRevision = 0;

  invalidateTracking(): void {
    this.trackingRevision++;
    this.trackedPoints.set([]);
  }

  objects = signal<MaskObject[]>([]);
  selectedObjectId = signal<number | null>(null);
  interactionMode = signal<'positive' | 'negative'>('positive');

  masks = signal<Map<number, Map<number, LiveMask>>>(new Map());
  points = signal<Map<number, Map<number, Point[]>>>(new Map());
  objectPointGroups = computed(() => {
    const frames = Array.from(this.points().entries()).sort(([a], [b]) => a - b);
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
  liveEditedObjectFrames = signal<Map<number, Set<number>>>(new Map());
  hasManifestMasks = signal<boolean>(false);
  saveName = signal<string>('');

  trackingOverlayStyle = signal<TrackingOverlayStyle>('short');
  trackingUseSupportGrid = signal<boolean>(false);
  trackedPoints = signal<TrackedPointSeries[]>([]);

  isLoading = signal<boolean>(false);
  isFrameLoading = signal<boolean>(false);
  isPointRequestInFlight = signal<boolean>(false);
  isInteractionBusy = computed(() => this.isLoading() || this.isPointRequestInFlight());
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
      this.masks.set(new Map());
      this.liveEditedObjectFrames.set(new Map());
      this.lastDiscardReason.set(
        `State epoch changed (${this.stateEpoch()} -> ${resolution.normalizedEpoch}) during ${source}; cleared live masks.`,
      );
    }
    this.stateEpoch.set(resolution.normalizedEpoch);
  }

  resetInteractiveMaps(): void {
    this.masks.set(new Map());
    this.points.set(new Map());
    this.liveEditedObjectFrames.set(new Map());
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
