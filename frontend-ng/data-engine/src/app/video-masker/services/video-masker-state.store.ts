import { Injectable, signal } from '@angular/core';
import {
  ApiHealthStatus,
  BackendJob,
  TrackPromptPointMetadata,
} from '../../services/backend.service';
import {
  AppToast,
  DebugMaskSource,
  LoadSourceMode,
  TrackingOverlayStyle,
} from '../state/video-masker-ui.types';

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

export interface SelectedPointRef {
  frameIdx: number;
  objId: number;
  pointIdx: number;
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
  stateEpoch = signal<number>(0);

  objects = signal<MaskObject[]>([]);
  selectedObjectId = signal<number | null>(null);
  selectedPoint = signal<SelectedPointRef | null>(null);
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
}
