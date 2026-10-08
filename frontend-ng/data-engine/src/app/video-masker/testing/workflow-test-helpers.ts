import { TestBed } from '@angular/core/testing';
import { Observable, of } from 'rxjs';
import { vi } from 'vitest';
import { BackendService, VideoAddPointsResponse } from '../../services/backend.service';
import { DesktopBridgeService } from '../../services/desktop-bridge.service';
import { BackendConnectionService } from '../services/backend-connection.service';
import { CanvasViewportService } from '../services/canvas-viewport.service';
import { EditorJobService } from '../services/editor-job.service';
import { FramePipelineService } from '../services/frame-pipeline.service';
import { MaskPropagationService } from '../services/mask-propagation.service';
import { ObjectOperationsService } from '../services/object-operations.service';
import { PointEditingService } from '../services/point-editing.service';
import { SessionService } from '../services/session.service';
import { ToastService } from '../services/toast.service';
import { TrackingService } from '../services/tracking.service';
import { VideoMaskerStateStore } from '../services/video-masker-state.store';

export const makeResponse = (
  overrides: Partial<VideoAddPointsResponse>,
): VideoAddPointsResponse => ({
  request_frame_idx: 5,
  frame_idx: 5,
  frame_file: '00005.jpg',
  out_obj_ids: [1],
  out_masks: [{ size: [2, 2], rle: [[0, 1]], bbox: [0, 0, 1, 1] }],
  mask_encoding: 'rle',
  mask_pixel_counts: { 1: 1 },
  mask_shapes: { 1: [2, 2] },
  state_epoch: 3,
  ...overrides,
});

export function createBackendMocks() {
  const backendMock = {
    addNewPointsOrBox: vi.fn(),
    health: vi.fn(() => of({ status: 'ok' })),
    initVideoState: vi.fn(),
    getJob: vi.fn(),
    trackPromptPoints: vi.fn(),
    propagateInVideo: vi.fn(),
    getTrackingResult: vi.fn(),
    clearJobResult: vi.fn(() => of({ cleared: true })),
    getApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
    setApiUrl: vi.fn((value: string) => value),
    resetApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
    saveVideoSession: vi.fn(),
    removeObject: vi.fn<(id: number) => Observable<unknown>>(() => of({})),
    resetVideoState: vi.fn(() => of({ state_epoch: 4 })),
  };
  const desktopBridgeMock = {
    isTauri: vi.fn(() => false),
    pickVideoFile: vi.fn(),
    pickFramesDirectory: vi.fn(),
  };

  return { backendMock, desktopBridgeMock };
}

export function mockProviders(mocks: ReturnType<typeof createBackendMocks>) {
  return [
    { provide: BackendService, useValue: mocks.backendMock },
    { provide: DesktopBridgeService, useValue: mocks.desktopBridgeMock },
  ];
}

export function initializeEditorStore(store: VideoMaskerStateStore): void {
  store.selectedObjectId.set(1);
  store.objects.set([{ id: 1, name: 'Object 1', color: '#ff0000' }]);
  store.stateEpoch.set(3);
  store.displayedFrameIdx.set(5);
}

/** Real workflow services share one store; no page component or view is created. */
export function createWorkflowFixture() {
  const mocks = createBackendMocks();
  TestBed.configureTestingModule({
    providers: [
      ...mockProviders(mocks),
      VideoMaskerStateStore,
      BackendConnectionService,
      CanvasViewportService,
      EditorJobService,
      FramePipelineService,
      MaskPropagationService,
      ObjectOperationsService,
      PointEditingService,
      SessionService,
      ToastService,
      TrackingService,
    ],
  });
  const editor = {
    store: TestBed.inject(VideoMaskerStateStore),
    sessions: TestBed.inject(SessionService),
    prompts: TestBed.inject(PointEditingService),
    tracking: TestBed.inject(TrackingService),
    propagation: TestBed.inject(MaskPropagationService),
    framePipeline: TestBed.inject(FramePipelineService),
  };
  initializeEditorStore(editor.store);
  return { ...mocks, editor, objectOperations: TestBed.inject(ObjectOperationsService) };
}
