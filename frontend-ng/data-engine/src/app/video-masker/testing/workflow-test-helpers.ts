import { TestBed } from '@angular/core/testing';
import { Observable, map, of } from 'rxjs';
import { vi } from 'vitest';
import {
  BackendService,
  VideoAddPointsResponse,
  VideoInitStateResponse,
  AnnotationSnapshot,
  VideoObject,
} from '../../services/backend.service';
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
    getVideoInfo: vi.fn(),
    trackPromptPoints: vi.fn(),
    propagateInVideo: vi.fn(),
    getTrackingResult: vi.fn(),
    clearJobResult: vi.fn(() => of({ cleared: true })),
    getApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
    setApiUrl: vi.fn((value: string) => value),
    resetApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
    saveVideoSession: vi.fn(),
    removeObject: vi.fn<(id: number) => Observable<AnnotationSnapshot>>(),
    upsertObject: vi.fn<(object: VideoObject) => Observable<AnnotationSnapshot>>(),
    resetVideoState: vi.fn(() => of({ state_epoch: 4 })),
  };
  backendMock.getVideoInfo.mockImplementation(() => {
    const job = backendMock.getJob.mock.results.at(-1)?.value as Observable<{
      job: { result: VideoInitStateResponse };
    }>;
    return job.pipe(
      map(({ job: { result } }) => ({
        session_id: 'session-1',
        state_epoch: result.state_epoch,
        num_frames: result.num_frames,
        frame_files: Array.from({ length: result.num_frames }, (_, frame) => `${frame}.jpg`),
        annotations: result.restored_session?.annotations ?? {
          version: 1,
          revision: 0,
          next_object_id: 1,
          objects: {},
          prompts: [],
        },
      })),
    );
  });
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
  store.applyAnnotationSnapshot(
    {
      session_id: 'session-1',
      state_epoch: 3,
      annotations: {
        version: 1,
        revision: 0,
        next_object_id: 2,
        objects: { '1': { id: 1, label: 'Object 1', color: '#ff0000' } },
        prompts: [],
      },
    },
    'http://127.0.0.1:8000',
    true,
  );
  store.displayedFrameIdx.set(5);
}

export function objectSnapshot(
  store: VideoMaskerStateStore,
  objects: VideoObject[],
): AnnotationSnapshot {
  const ids = new Set(objects.map((object) => object.id));
  return {
    session_id: store.sessionId(),
    state_epoch: store.stateEpoch(),
    annotations: {
      version: 1,
      revision: (store.annotationRevision() ?? 0) + 1,
      next_object_id: Math.max(
        store.nextObjectId,
        store.annotations()?.next_object_id ?? 1,
        ...objects.map((object) => object.id + 1),
      ),
      objects: Object.fromEntries(objects.map((object) => [object.id, object])),
      prompts: (store.annotations()?.prompts ?? []).filter((prompt) => ids.has(prompt.obj_id)),
    },
  };
}

export function configureObjectMutationMocks(
  backend: ReturnType<typeof createBackendMocks>['backendMock'],
  store: VideoMaskerStateStore,
): void {
  backend.upsertObject.mockImplementation((object) =>
    of(
      objectSnapshot(store, [...store.objects().filter((entry) => entry.id !== object.id), object]),
    ),
  );
  backend.removeObject.mockImplementation((id) =>
    of(
      objectSnapshot(
        store,
        store.objects().filter((object) => object.id !== id),
      ),
    ),
  );
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
  configureObjectMutationMocks(mocks.backendMock, editor.store);
  return { ...mocks, editor, objectOperations: TestBed.inject(ObjectOperationsService) };
}
