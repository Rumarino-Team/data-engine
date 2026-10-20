import { TestBed } from '@angular/core/testing';
import { Observable, map, of } from 'rxjs';
import { vi } from 'vitest';
import {
  BackendService,
  VideoAddPointsResponse,
  VideoInitStateResponse,
  AnnotationDelta,
  AnnotationSnapshot,
  AnnotationUpdate,
  AnnotationPrecondition,
  ObjectCreationResponse,
  VideoObject,
  LiveMask,
  VideoMaskDataResponse,
  VideoAnnotations,
  VideoInfoResponse,
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
import { Point, VideoMaskerStateStore } from '../services/video-masker-state.store';

const INITIAL_ANNOTATIONS: VideoAnnotations = {
  version: 1,
  revision: 0,
  next_object_id: 2,
  objects: { '1': { id: 1, label: 'Object 1', color: '#ff0000' } },
  prompts: [],
  mask_overrides: [],
  input_revisions: {},
};

/** Accepted state after the default point response: one prompt on frame 5. */
export const ACCEPTED_POINT_ANNOTATIONS: VideoAnnotations = {
  ...INITIAL_ANNOTATIONS,
  revision: 1,
  prompts: [{ frame_idx: 5, obj_id: 1, points: [[10, 20]], labels: [1], box: null }],
  input_revisions: { '1': 1 },
};

const promptKey = (entry: { frame_idx: number; obj_id: number }): [number, number] => [
  entry.frame_idx,
  entry.obj_id,
];

/** Builds the delta a backend would send to move `before` to `after`. */
export function annotationDelta(
  before: VideoAnnotations,
  after: VideoAnnotations,
): AnnotationDelta {
  const keyed = <T extends { frame_idx: number; obj_id: number }>(entries: T[] = []) =>
    new Map(entries.map((entry) => [promptKey(entry).join(':'), entry]));
  const diffKeyed = <T extends { frame_idx: number; obj_id: number }>(
    previous: T[] = [],
    next: T[] = [],
  ) => {
    const old = keyed(previous);
    const current = keyed(next);
    return {
      upserts: [...current]
        .filter(([key, entry]) => JSON.stringify(old.get(key)) !== JSON.stringify(entry))
        .map(([, entry]) => entry),
      removed: [...old].filter(([key]) => !current.has(key)).map(([, entry]) => promptKey(entry)),
    };
  };
  const prompts = diffKeyed(before.prompts, after.prompts);
  const masks = diffKeyed(before.mask_overrides, after.mask_overrides);
  const beforeInputs = before.input_revisions ?? {};
  const afterInputs = after.input_revisions ?? {};
  return {
    base_revision: before.revision,
    revision: after.revision,
    next_object_id: after.next_object_id,
    objects: Object.fromEntries(
      Object.entries(after.objects).filter(
        ([id, object]) => JSON.stringify(before.objects[id]) !== JSON.stringify(object),
      ),
    ),
    removed_objects: Object.keys(before.objects)
      .filter((id) => !(id in after.objects))
      .map(Number),
    prompts: prompts.upserts,
    removed_prompts: prompts.removed,
    mask_overrides: masks.upserts,
    removed_mask_overrides: masks.removed,
    input_revisions: Object.fromEntries(
      Object.entries(afterInputs).filter(([id, revision]) => beforeInputs[id] !== revision),
    ),
    removed_input_revisions: Object.keys(beforeInputs)
      .filter((id) => !(id in afterInputs))
      .map(Number),
  };
}

/** Converts an intended accepted snapshot into the mutation response the store receives. */
export function toUpdate<T extends AnnotationSnapshot>(
  store: VideoMaskerStateStore,
  snapshot: T,
): Omit<T, 'annotations'> & AnnotationUpdate {
  const { annotations, ...rest } = snapshot;
  return {
    ...rest,
    annotation_delta: annotationDelta(store.annotations() ?? INITIAL_ANNOTATIONS, annotations!),
  };
}

export const makeResponse = (
  overrides: Partial<VideoAddPointsResponse>,
): VideoAddPointsResponse => ({
  session_id: 'session-1',
  annotation_delta: annotationDelta(INITIAL_ANNOTATIONS, ACCEPTED_POINT_ANNOTATIONS),
  mask_data: {
    session_id: 'session-1',
    annotation_revision: 1,
    state_epoch: 3,
    frame_idx: 5,
    objects: { '1': { size: [2, 2], rle: [[0, 1]], bbox: [0, 0, 1, 1] } },
    sources: { '1': 'live' },
  },
  request_frame_idx: 5,
  frame_idx: 5,
  frame_file: '00005.jpg',
  mask_pixel_counts: { 1: 1 },
  mask_shapes: { 1: [2, 2] },
  state_epoch: 3,
  ...overrides,
});

/** /video/info after the default point response was accepted. */
export const makeInfo = (overrides: Partial<VideoInfoResponse> = {}): VideoInfoResponse => ({
  session_id: 'session-1',
  state_epoch: 3,
  annotations: structuredClone(ACCEPTED_POINT_ANNOTATIONS),
  predictor_ready: true,
  num_frames: 10,
  frame_files: [],
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
    createObject:
      vi.fn<(precondition: AnnotationPrecondition) => Observable<ObjectCreationResponse>>(),
    clearAllPromptsInFrame: vi.fn<(frame: number, id: number) => Observable<AnnotationUpdate>>(),
    removeObject:
      vi.fn<(id: number, precondition: AnnotationPrecondition) => Observable<AnnotationUpdate>>(),
    updateObject:
      vi.fn<
        (object: VideoObject, precondition: AnnotationPrecondition) => Observable<AnnotationUpdate>
      >(),
    resetVideoState:
      vi.fn<(precondition: AnnotationPrecondition) => Observable<AnnotationUpdate>>(),
  };
  backendMock.getVideoInfo.mockImplementation(() => {
    const job = backendMock.getJob.mock.results.at(-1)?.value as Observable<{
      job: { result: VideoInitStateResponse };
    }>;
    return job.pipe(
      map(({ job: { result } }) => ({
        predictor_ready: true,
        session_id: 'session-1',
        state_epoch: result.state_epoch,
        num_frames: result.num_frames,
        frame_files: Array.from({ length: result.num_frames }, (_, frame) => `${frame}.jpg`),
        annotations: result.restored_session?.annotations ?? {
          version: 1,
          revision: 1,
          next_object_id: 2,
          objects: { 1: { id: 1, label: 'Object 1', color: '#ff0000' } },
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
      annotations: structuredClone(INITIAL_ANNOTATIONS),
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
        store.annotations()?.next_object_id ?? 1,
        ...objects.map((object) => object.id + 1),
      ),
      objects: Object.fromEntries(objects.map((object) => [object.id, object])),
      prompts: (store.annotations()?.prompts ?? []).filter((prompt) => ids.has(prompt.obj_id)),
      mask_overrides: (store.annotations()?.mask_overrides ?? []).filter((mask) =>
        ids.has(mask.obj_id),
      ),
      input_revisions: Object.fromEntries(
        Object.entries(store.annotations()?.input_revisions ?? {}).filter(([id]) =>
          ids.has(Number(id)),
        ),
      ),
    },
  };
}

/** Mutation response that moves the store to `objectSnapshot(store, objects)`. */
export function objectUpdate(store: VideoMaskerStateStore, objects: VideoObject[]) {
  return toUpdate(store, objectSnapshot(store, objects));
}

export function configureObjectMutationMocks(
  backend: ReturnType<typeof createBackendMocks>['backendMock'],
  store: VideoMaskerStateStore,
): void {
  backend.clearAllPromptsInFrame.mockImplementation((frame, id) => {
    const snapshot = objectSnapshot(store, store.objects());
    snapshot.annotations!.prompts = snapshot.annotations!.prompts.filter(
      (prompt) => prompt.frame_idx !== frame || prompt.obj_id !== id,
    );
    snapshot.mask_data = {
      session_id: snapshot.session_id!,
      annotation_revision: snapshot.annotations!.revision,
      state_epoch: snapshot.state_epoch,
      frame_idx: frame,
      objects: {},
      sources: {},
    };
    return of(toUpdate(store, snapshot));
  });
  backend.resetVideoState.mockImplementation(() =>
    of(
      toUpdate(store, {
        ...objectSnapshot(store, store.objects()),
        state_epoch: 4,
        annotations: {
          ...objectSnapshot(store, store.objects()).annotations!,
          prompts: [],
          mask_overrides: [],
        },
      }),
    ),
  );
  backend.createObject.mockImplementation(() => {
    const id = store.annotations()?.next_object_id ?? 1;
    const object = { id, label: `Object ${id}`, color: '#00ff00' };
    return of({ ...objectUpdate(store, [...store.objects(), object]), created_object_id: id });
  });
  backend.updateObject.mockImplementation((object) =>
    of(objectUpdate(store, [...store.objects().filter((entry) => entry.id !== object.id), object])),
  );
  backend.removeObject.mockImplementation((id) =>
    of(
      objectUpdate(
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

/** Seed accepted prompt inputs for workflow/rendering tests. */
export function setAcceptedPoints(
  store: VideoMaskerStateStore,
  frames: Map<number, Map<number, Point[]>>,
): void {
  const annotations = store.annotations() ?? {
    version: 1 as const,
    revision: 0,
    next_object_id: 1,
    objects: {},
    prompts: [],
  };
  store.annotations.set({
    ...annotations,
    prompts: [...frames].flatMap(([frame_idx, objects]) =>
      [...objects].map(([obj_id, points]) => ({
        frame_idx,
        obj_id,
        points: points.map(({ x, y }) => [x, y]),
        labels: points.map(({ label }) => label),
        box: null,
      })),
    ),
  });
}

export function cachedMasks(pipeline: FramePipelineService): Map<number, Map<number, LiveMask>> {
  const cache = (
    pipeline as unknown as { state: { maskDataCache: Map<number, Record<string, LiveMask>> } }
  ).state.maskDataCache;
  return new Map(
    [...cache].map(([frame, objects]) => [
      frame,
      new Map(Object.entries(objects).map(([id, mask]) => [Number(id), mask])),
    ]),
  );
}

export function seedMasks(
  editor: { store: VideoMaskerStateStore; framePipeline: FramePipelineService },
  masks: Map<number, Map<number, LiveMask>>,
): void {
  for (const [frame, objects] of masks) {
    const encoded: VideoMaskDataResponse['objects'] = {};
    for (const [id, mask] of objects) {
      if (!Array.isArray(mask)) {
        encoded[id] = mask;
        continue;
      }
      const flat = mask.flat();
      const rle: number[][] = [];
      for (let i = 0; i < flat.length; ) {
        if (!flat[i]) {
          i++;
          continue;
        }
        const start = i;
        while (i < flat.length && flat[i]) i++;
        rle.push([start, i - start]);
      }
      encoded[id] = {
        size: [mask.length, mask[0].length],
        rle,
        bbox: [0, 0, mask[0].length, mask.length],
      };
    }
    editor.framePipeline.acceptMaskData(
      {
        session_id: editor.store.sessionId()!,
        annotation_revision: editor.store.annotationRevision()!,
        state_epoch: editor.store.stateEpoch(),
        frame_idx: frame,
        objects: encoded,
        sources: Object.fromEntries([...objects.keys()].map((id) => [id, 'live'])),
      },
      frame,
    );
  }
}
