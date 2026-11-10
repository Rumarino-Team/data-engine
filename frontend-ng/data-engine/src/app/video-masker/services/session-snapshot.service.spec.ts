import { cachedMasks, seedMasks } from '../testing/workflow-test-helpers';
import { setAcceptedPoints } from '../testing/workflow-test-helpers';
import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { VideoInfoResponse } from '../../services/backend.service';
import { annotationDelta, createWorkflowFixture } from '../testing/workflow-test-helpers';
import { CanvasViewportService } from './canvas-viewport.service';

const snapshot = (revision = 1): VideoInfoResponse => ({
  predictor_ready: true,
  session_id: 'session-1',
  num_frames: 10,
  frame_files: Array.from({ length: 10 }, (_, frame) => `${frame}.jpg`),
  annotations: {
    version: 1,
    revision,
    next_object_id: 3,
    objects: {
      '1': { id: 1, label: 'Hand', color: '#ff0000' },
      '2': { id: 2, label: 'Cup', color: '#00ff00' },
    },
    prompts: [
      {
        frame_idx: 2,
        obj_id: 1,
        points: [[10, 20]],
        labels: [1],
        box: null,
      },
    ],
  },
});

describe('session annotation snapshots', () => {
  let fixture: ReturnType<typeof createWorkflowFixture>;

  beforeEach(() => {
    fixture = createWorkflowFixture();
  });

  function connect(revision = 1) {
    const { editor, backendMock } = fixture;
    editor.store.applyAnnotationSnapshot(snapshot(revision), true);
    editor.store.isInitialized.set(true);
    backendMock.getVideoInfo.mockReturnValue(of(snapshot(revision)));
  }

  function mockInitialization() {
    const { editor, backendMock } = fixture;
    editor.store.videoDir.set('C:/frames');
    backendMock.initVideoState.mockReturnValue(of({ job_id: 'init' }));
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          status: 'completed',
          result: {
            num_frames: 5,
            restored_session: {
              annotations: snapshot().annotations,
              interactive_state: { version: 1, current_frame_idx: 2 },
            },
          },
        },
      }),
    );
  }

  it('initializes from video/info instead of annotations or points in the job payload', async () => {
    const { editor, backendMock } = fixture;
    mockInitialization();
    const accepted = snapshot(8);
    accepted.annotations!.objects['1'].label = 'Accepted hand';
    backendMock.getVideoInfo.mockReturnValue(of(accepted));

    expect(await editor.sessions.initVideo()).toBe(true);

    expect(backendMock.getVideoInfo).toHaveBeenCalledTimes(1);
    expect(editor.store.sessionId()).toBe('session-1');
    expect(editor.store.annotationRevision()).toBe(8);
    expect(editor.store.numFrames()).toBe(10);
    expect(editor.store.objects()[0].label).toBe('Accepted hand');
    expect(editor.store.points().get(2)?.get(1)).toEqual([{ x: 10, y: 20, label: 1 }]);
    expect(backendMock.updateObject).not.toHaveBeenCalled();
  });

  it('does not claim initialization succeeded when video/info is unavailable', async () => {
    const { editor, backendMock } = fixture;
    mockInitialization();
    const previous = editor.store.annotations();
    backendMock.getVideoInfo.mockReturnValue(throwError(() => new Error('offline')));

    expect(await editor.sessions.initVideo()).toBe(false);
    expect(editor.store.isInitialized()).toBe(false);
    expect(editor.store.isLoading()).toBe(false);
    expect(editor.store.sessionId()).toBe('session-1');
    expect(editor.store.annotations()).toBe(previous);
  });

  it('reconstructs discarded frontend annotations without reopening or resetting the view', async () => {
    const { editor, backendMock } = fixture;
    connect();
    const viewport = TestBed.inject(CanvasViewportService);
    const reset = vi.spyOn(viewport, 'reset');
    setAcceptedPoints(editor.store, new Map());
    editor.store.annotations.set(null);
    editor.store.selectedObjectId.set(2);
    editor.store.targetFrameIdx.set(7);
    editor.store.displayedFrameIdx.set(7);
    editor.store.interactionMode.set('negative');
    viewport.zoom.set(2);
    viewport.panMode.set(true);

    expect(await editor.sessions.refreshAnnotations()).toBe(true);

    expect(editor.store.objects().map(({ id }) => id)).toEqual([1, 2]);
    expect(editor.store.points().get(2)?.get(1)).toEqual([{ x: 10, y: 20, label: 1 }]);
    expect(editor.store.selectedObjectId()).toBe(2);
    expect(editor.store.targetFrameIdx()).toBe(7);
    expect(editor.store.displayedFrameIdx()).toBe(7);
    expect(editor.store.interactionMode()).toBe('negative');
    expect(viewport.zoom()).toBe(2);
    expect(viewport.panMode()).toBe(true);
    expect(reset).not.toHaveBeenCalled();
    expect(backendMock.initVideoState).not.toHaveBeenCalled();
  });

  it('derives point and box-only prompt views from the accepted cache', () => {
    const { editor } = fixture;
    const accepted = snapshot();
    const point = {
      frame_idx: 0,
      obj_id: 1,
      points: [
        [1, 1],
        [2, 2],
      ],
      labels: [1, 0],
      box: null,
    };
    accepted.annotations!.prompts = [
      point,
      { ...point, frame_idx: 1, points: [], labels: [], box: [0, 0, 2, 2] },
    ];
    editor.store.applyAnnotationSnapshot(accepted, true);
    expect(editor.store.points().get(0)?.get(1)).toEqual([
      { x: 1, y: 1, label: 1 },
      { x: 2, y: 2, label: 0 },
    ]);
    expect(editor.store.points().has(1)).toBe(false);
    accepted.annotations!.prompts[0].points[0][0] = 999;
    expect(editor.store.points().get(0)?.get(1)?.[0].x).toBe(1);
  });

  it('patches accepted annotations from a mutation delta', () => {
    const { editor } = fixture;
    const accepted = snapshot();
    accepted.annotations!.mask_overrides = [
      { frame_idx: 3, obj_id: 2, artifact: `${'a'.repeat(32)}.json` },
    ];
    editor.store.applyAnnotationSnapshot(accepted, true);
    const applied = editor.store.applyAnnotationUpdate({
      session_id: 'session-1',
      annotation_delta: {
        ...annotationDelta(accepted.annotations!, accepted.annotations!),
        revision: 2,
        objects: { '1': { id: 1, label: 'Left hand', color: '#ff0000' } },
        removed_objects: [],
        prompts: [{ frame_idx: 2, obj_id: 1, points: [[5, 6]], labels: [0], box: null }],
        removed_mask_overrides: [[3, 2]],
        input_revisions: { '1': 2 },
      },
    });
    expect(applied).toBe(true);
    const annotations = editor.store.annotations()!;
    expect(annotations.revision).toBe(2);
    expect(annotations.objects['1'].label).toBe('Left hand');
    expect(annotations.objects['2'].label).toBe('Cup');
    expect(annotations.prompts).toEqual([
      { frame_idx: 2, obj_id: 1, points: [[5, 6]], labels: [0], box: null },
    ]);
    expect(annotations.mask_overrides).toEqual([]);
    expect(annotations.input_revisions).toEqual({ '1': 2 });
    expect(editor.store.objects().map((object) => object.label)).toEqual(['Left hand', 'Cup']);
  });

  it('rejects a delta built on another revision or session without changing state', () => {
    const { editor } = fixture;
    const accepted = snapshot(4);
    editor.store.applyAnnotationSnapshot(accepted, true);
    const before = editor.store.annotations();
    const delta = {
      ...annotationDelta(accepted.annotations!, accepted.annotations!),
      revision: 5,
      removed_objects: [2],
    };
    for (const update of [
      { session_id: 'session-1', annotation_delta: { ...delta, base_revision: 3 } },
      { session_id: 'other', annotation_delta: delta },
    ]) {
      expect(editor.store.applyAnnotationUpdate(update)).toBe(false);
      expect(editor.store.annotations()).toBe(before);
    }
    expect(
      editor.store.applyAnnotationUpdate({ session_id: 'session-1', annotation_delta: delta }),
    ).toBe(true);
    expect(Object.keys(editor.store.annotations()!.objects)).toEqual(['1']);
    expect(editor.store.annotations()!.prompts.map((prompt) => prompt.obj_id)).toEqual([1]);
  });

  it.each([
    {
      reason: 'an older accepted revision',
      prepare: () => fixture.editor.store.applyAnnotationSnapshot(snapshot(5), true),
      info: () => snapshot(4),
      expected: { revision: 5, initialized: true },
    },
    {
      reason: 'a replacement backend session',
      prepare: () => undefined,
      info: () => ({ ...snapshot(), session_id: 'other-session' }),
      expected: { revision: 1, initialized: false },
    },
  ])('recovery does not adopt $reason', async ({ prepare, info, expected }) => {
    const { editor, backendMock } = fixture;
    connect();
    prepare();
    editor.store.isInitialized.set(true);
    backendMock.getVideoInfo.mockReturnValue(of(info()));
    expect(await editor.sessions.refreshAnnotations()).toBe(false);
    expect(editor.store.sessionId()).toBe('session-1');
    expect(editor.store.annotationRevision()).toBe(expected.revision);
    expect(editor.store.isInitialized()).toBe(expected.initialized);
  });

  it.each([
    [
      'an object label',
      (info: VideoInfoResponse) => (info.annotations!.objects['1'].label = 'Renamed hand'),
    ],
    [
      'a box input without any point change',
      (info: VideoInfoResponse) => (info.annotations!.prompts[0].box = [0, 0, 30, 30]),
    ],
  ])('recovery refetches backend masks when %s changes', async (_change, mutate) => {
    const { editor, backendMock } = fixture;
    connect();
    seedMasks(editor, new Map([[2, new Map([[1, [[true]]]])]]));
    const changed = snapshot(2);
    mutate(changed);
    backendMock.getVideoInfo.mockReturnValue(of(changed));
    expect(await editor.sessions.refreshAnnotations()).toBe(true);
    expect(cachedMasks(editor.framePipeline).size).toBe(0);
  });

  it('refuses a second refresh while one is in flight', async () => {
    const { editor, backendMock } = fixture;
    connect();
    const pending = new Subject<VideoInfoResponse>();
    backendMock.getVideoInfo.mockReturnValue(pending);
    const first = editor.sessions.refreshAnnotations();
    expect(editor.store.isRecoveringAnnotations()).toBe(true);
    expect(await editor.sessions.refreshAnnotations()).toBe(false);
    expect(backendMock.getVideoInfo).toHaveBeenCalledTimes(1);
    pending.next(snapshot(2));
    expect(await first).toBe(true);
    expect(editor.store.annotationRevision()).toBe(2);
    expect(editor.store.isRecoveringAnnotations()).toBe(false);
  });

  it('preserves the last accepted cache when recovery fails', async () => {
    const { editor, backendMock } = fixture;
    connect();
    const accepted = editor.store.annotations();
    backendMock.getVideoInfo.mockReturnValue(throwError(() => new Error('offline')));
    expect(await editor.sessions.refreshAnnotations()).toBe(false);
    expect(editor.store.annotations()).toBe(accepted);
    expect(editor.store.toasts()[0].title).toBe('Annotation refresh failed');
  });

  it('refreshes once on reconnect and defers that recovery while editing', async () => {
    const { editor, backendMock } = fixture;
    connect();
    TestBed.tick();
    editor.store.apiHealthStatus.set('offline');
    TestBed.tick();
    editor.store.isPointRequestInFlight.set(true);
    editor.store.apiHealthStatus.set('online');
    TestBed.tick();
    expect(backendMock.getVideoInfo).not.toHaveBeenCalled();
    editor.store.isPointRequestInFlight.set(false);
    TestBed.tick();
    await Promise.resolve();
    expect(backendMock.getVideoInfo).toHaveBeenCalledTimes(1);
    editor.store.apiHealthStatus.set('checking');
    TestBed.tick();
    editor.store.apiHealthStatus.set('online');
    TestBed.tick();
    expect(backendMock.getVideoInfo).toHaveBeenCalledTimes(1);
  });

  it('cancels recovery when the editor is destroyed', async () => {
    const { editor, backendMock } = fixture;
    connect();
    const pending = new Subject<VideoInfoResponse>();
    backendMock.getVideoInfo.mockReturnValue(pending);
    const refresh = editor.sessions.refreshAnnotations();
    expect(pending.observed).toBe(true);
    TestBed.resetTestingModule();
    expect(await refresh).toBe(false);
    expect(pending.observed).toBe(false);
  });
});
