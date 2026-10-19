import { setAcceptedPoints } from '../testing/workflow-test-helpers';
import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { VideoInfoResponse, AnnotationSnapshot } from '../../services/backend.service';
import { createWorkflowFixture, objectSnapshot } from '../testing/workflow-test-helpers';
import { CanvasViewportService } from './canvas-viewport.service';

const apiUrl = 'http://127.0.0.1:8000';
const snapshot = (revision = 1): VideoInfoResponse => ({
  predictor_ready: true,
  session_id: 'session-1',
  state_epoch: 3,
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
        clear_old_points: true,
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
    editor.store.applyAnnotationSnapshot(snapshot(revision), apiUrl, true);
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
            state_epoch: 3,
            num_frames: 5,
            restored_session: {
              annotations: snapshot().annotations,
              interactive_state: {
                points: [{ frame_idx: 2, obj_id: 1, x: 999, y: 999, label: 1 }],
                live_masks: [],
              },
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
    expect(editor.store.annotationsBackendUrl()).toBe(apiUrl);
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
    editor.store.objects.set([]);
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
    expect(editor.store.acceptedPoints().get(2)?.get(1)).toEqual([{ x: 10, y: 20, label: 1 }]);
    expect(editor.store.selectedObjectId()).toBe(2);
    expect(editor.store.targetFrameIdx()).toBe(7);
    expect(editor.store.displayedFrameIdx()).toBe(7);
    expect(editor.store.interactionMode()).toBe('negative');
    expect(viewport.zoom()).toBe(2);
    expect(viewport.panMode()).toBe(true);
    expect(reset).not.toHaveBeenCalled();
    expect(backendMock.initVideoState).not.toHaveBeenCalled();
  });

  it('derives replacement, appended and box-only prompt views from the accepted cache', () => {
    const { editor } = fixture;
    const accepted = snapshot();
    const point = {
      frame_idx: 0,
      obj_id: 1,
      points: [[1, 1]],
      labels: [1],
      clear_old_points: true,
      box: null,
    };
    accepted.annotations!.prompts = [
      point,
      { ...point, points: [[2, 2]], clear_old_points: false },
      { ...point, frame_idx: 1 },
      { ...point, frame_idx: 1, points: [], labels: [], box: [0, 0, 2, 2] },
    ];
    editor.store.applyAnnotationSnapshot(accepted, apiUrl, true);
    expect(editor.store.points().get(0)?.get(1)?.length).toBe(2);
    expect(editor.store.points().has(1)).toBe(false);
    accepted.annotations!.prompts[0].points[0][0] = 999;
    expect(editor.store.acceptedPoints().get(0)?.get(1)?.[0].x).toBe(1);
  });

  it('rejects a snapshot with an older accepted revision', async () => {
    const { editor, backendMock } = fixture;
    connect(5);
    backendMock.getVideoInfo.mockReturnValue(of(snapshot(4)));
    expect(await editor.sessions.refreshAnnotations()).toBe(false);
    expect(editor.store.annotationRevision()).toBe(5);
  });

  it('keeps a newer response when an earlier snapshot request finishes last', async () => {
    const { editor, backendMock } = fixture;
    connect();
    const first = new Subject<VideoInfoResponse>();
    const second = new Subject<VideoInfoResponse>();
    backendMock.getVideoInfo.mockReturnValueOnce(first).mockReturnValueOnce(second);
    const older = editor.sessions.refreshAnnotations();
    const newer = editor.sessions.refreshAnnotations();
    second.next(snapshot(3));
    expect(await newer).toBe(true);
    first.next(snapshot(2));
    expect(await older).toBe(false);
    expect(editor.store.annotationRevision()).toBe(3);
  });

  it('does not adopt a replacement backend session during recovery', async () => {
    const { editor, backendMock } = fixture;
    connect();
    backendMock.getVideoInfo.mockReturnValue(of({ ...snapshot(), session_id: 'other-session' }));
    expect(await editor.sessions.refreshAnnotations()).toBe(false);
    expect(editor.store.sessionId()).toBe('session-1');
    expect(editor.store.isInitialized()).toBe(false);
  });

  it('discards a response after the configured backend changes', async () => {
    const { editor, backendMock } = fixture;
    connect();
    const pending = new Subject<VideoInfoResponse>();
    backendMock.getVideoInfo.mockReturnValue(pending);
    const refresh = editor.sessions.refreshAnnotations();
    backendMock.getApiUrl.mockReturnValue('http://other-backend');
    pending.next(snapshot(2));
    expect(await refresh).toBe(false);
    expect(editor.store.annotationsBackendUrl()).toBe(apiUrl);
    expect(editor.store.annotationRevision()).toBe(1);
  });

  it('does not overwrite optimistic point input entered during a snapshot request', async () => {
    const { editor, backendMock } = fixture;
    connect();
    const pending = new Subject<VideoInfoResponse>();
    const pointResponse = new Subject<never>();
    backendMock.getVideoInfo.mockReturnValue(pending);
    backendMock.addNewPointsOrBox.mockReturnValue(pointResponse);
    const refresh = editor.sessions.refreshAnnotations();
    const edit = editor.prompts.addPoint(30, 40, 1, 2);
    pending.next(snapshot());
    expect(await refresh).toBe(false);
    expect(editor.store.points().get(2)?.get(1)?.length).toBe(1);
    expect(editor.store.previewPoints().get(2)?.get(1)?.length).toBe(2);
    pointResponse.error(new Error('rejected'));
    await edit;
  });

  it('waits for object updates and rejects an in-flight snapshot overtaken by a rename', async () => {
    const { editor, backendMock, objectOperations } = fixture;
    connect();
    const pending = new Subject<VideoInfoResponse>();
    const objectResponse = new Subject<AnnotationSnapshot>();
    backendMock.getVideoInfo.mockReturnValue(pending);
    const refresh = editor.sessions.refreshAnnotations();
    backendMock.updateObject.mockReturnValue(objectResponse);
    objectOperations.renameObject(1, 'Draft label');
    expect(editor.store.pendingObjectUpdates()).toBe(1);
    expect(await editor.sessions.refreshAnnotations()).toBe(false);
    objectResponse.next(
      objectSnapshot(
        editor.store,
        editor.store
          .objects()
          .map((object) => (object.id === 1 ? { ...object, label: 'Draft label' } : object)),
      ),
    );
    objectResponse.complete();
    expect(editor.store.pendingObjectUpdates()).toBe(0);
    pending.next(snapshot());
    expect(await refresh).toBe(false);
    expect(editor.store.objects()[0].label).toBe('Draft label');
  });

  it('does not apply a snapshot from an earlier predictor epoch', async () => {
    const { editor, backendMock } = fixture;
    connect();
    editor.store.updateStateEpoch(4, 'rebuild');
    backendMock.getVideoInfo.mockReturnValue(of(snapshot(2)));
    expect(await editor.sessions.refreshAnnotations()).toBe(false);
    expect(editor.store.stateEpoch()).toBe(4);
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

  it('retains live masks when recovery only changes an object label', async () => {
    const { editor, backendMock } = fixture;
    connect();
    const masks = new Map([[2, new Map([[1, [[true]]]])]]);
    editor.store.masks.set(masks);
    const renamed = snapshot(2);
    renamed.annotations!.objects['1'].label = 'Renamed hand';
    backendMock.getVideoInfo.mockReturnValue(of(renamed));
    expect(await editor.sessions.refreshAnnotations()).toBe(true);
    expect(editor.store.masks()).toBe(masks);
  });

  it('invalidates live masks when accepted box inputs change without changing points', async () => {
    const { editor, backendMock } = fixture;
    connect();
    editor.store.masks.set(new Map([[2, new Map([[1, [[true]]]])]]));
    const boxed = snapshot(2);
    boxed.annotations!.prompts[0].box = [0, 0, 30, 30];
    backendMock.getVideoInfo.mockReturnValue(of(boxed));
    expect(await editor.sessions.refreshAnnotations()).toBe(true);
    expect(editor.store.masks().size).toBe(0);
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
