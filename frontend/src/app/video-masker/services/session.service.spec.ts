import { TestBed } from '@angular/core/testing';
import { VideoJobsService } from './video-jobs.service';
import { cachedMasks } from '../testing/workflow-test-helpers';
import { describe, beforeEach, expect, it, vi } from 'vitest';
import { of, throwError } from 'rxjs';
import { createWorkflowFixture, mockInitJob, mockJob } from '../testing/workflow-test-helpers';

describe('session workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];
  let objectOperations: ReturnType<typeof createWorkflowFixture>['objectOperations'];

  beforeEach(() => {
    ({ editor, backendMock, objectOperations } = createWorkflowFixture());
  });

  it('starts a video init job, polls completion, and applies the result', async () => {
    mockInitJob(backendMock);
    editor.store.videoDir.set('C:/frames');

    await editor.sessions.initVideo();

    expect(backendMock.initVideoState).toHaveBeenCalledWith('C:/frames', { frame_loading: 'lazy' });
    expect(backendMock.getJob).toHaveBeenCalledWith('job-1');
    expect(editor.store.isInitialized()).toBe(true);
    expect(editor.store.numFrames()).toBe(12);
    const [created] = editor.store.objects();
    expect(created).toEqual(expect.objectContaining({ id: 1, label: 'Object 1' }));
    expect(editor.store.selectedObjectId()).toBe(1);
    expect(backendMock.createObject).not.toHaveBeenCalled();
    expect(backendMock.updateObject).not.toHaveBeenCalled();
  });

  it('restores interactive state from saved-session init result', async () => {
    mockInitJob(backendMock, {
      num_frames: 10,
      source_type: 'saved_session',
      restored_session: {
        annotations: {
          version: 1,
          revision: 1,
          next_object_id: 2,
          objects: { '1': { id: 1, label: 'buoy', color: '#ff6600' } },
          prompts: [{ frame_idx: 4, obj_id: 1, points: [[10, 20]], labels: [1], box: null }],
        },
        has_mask_manifest: true,
        interactive_state: {
          version: 1,
          selected_object_id: 1,
          interaction_mode: 'negative',
          current_frame_idx: 4,
        },
      },
    });
    editor.store.loadSourceMode.set('saved_session_dir');
    editor.store.videoDir.set('C:/backend/saved/review-run');

    await editor.sessions.initVideo();

    expect(editor.store.interactionMode()).toBe('negative');
    expect(editor.store.targetFrameIdx()).toBe(4);
    expect(editor.store.points().get(4)?.get(1)?.length).toBe(1);
    expect(cachedMasks(editor.framePipeline).size).toBe(0);
    expect(editor.store.objects()).toEqual([{ id: 1, label: 'buoy', color: '#ff6600' }]);
    expect(editor.store.selectedObjectId()).toBe(1);
    expect(backendMock.updateObject).not.toHaveBeenCalled();
  });

  describe('restoring a persisted tracking result', () => {
    const initWithTrackingResult = (trackingResult: {
      result_id: string;
      summary?: Record<string, unknown>;
    }) => {
      mockInitJob(backendMock, {
        num_frames: 10,
        source_type: 'saved_session',
        restored_session: {
          annotations: { version: 1, revision: 0, next_object_id: 1, objects: {}, prompts: [] },
          has_mask_manifest: true,
          tracking_result: trackingResult,
        },
      });
      editor.store.loadSourceMode.set('saved_session_dir');
      editor.store.videoDir.set('C:/backend/saved/review-run');
    };

    it('loads the result named by the saved session', async () => {
      initWithTrackingResult({ result_id: 'track-restored', summary: { num_points: 1 } });
      backendMock.getTrackingResult.mockReturnValue(
        of({
          result: {
            version: 1,
            result_id: 'track-restored',
            session_id: 'session-1',
            annotation_revision: 0,
            model_name: 'cotracker3_online',
            num_points: 1,
            num_frames: 2,
            add_support_grid_used: false,
            tracking_mode: 'streaming',
            streaming_frame_threshold: 256,
            tracks: [
              [
                [10, 20],
                [11, 21],
              ],
            ],
            visibility: [[true, true]],
            points: [
              { point_id: 'p0_0', obj_id: 1, source_frame_idx: 0, source_x: 10, source_y: 20 },
            ],
          },
        }),
      );

      await editor.sessions.initVideo();

      expect(backendMock.getTrackingResult).toHaveBeenCalledWith('track-restored');
      expect(editor.store.trackedPoints()[0].tracks).toEqual([
        [10, 20],
        [11, 21],
      ]);
    });
  });

  it('sends the chosen frame loading mode with the video to load', async () => {
    mockInitJob(backendMock);
    editor.store.videoDir.set('C:/frames');
    editor.store.frameLoadingMode.set('preload');

    await editor.sessions.initVideo();

    expect(backendMock.initVideoState).toHaveBeenCalledWith('C:/frames', {
      frame_loading: 'preload',
    });
  });

  it('shows the mode a saved session reopened with, not the one that was selected', async () => {
    mockInitJob(backendMock, { frame_loading: 'preload', source_type: 'saved_session' });
    editor.store.videoDir.set('C:/saved');
    editor.store.frameLoadingMode.set('eager');

    await editor.sessions.initVideo();

    expect(editor.store.frameLoadingMode()).toBe('preload');
  });

  it('creates an error toast when a job fails', async () => {
    mockJob(
      backendMock.initVideoState,
      backendMock.getJob,
      'video_init',
      {
        status: 'failed',
        stage: 'initializing_state',
        stage_label: 'Initializing video state',
        progress: 0.5,
        message: 'failed',
        error: { code: 'validation_error', message: 'Path not found', detail: null },
      },
      'job-2',
    );
    editor.store.videoDir.set('C:/missing');

    await editor.sessions.initVideo();

    expect(editor.store.isInitialized()).toBe(false);
    expect(editor.store.toasts()[0].title).toBe('Loading video');
    expect(editor.store.toasts()[0].message).toBe('Path not found');
  });

  it('saves the current session with the typed name', async () => {
    backendMock.saveVideoSession.mockReturnValue(
      of({
        message: 'Session saved successfully',
        name: 'review-run',
        checkpoint_id: 'checkpoint-1',
        saved_path: 'C:/project/backend/saved/review-run',
        saved_state: {
          checkpoint_id: 'checkpoint-1',
          display_name: 'review-run',
          saved_at: 'now',
          dirty: false,
        },
      }),
    );
    editor.store.isInitialized.set(true);
    editor.store.saveName.set('review-run');
    objectOperations.renameObject(1, 'Left hand');

    expect(backendMock.updateObject).toHaveBeenCalledWith(
      {
        id: 1,
        label: 'Left hand',
        color: '#ff0000',
      },
      expect.objectContaining({ session_id: 'session-1' }),
    );

    await expect(editor.sessions.save()).resolves.toBe(true);

    const [request, precondition, interactiveState] = backendMock.saveVideoSession.mock.calls[0];
    expect(request).toEqual({ name: 'review-run', saveAs: false });
    expect(precondition).toEqual(expect.objectContaining({ session_id: 'session-1' }));
    expect(interactiveState).toEqual(
      expect.objectContaining({
        version: 1,
        selected_object_id: 1,
        interaction_mode: 'positive',
        current_frame_idx: 0,
      }),
    );
    expect(interactiveState).not.toHaveProperty('objects');
    expect(editor.store.toasts()[0].title).toBe('Session saved');
  });
  it('retains the current session after rejected replacement and requests reconciliation', async () => {
    editor.store.isInitialized.set(true);
    editor.store.videoDir.set('invalid-replacement');
    editor.store.displayedFrameIdx.set(5);
    const previous = editor.store.annotations();
    vi.spyOn(TestBed.inject(VideoJobsService), 'run').mockResolvedValue({
      result: null,
      completedJobId: null,
    });
    const clear = vi.spyOn(editor.framePipeline, 'clearFrameCaches');
    expect(await editor.sessions.initVideo()).toBe(false);
    expect(editor.store.isInitialized()).toBe(true);
    expect(editor.store.annotations()).toBe(previous);
    expect(editor.store.displayedFrameIdx()).toBe(5);
    expect(editor.store.annotationRecoveryRequested()).toBe(true);
    expect(clear).not.toHaveBeenCalled();
  });
});

describe('a session the backend still holds', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];
  let leave: ReturnType<typeof createWorkflowFixture>['leave'];
  let confirm: ReturnType<typeof createWorkflowFixture>['confirm'];

  const heldByBackend = (overrides: Record<string, unknown> = {}) => ({
    session_id: 'held-1',
    num_frames: 924,
    frame_files: [],
    predictor_ready: true,
    saved_state: { checkpoint_id: null, display_name: null, saved_at: null, dirty: true },
    annotations: {
      version: 1,
      revision: 4,
      next_object_id: 3,
      objects: {
        1: { id: 1, label: 'Boy', color: '#112233' },
        2: { id: 2, label: 'Goal', color: '#445566' },
      },
      prompts: [],
    },
    ...overrides,
  });

  const offer = (dirty: boolean) =>
    editor.store.openBackendSession.set({
      sessionId: 'held-1',
      revision: 4,
      numFrames: 924,
      dirty,
      resumable: true,
    });

  beforeEach(() => {
    ({ editor, backendMock, leave, confirm } = createWorkflowFixture());
    // The fixture starts as an open editor; these tests are about a tab that shows nothing.
    editor.store.resetSession();
  });

  describe('noticing it', () => {
    it('reports what the backend holds without loading or discarding anything', async () => {
      backendMock.getVideoInfo.mockReturnValue(of(heldByBackend()));

      await editor.sessions.checkForOpenSession();

      expect(editor.store.openBackendSession()).toEqual({
        sessionId: 'held-1',
        revision: 4,
        numFrames: 924,
        dirty: true,
        resumable: true,
      });
      expect(editor.store.isInitialized()).toBe(false);
      expect(backendMock.initVideoState).not.toHaveBeenCalled();
    });

    it('marks a session whose predictor is gone as not resumable', async () => {
      backendMock.getVideoInfo.mockReturnValue(of(heldByBackend({ predictor_ready: false })));
      await editor.sessions.checkForOpenSession();
      expect(editor.store.openBackendSession()?.resumable).toBe(false);
      await expect(editor.sessions.resumeOpenSession()).resolves.toBe(false);
      expect(editor.store.isInitialized()).toBe(false);
    });

    it('looks by itself once the backend is online and nothing is shown', () => {
      backendMock.getVideoInfo.mockReturnValue(of(heldByBackend()));
      editor.store.apiHealthStatus.set('online');
      TestBed.tick();
      expect(backendMock.getVideoInfo).toHaveBeenCalledTimes(1);
    });

    it('looks again after a load the backend refused, so the user can see why', async () => {
      backendMock.initVideoState.mockReturnValue(
        throwError(() => ({
          status: 409,
          error: { detail: 'The open session has unsaved changes. Resume it or close it.' },
        })),
      );
      backendMock.getVideoInfo.mockReturnValue(of(heldByBackend()));
      editor.store.videoDir.set('C:/other');

      await editor.sessions.initVideo();
      await Promise.resolve();

      expect(editor.store.toasts()[0].message).toContain('unsaved changes');
      expect(editor.store.openBackendSession()?.sessionId).toBe('held-1');
    });
  });

  describe('resuming it', () => {
    beforeEach(async () => {
      backendMock.getVideoInfo.mockReturnValue(of(heldByBackend()));
      await editor.sessions.checkForOpenSession();
    });

    it('shows the session as it is on the backend, without loading the video again', async () => {
      await expect(editor.sessions.resumeOpenSession()).resolves.toBe(true);

      expect(editor.store.isInitialized()).toBe(true);
      expect(editor.store.sessionId()).toBe('held-1');
      expect(editor.store.numFrames()).toBe(924);
      expect(editor.store.annotationRevision()).toBe(4);
      expect(editor.store.objects().map((object) => object.label)).toEqual(['Boy', 'Goal']);
      expect(editor.store.selectedObjectId()).toBe(1);
      expect(editor.store.propagationEndFrameIdx()).toBe(923);
      expect(editor.store.openBackendSession()).toBeNull();
      // Nothing reported: the selector is left as it was.
      expect(editor.store.frameLoadingMode()).toBe('lazy');
      expect(editor.store.isLoading()).toBe(false);
      expect(backendMock.initVideoState).not.toHaveBeenCalled();
    });

    describe('with tracked points', () => {
      const withTracking = (validity: unknown = { result_id: 'track-1', point_ids: ['p0_0'] }) =>
        backendMock.getVideoInfo.mockReturnValue(
          of(heldByBackend({ tracking_validity: validity })),
        );
      const storedResult = () =>
        of({
          result: {
            version: 1,
            result_id: 'track-1',
            session_id: 'held-1',
            annotation_revision: 4,
            model_name: 'cotracker3_online',
            num_points: 1,
            num_frames: 924,
            add_support_grid_used: false,
            tracking_mode: 'streaming',
            streaming_frame_threshold: 256,
            tracks: [
              [
                [10, 20],
                [11, 21],
              ],
            ],
            visibility: [[true, true]],
            points: [
              { point_id: 'p0_0', obj_id: 1, source_frame_idx: 0, source_x: 10, source_y: 20 },
            ],
          },
        });

      it('brings the overlays back from the result the backend reports, without tracking again', async () => {
        withTracking();
        backendMock.getTrackingResult.mockReturnValue(storedResult());

        await expect(editor.sessions.resumeOpenSession()).resolves.toBe(true);

        expect(backendMock.getTrackingResult).toHaveBeenCalledWith('track-1');
        expect(editor.store.trackingResultId).toBe('track-1');
        expect(editor.store.trackedPoints()).toHaveLength(1);
        expect(editor.store.trackedPoints()[0].tracks).toEqual([
          [10, 20],
          [11, 21],
        ]);
        expect(backendMock.trackPromptPoints).not.toHaveBeenCalled();
      });

      it('still resumes when the result cannot be read, and says so', async () => {
        withTracking();
        backendMock.getTrackingResult.mockReturnValue(
          throwError(() => ({ status: 409, error: { detail: 'Tracking result does not match.' } })),
        );

        await expect(editor.sessions.resumeOpenSession()).resolves.toBe(true);

        expect(editor.store.isInitialized()).toBe(true);
        expect(editor.store.trackedPoints()).toEqual([]);
        expect(editor.store.toasts()[0].title).toBe('Tracking result unavailable');
      });
    });

    it('brings back which stretches of the timeline come from an earlier propagation', async () => {
      backendMock.getVideoInfo.mockReturnValue(
        of(heldByBackend({ outdated_masks: { '1': [[0, 500]] } })),
      );
      await editor.sessions.resumeOpenSession();
      expect(editor.store.outdatedMasks()).toEqual({ '1': [[0, 500]] });
    });

    it('refuses a session that was replaced meanwhile and shows what is there now', async () => {
      backendMock.getVideoInfo.mockReturnValue(of(heldByBackend({ session_id: 'held-2' })));

      await expect(editor.sessions.resumeOpenSession()).resolves.toBe(false);
      await Promise.resolve();

      expect(editor.store.isInitialized()).toBe(false);
      expect(editor.store.toasts()[0].title).toBe('Resuming session');
      expect(editor.store.openBackendSession()?.sessionId).toBe('held-2');
    });
  });

  describe('closing it from a tab that does not show it', () => {
    it('closes a session without unsaved changes straight away, naming session and revision', async () => {
      offer(false);
      await expect(leave.closeOpenBackendSession()).resolves.toBe(true);
      expect(confirm.request()).toBeNull();
      expect(backendMock.closeVideoSession).toHaveBeenCalledWith({
        session_id: 'held-1',
        expected_revision: 4,
      });
      expect(editor.store.openBackendSession()).toBeNull();
    });

    it('asks before throwing unsaved changes away, and closes only on that answer', async () => {
      offer(true);
      const pending = leave.closeOpenBackendSession();
      expect(confirm.request()?.title).toBe('Unsaved changes');
      expect(confirm.request()?.actions.map((action) => action.value)).toEqual(['discard']);
      confirm.answer('discard');
      await expect(pending).resolves.toBe(true);
      expect(backendMock.closeVideoSession).toHaveBeenCalledTimes(1);
      expect(editor.store.openBackendSession()).toBeNull();
    });

    it('shows what is there now when the session changed since it was offered', async () => {
      offer(false);
      backendMock.closeVideoSession.mockReturnValue(
        throwError(() => ({ status: 409, error: { detail: 'Annotations changed.' } })),
      );
      backendMock.getVideoInfo.mockReturnValue(of(heldByBackend()));

      await expect(leave.closeOpenBackendSession()).resolves.toBe(false);
      await Promise.resolve();

      expect(editor.store.toasts()[0]).toEqual(expect.objectContaining({ title: 'Close failed' }));
      expect(editor.store.openBackendSession()?.revision).toBe(4);
      expect(editor.store.annotationRecoveryRequested()).toBe(false);
    });

    it('keeps the offer and says so when the backend cannot be reached', async () => {
      offer(false);
      backendMock.closeVideoSession.mockReturnValue(throwError(() => ({ status: 0 })));
      await expect(leave.closeOpenBackendSession()).resolves.toBe(false);
      expect(editor.store.openBackendSession()).not.toBeNull();
      expect(editor.store.toasts()[0]).toEqual(
        expect.objectContaining({ severity: 'warning', title: 'Backend unreachable' }),
      );
    });
  });
});
