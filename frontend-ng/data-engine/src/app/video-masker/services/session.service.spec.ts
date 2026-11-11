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

    expect(backendMock.initVideoState).toHaveBeenCalledWith('C:/frames');
    expect(backendMock.getJob).toHaveBeenCalledWith('job-1');
    expect(editor.store.isInitialized()).toBe(true);
    expect(editor.store.numFrames()).toBe(12);
    const [created] = editor.store.objects();
    expect(created).toEqual(expect.objectContaining({ id: 1, label: 'Object 1' }));
    expect(editor.store.selectedObjectId()).toBe(1);
    expect(backendMock.createObject).not.toHaveBeenCalled();
    expect(backendMock.updateObject).not.toHaveBeenCalled();
  });

  it('preserves an intentionally empty canonical session and its reserved object IDs', async () => {
    mockInitJob(backendMock, {
      num_frames: 1,
      restored_session: {
        annotations: { version: 1, revision: 8, next_object_id: 7, objects: {}, prompts: [] },
        interactive_state: { version: 1 },
        has_mask_manifest: false,
      },
    });
    editor.store.videoDir.set('C:/saved/session');

    await editor.sessions.initVideo();

    expect(editor.store.objects()).toEqual([]);
    expect(editor.store.selectedObjectId()).toBeNull();
    expect(editor.store.annotations()?.next_object_id).toBe(7);
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

    it('keeps init successful when the result is unavailable', async () => {
      initWithTrackingResult({ result_id: 'missing-track' });
      backendMock.getTrackingResult.mockReturnValue(
        throwError(() => ({ error: { detail: 'missing' } })),
      );

      await editor.sessions.initVideo();

      expect(editor.store.isInitialized()).toBe(true);
      expect(editor.store.toasts()[0].title).toBe('Tracking result unavailable');
    });
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
