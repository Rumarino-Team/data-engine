import { describe, beforeEach, expect, it, vi } from 'vitest';
import { of, throwError } from 'rxjs';
import { createWorkflowFixture } from '../testing/workflow-test-helpers';

describe('session workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];
  let objectOperations: ReturnType<typeof createWorkflowFixture>['objectOperations'];

  beforeEach(() => {
    ({ editor, backendMock, objectOperations } = createWorkflowFixture());
  });

  it('starts a video init job, polls completion, and applies the result', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-1',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-1',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 12,
          total: 12,
          window_index: null,
          window_count: null,
          frame_idx: null,
          stage_history: [],
          message: 'done',
          error: null,
          started_at: 'now',
          updated_at: 'now',
          completed_at: 'now',
          result: {
            message: 'Video state initialized successfully',
            num_frames: 12,
            resolved_video_frames_dir: 'C:/frames',
            source_video_path: null,
            online_mode: true,
            batch_size: 32,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 7,
          },
        },
      }),
    );
    editor.store.videoDir.set('C:/frames');

    await editor.sessions.initVideo();

    expect(backendMock.initVideoState).toHaveBeenCalledWith('C:/frames');
    expect(backendMock.getJob).toHaveBeenCalledWith('job-1');
    expect(editor.store.isInitialized()).toBe(true);
    expect(editor.store.numFrames()).toBe(12);
    expect(editor.store.stateEpoch()).toBe(7);
  });

  it('uses the same init endpoint path flow for saved-session mode', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-saved',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-saved',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 8,
          total: 8,
          window_index: null,
          window_count: null,
          frame_idx: null,
          stage_history: [],
          message: 'done',
          error: null,
          started_at: 'now',
          updated_at: 'now',
          completed_at: 'now',
          result: {
            message: 'Video state initialized successfully',
            num_frames: 8,
            resolved_video_frames_dir: 'C:/backend/saved/review-run/frames',
            source_video_path: null,
            online_mode: true,
            batch_size: 32,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 11,
          },
        },
      }),
    );
    editor.store.loadSourceMode.set('saved_session_dir');
    editor.store.videoDir.set('C:/backend/saved/review-run');

    await editor.sessions.initVideo();

    expect(backendMock.initVideoState).toHaveBeenCalledWith('C:/backend/saved/review-run');
    expect(editor.store.isInitialized()).toBe(true);
  });

  it('restores interactive state from saved-session init result', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-restore',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-restore',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 10,
          total: 10,
          window_index: null,
          window_count: null,
          frame_idx: null,
          stage_history: [],
          message: 'done',
          error: null,
          started_at: 'now',
          updated_at: 'now',
          completed_at: 'now',
          result: {
            message: 'Video state initialized successfully',
            num_frames: 10,
            resolved_video_frames_dir: 'C:/backend/saved/review-run/frames',
            source_video_path: null,
            online_mode: true,
            batch_size: 32,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 12,
            source_type: 'saved_session',
            restored_session: {
              session_meta: { schema_version: 2 },
              has_mask_manifest: true,
              interactive_state: {
                version: 1,
                objects: [{ id: 1, name: 'Object 1', color: '#ff6600' }],
                selected_object_id: 1,
                interaction_mode: 'negative',
                current_frame_idx: 4,
                points: [{ frame_idx: 4, obj_id: 1, x: 10, y: 20, label: 1 }],
                live_masks: [{ frame_idx: 4, obj_id: 1, height: 2, width: 2, counts: [0, 1, 3] }],
              },
            },
          },
        },
      }),
    );
    editor.store.loadSourceMode.set('saved_session_dir');
    editor.store.videoDir.set('C:/backend/saved/review-run');

    await editor.sessions.initVideo();

    expect(editor.store.hasManifestMasks()).toBe(true);
    expect(editor.store.interactionMode()).toBe('negative');
    expect(editor.store.targetFrameIdx()).toBe(4);
    expect(editor.store.points().get(4)?.get(1)?.length).toBe(1);
    expect(editor.store.masks().get(4)?.get(1)).toEqual({
      size: [2, 2],
      rle: [[0, 1]],
      bbox: [0, 0, 1, 1],
    });
  });

  it('restores persisted tracking result from saved-session init result', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-restore-track',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-restore-track',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 10,
          total: 10,
          window_index: null,
          window_count: null,
          frame_idx: null,
          stage_history: [],
          message: 'done',
          error: null,
          started_at: 'now',
          updated_at: 'now',
          completed_at: 'now',
          result: {
            message: 'Video state initialized successfully',
            num_frames: 10,
            resolved_video_frames_dir: 'C:/backend/saved/review-run/frames',
            source_video_path: null,
            online_mode: true,
            batch_size: 32,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 12,
            source_type: 'saved_session',
            restored_session: {
              session_meta: { schema_version: 2 },
              has_mask_manifest: true,
              tracking_result: {
                result_id: 'track-restored',
                summary: { num_points: 1 },
              },
            },
          },
        },
      }),
    );
    backendMock.getTrackingResult.mockReturnValue(
      of({
        result: {
          version: 1,
          result_id: 'track-restored',
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
            {
              point_id: 'p0_0',
              obj_id: 1,
              source_frame_idx: 0,
              source_x: 10,
              source_y: 20,
            },
          ],
        },
      }),
    );
    editor.store.loadSourceMode.set('saved_session_dir');
    editor.store.videoDir.set('C:/backend/saved/review-run');

    await editor.sessions.initVideo();

    expect(backendMock.getTrackingResult).toHaveBeenCalledWith('track-restored');
    expect(editor.store.trackedPoints()[0].tracks).toEqual([
      [10, 20],
      [11, 21],
    ]);
  });

  it('keeps saved-session init successful when restored tracking result is unavailable', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-restore-track-missing',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-restore-track-missing',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 10,
          total: 10,
          window_index: null,
          window_count: null,
          frame_idx: null,
          stage_history: [],
          message: 'done',
          error: null,
          started_at: 'now',
          updated_at: 'now',
          completed_at: 'now',
          result: {
            message: 'Video state initialized successfully',
            num_frames: 10,
            resolved_video_frames_dir: 'C:/backend/saved/review-run/frames',
            source_video_path: null,
            online_mode: true,
            batch_size: 32,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 12,
            source_type: 'saved_session',
            restored_session: {
              session_meta: { schema_version: 2 },
              has_mask_manifest: true,
              tracking_result: { result_id: 'missing-track' },
            },
          },
        },
      }),
    );
    backendMock.getTrackingResult.mockReturnValue(
      throwError(() => ({ error: { detail: 'missing' } })),
    );
    editor.store.loadSourceMode.set('saved_session_dir');
    editor.store.videoDir.set('C:/backend/saved/review-run');

    await editor.sessions.initVideo();

    expect(editor.store.isInitialized()).toBe(true);
    expect(editor.store.toasts()[0].title).toBe('Tracking result unavailable');
  });

  it('creates an error toast when a job fails', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-2',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-2',
          operation: 'video_init',
          status: 'failed',
          stage: 'initializing_state',
          stage_label: 'Initializing video state',
          progress: 0.5,
          current: null,
          total: null,
          window_index: null,
          window_count: null,
          frame_idx: null,
          stage_history: [],
          message: 'failed',
          error: { code: 'validation_error', message: 'Path not found', detail: null },
          started_at: 'now',
          updated_at: 'now',
          completed_at: 'now',
          result: null,
        },
      }),
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
        saved_path: 'C:/project/backend/saved/review-run',
        state_epoch: 3,
      }),
    );
    editor.store.isInitialized.set(true);
    editor.store.saveName.set('review-run');
    objectOperations.renameObject(1, 'Left hand');

    editor.sessions.save();
    await Promise.resolve();

    expect(backendMock.saveVideoSession).toHaveBeenCalledWith(
      'review-run',
      expect.objectContaining({
        version: 1,
        objects: [{ id: 1, name: 'Left hand', color: '#ff0000' }],
        points: expect.any(Array),
        live_masks: expect.any(Array),
      }),
    );
    expect(editor.store.toasts()[0].title).toBe('Session saved');
  });
});
