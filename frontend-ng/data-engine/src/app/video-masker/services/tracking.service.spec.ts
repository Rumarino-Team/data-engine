import { describe, beforeEach, expect, it, vi } from 'vitest';
import { Subject, of } from 'rxjs';
import { TestBed } from '@angular/core/testing';
import { VideoJobsService } from './video-jobs.service';
import { createWorkflowFixture, makeResponse } from '../testing/workflow-test-helpers';

describe('tracking workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];

  beforeEach(() => {
    ({ editor, backendMock } = createWorkflowFixture());
  });

  it('does not restore an old tracking result after a successful prompt edit', async () => {
    vi.spyOn(TestBed.inject(VideoJobsService), 'run').mockResolvedValue({
      result: { tracking_result_id: 'old-tracks', state_epoch: 3 },
      completedJobId: null,
    });
    const result$ = new Subject<unknown>();
    backendMock.getTrackingResult.mockReturnValue(result$);
    const tracking = editor.tracking.runTracking();
    await vi.waitFor(() =>
      expect(backendMock.getTrackingResult).toHaveBeenCalledWith('old-tracks'),
    );
    backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({})));
    await editor.prompts.addPoint(10, 20, 1, 5);
    result$.next({
      result: {
        points: [{ point_id: 'p1', obj_id: 1, source_frame_idx: 0, source_x: 1, source_y: 2 }],
        tracks: [[[1, 2]]],
        visibility: [[true]],
      },
    });
    result$.complete();
    await tracking;
    expect(editor.store.trackedPoints()).toEqual([]);
  });

  it('runs CoTracker without sending a model selector value', async () => {
    backendMock.trackPromptPoints.mockReturnValue(
      of({
        job_id: 'job-track',
        status: 'queued',
        operation: 'prompt_tracking',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-track',
          operation: 'prompt_tracking',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 1,
          total: 1,
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
            message: 'Prompt-point tracking completed',
            model_name: 'cotracker3_online',
            num_points: 1,
            num_frames: 2,
            add_support_grid_used: true,
            tracking_mode: 'streaming',
            streaming_frame_threshold: 256,
            tracking_result_id: 'track-1',
            state_epoch: 3,
          },
        },
      }),
    );
    backendMock.getTrackingResult.mockReturnValue(
      of({
        result: {
          version: 1,
          result_id: 'track-1',
          message: 'Prompt-point tracking completed',
          model_name: 'cotracker3_online',
          num_points: 1,
          num_frames: 2,
          add_support_grid_used: true,
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
    editor.store.trackingUseSupportGrid.set(true);

    await editor.tracking.runTracking();

    expect(backendMock.trackPromptPoints).toHaveBeenCalledWith({ add_support_grid: true });
    expect(backendMock.trackPromptPoints.mock.calls[0][0]).not.toHaveProperty('model_name');
    expect(backendMock.getTrackingResult).toHaveBeenCalledWith('track-1');
    expect(backendMock.clearJobResult).toHaveBeenCalledWith('job-track');
    expect(editor.store.trackedPoints()[0].tracks).toEqual([
      [10, 20],
      [11, 21],
    ]);
  });
});
