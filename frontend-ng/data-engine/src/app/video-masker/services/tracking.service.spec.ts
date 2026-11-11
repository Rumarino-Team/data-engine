import { describe, beforeEach, expect, it, vi } from 'vitest';
import { Subject, of } from 'rxjs';
import { TestBed } from '@angular/core/testing';
import { VideoJobsService } from './video-jobs.service';
import { createWorkflowFixture, makeResponse, mockJob } from '../testing/workflow-test-helpers';

describe('tracking workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];

  beforeEach(() => {
    ({ editor, backendMock } = createWorkflowFixture());
  });

  it('keeps the editor busy while the finished result loads', async () => {
    vi.spyOn(TestBed.inject(VideoJobsService), 'run').mockResolvedValue({
      result: { tracking_result_id: 'tracks-1' },
      completedJobId: null,
    });
    const result$ = new Subject<unknown>();
    backendMock.getTrackingResult.mockReturnValue(result$);
    const tracking = editor.tracking.runTracking();
    await vi.waitFor(() => expect(backendMock.getTrackingResult).toHaveBeenCalledWith('tracks-1'));
    expect(editor.store.isInteractionBusy()).toBe(true);
    backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({})));
    await editor.prompts.addPoint(10, 20, 1, 5);
    expect(backendMock.addNewPointsOrBox).not.toHaveBeenCalled();
    result$.next({
      result: {
        points: [{ point_id: 'p1', obj_id: 1, source_frame_idx: 0, source_x: 1, source_y: 2 }],
        tracks: [[[1, 2]]],
        visibility: [[true]],
      },
    });
    result$.complete();
    await tracking;
    expect(editor.store.trackedPoints()).toHaveLength(1);
    expect(editor.store.isInteractionBusy()).toBe(false);
  });

  it('runs CoTracker without sending a model selector value', async () => {
    mockJob(
      backendMock.trackPromptPoints,
      backendMock.getJob,
      'prompt_tracking',
      {
        result: {
          message: 'Prompt-point tracking completed',
          model_name: 'cotracker3_online',
          num_points: 1,
          num_frames: 2,
          add_support_grid_used: true,
          tracking_mode: 'streaming',
          streaming_frame_threshold: 256,
          tracking_result_id: 'track-1',
        },
      },
      'job-track',
    );
    backendMock.getTrackingResult.mockReturnValue(
      of({
        result: {
          version: 3,
          session_id: 'session-1',
          annotation_revision: editor.store.annotationRevision(),
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

    expect(backendMock.trackPromptPoints).toHaveBeenCalledWith(
      { add_support_grid: true },
      { session_id: 'session-1', expected_revision: editor.store.annotationRevision() },
    );
    expect(backendMock.trackPromptPoints.mock.calls[0][0]).not.toHaveProperty('model_name');
    expect(backendMock.getTrackingResult).toHaveBeenCalledWith('track-1');
    expect(backendMock.clearJobResult).toHaveBeenCalledWith('job-track');
    expect(editor.store.trackedPoints()[0].tracks).toEqual([
      [10, 20],
      [11, 21],
    ]);
  });
  it('retains only backend-selected tracks after annotation recovery', () => {
    const point = {
      point_id: 'keep',
      obj_id: 1,
      source_frame_idx: 0,
      source_x: 1,
      source_y: 2,
      tracks: [[1, 2]],
      visibility: [true],
    };
    editor.store.trackingResultId = 'tracks';
    editor.store.trackedPoints.set([point, { ...point, point_id: 'remove' }]);
    editor.store.applyAnnotationSnapshot({
      session_id: editor.store.sessionId(),
      annotations: editor.store.annotations(),
      tracking_validity: { result_id: 'tracks', point_ids: ['keep'] },
    });
    expect(editor.store.trackedPoints()).toEqual([point]);
    editor.store.applyAnnotationSnapshot({
      session_id: editor.store.sessionId(),
      annotations: editor.store.annotations(),
      tracking_validity: null,
    });
    expect(editor.store.trackedPoints()).toEqual([]);
  });
});
