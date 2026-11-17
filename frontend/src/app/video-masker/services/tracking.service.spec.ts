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

  it('takes the undo availability from a finished tracking run, which is a step', async () => {
    vi.spyOn(TestBed.inject(VideoJobsService), 'run').mockResolvedValue({
      result: {
        tracking_result_id: 'tracks-1',
        history: { undo: 2, redo: 0, next_undo: 'Track points', next_redo: null },
      },
      completedJobId: null,
    });
    backendMock.getTrackingResult.mockReturnValue(
      of({
        result: {
          points: [{ point_id: 'p1', obj_id: 1, source_frame_idx: 0, source_x: 1, source_y: 2 }],
          tracks: [[[1, 2]]],
          visibility: [[true]],
        },
      }),
    );
    await editor.tracking.runTracking();
    expect(editor.store.history()).toEqual({
      undo: 2,
      redo: 0,
      next_undo: 'Track points',
      next_redo: null,
    });
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
