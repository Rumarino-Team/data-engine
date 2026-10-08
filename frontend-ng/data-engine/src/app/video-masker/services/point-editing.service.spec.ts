import { VideoAddPointsResponse } from '../../services/backend.service';
import { describe, beforeEach, expect, it, vi } from 'vitest';
import { Subject, of, throwError } from 'rxjs';
import { createWorkflowFixture, makeResponse } from '../testing/workflow-test-helpers';

describe('point-editing workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];
  let objectOperations: ReturnType<typeof createWorkflowFixture>['objectOperations'];

  beforeEach(() => {
    ({ editor, backendMock, objectOperations } = createWorkflowFixture());
  });

  it('blocks conflicting workflows while a point update is pending', async () => {
    const response$ = new Subject<VideoAddPointsResponse>();
    backendMock.addNewPointsOrBox.mockReturnValue(response$);
    const pending = editor.prompts.addPoint(10, 20, 1, 5);
    await editor.prompts.addPoint(30, 40, 0, 5);
    objectOperations.removeObject();
    await objectOperations.removeAllObjects();
    editor.propagation.clearMasks();
    objectOperations.addObject();
    editor.sessions.save();
    await editor.propagation.propagate();
    await editor.tracking.runTracking();
    expect(backendMock.addNewPointsOrBox).toHaveBeenCalledTimes(1);
    expect(backendMock.removeObject).not.toHaveBeenCalled();
    expect(backendMock.resetVideoState).not.toHaveBeenCalled();
    expect(backendMock.propagateInVideo).not.toHaveBeenCalled();
    expect(backendMock.trackPromptPoints).not.toHaveBeenCalled();
    expect(backendMock.saveVideoSession).not.toHaveBeenCalled();
    expect(editor.store.objects()).toHaveLength(1);
    response$.next(makeResponse({}));
    response$.complete();
    await pending;
    expect(editor.store.isInteractionBusy()).toBe(false);
  });

  it.each(['success', 'failure'])(
    'ignores a late point %s after the session epoch changes',
    async (outcome) => {
      const response$ = new Subject<VideoAddPointsResponse>();
      backendMock.addNewPointsOrBox.mockReturnValue(response$);
      const pending = editor.prompts.addPoint(10, 20, 1, 5);
      editor.store.stateEpoch.set(4);
      const replacementPoints = new Map([[5, new Map([[1, [{ x: 50, y: 60, label: 0 }]]])]]);
      const replacementMasks = new Map([[5, new Map([[1, [[false]]]])]]);
      editor.store.points.set(replacementPoints);
      editor.store.masks.set(replacementMasks);
      if (outcome === 'success') {
        response$.next(makeResponse({}));
        response$.complete();
      } else {
        response$.error(new Error('Old failure'));
      }
      await pending;
      expect(editor.store.stateEpoch()).toBe(4);
      expect(editor.store.points()).toBe(replacementPoints);
      expect(editor.store.masks()).toBe(replacementMasks);
      expect(editor.store.isPointRequestInFlight()).toBe(false);
    },
  );

  it('does not move the current epoch backwards on an older backend response', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({ state_epoch: 2 })));
    await editor.prompts.addPoint(10, 20, 1, 5);
    expect(editor.store.stateEpoch()).toBe(3);
    expect(editor.store.points().size).toBe(0);
  });

  it('rejects a response when only the current epoch changed during the request', async () => {
    const response$ = new Subject<VideoAddPointsResponse>();
    backendMock.addNewPointsOrBox.mockReturnValue(response$);
    const pending = editor.prompts.addPoint(10, 20, 1, 5);
    editor.store.stateEpoch.set(4);
    response$.next(makeResponse({}));
    response$.complete();
    await pending;
    expect(editor.store.stateEpoch()).toBe(4);
    expect(editor.store.masks().size).toBe(0);
    expect(editor.store.lastDiscardReason()).toContain('editing state changed');
  });

  it.each([true, false])(
    'invalidates tracking only after a successful point update (success=%s)',
    async (succeeds) => {
      const tracks = [
        {
          point_id: 'p1',
          obj_id: 1,
          source_frame_idx: 0,
          source_x: 1,
          source_y: 2,
          tracks: [[1, 2]],
          visibility: [true],
        },
      ];
      editor.store.trackedPoints.set(tracks);
      backendMock.addNewPointsOrBox.mockReturnValue(
        succeeds ? of(makeResponse({})) : throwError(() => new Error('Failed')),
      );
      await editor.prompts.addPoint(10, 20, 1, 5);
      expect(editor.store.trackedPoints()).toEqual(succeeds ? [] : tracks);
    },
  );

  it('uses displayed frame index in request and stores mask on that frame', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({})));

    await editor.prompts.addPoint(12, 24, 1, 5);

    expect(backendMock.addNewPointsOrBox).toHaveBeenCalledWith(
      expect.objectContaining({ frame_idx: 5 }),
    );
    expect(editor.store.masks().get(5)?.get(1)).toEqual({
      size: [2, 2],
      rle: [[0, 1]],
      bbox: [0, 0, 1, 1],
    });
    expect(editor.store.lastDiscardReason()).toBeNull();
  });

  it('discards mismatched response frame and rolls back optimistic point', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(
      of(makeResponse({ request_frame_idx: 5, frame_idx: 4 })),
    );

    await editor.prompts.addPoint(10, 20, 1, 5);

    expect(editor.store.masks().get(5)?.get(1)).toBeUndefined();
    expect(editor.store.points().get(5)?.get(1)?.length ?? 0).toBe(0);
    expect(editor.store.lastDiscardReason()).toContain('frame mismatch');
  });

  it('accepts boolean-grid responses from an older backend', async () => {
    const mask = [
      [true, false],
      [false, false],
    ];
    backendMock.addNewPointsOrBox.mockReturnValue(
      of(makeResponse({ out_masks: [mask], mask_encoding: undefined })),
    );
    await editor.prompts.addPoint(12, 24, 1, 5);
    expect(editor.store.masks().get(5)?.get(1)).toBe(mask);
  });

  it('accepts packed masks with matching pixel metadata', async () => {
    const mask = {
      size: [2, 2] as [number, number],
      encoding: 'packed-bits' as const,
      data: 'gA==',
    };
    backendMock.addNewPointsOrBox.mockReturnValue(
      of(makeResponse({ out_masks: [mask], mask_encoding: 'mixed' })),
    );
    await editor.prompts.addPoint(12, 24, 1, 5);
    expect(editor.store.masks().get(5)?.get(1)).toBe(mask);
  });

  it.each([
    { out_masks: [] },
    {
      out_masks: [{ size: [2, 2], encoding: 'packed-bits', data: 'gQ==' }],
      mask_encoding: 'mixed',
    },
    {
      out_masks: [{ size: [2, 2], encoding: 'packed-bits', data: 'wA==' }],
      mask_encoding: 'mixed',
    },
    { out_masks: [{ size: [2, 2], rle: [[3, 2]], bbox: [0, 0, 2, 2] }] },
    { mask_pixel_counts: { 1: 2 } },
    { mask_shapes: { 1: [3, 2] } },
  ])('rejects malformed encoded responses before applying masks (%j)', async (overrides) => {
    const previous = {
      size: [2, 2] as [number, number],
      rle: [],
      bbox: [0, 0, 0, 0] as [number, number, number, number],
    };
    editor.store.masks.set(new Map([[5, new Map([[1, previous]])]]));
    backendMock.addNewPointsOrBox.mockReturnValue(
      of(makeResponse(overrides as Partial<VideoAddPointsResponse>)),
    );
    await editor.prompts.addPoint(12, 24, 1, 5);
    expect(editor.store.masks().get(5)?.get(1)).toBe(previous);
    expect(editor.store.points().get(5)?.get(1)?.length ?? 0).toBe(0);
  });

  it('discards stale epoch responses and clears live masks', async () => {
    const existingMasks = new Map<number, Map<number, boolean[][]>>();
    existingMasks.set(2, new Map([[1, [[true]]]]));
    editor.store.masks.set(existingMasks);
    editor.store.liveEditedObjectFrames.set(new Map([[2, new Set([1])]]));

    backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({ state_epoch: 4 })));

    await editor.prompts.addPoint(14, 18, 1, 5);

    expect(editor.store.stateEpoch()).toBe(4);
    expect(editor.store.masks().size).toBe(0);
    expect(editor.store.liveEditedObjectFrames().size).toBe(0);
    expect(editor.store.lastDiscardReason()).toContain('epoch mismatch');
  });

  it('keeps frame/object marked as live-edited even when returned mask is empty', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(
      of(
        makeResponse({
          out_masks: [{ size: [2, 2], rle: [], bbox: [0, 0, 0, 0] }],
          mask_pixel_counts: { 1: 0 },
        }),
      ),
    );

    await editor.prompts.addPoint(30, 40, 1, 5);

    expect(editor.store.liveEditedObjectFrames().get(5)?.has(1)).toBe(true);
    expect(editor.store.lastMaskPixelCount()).toBe(0);
  });

  it('does not mark an object live-edited until the point mask response returns', async () => {
    const response$ = new Subject<VideoAddPointsResponse>();
    backendMock.addNewPointsOrBox.mockReturnValue(response$);

    const pendingRequest = editor.prompts.addPoint(30, 40, 1, 5);

    expect(editor.store.points().get(5)?.get(1)?.length).toBe(1);
    expect(editor.store.liveEditedObjectFrames().get(5)?.has(1)).toBeFalsy();

    response$.next(makeResponse({}));
    response$.complete();
    await pendingRequest;

    expect(editor.store.liveEditedObjectFrames().get(5)?.has(1)).toBe(true);
  });

  it('rolls back optimistic point and shows a toast when point update conflicts', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(
      throwError(() => ({ error: { detail: 'Another operation is already running.' } })),
    );

    await editor.prompts.addPoint(30, 40, 1, 5);

    expect(editor.store.points().get(5)?.get(1)?.length ?? 0).toBe(0);
    expect(editor.store.toasts()[0].title).toBe('Point update failed');
    expect(editor.store.toasts()[0].message).toBe('Another operation is already running.');
  });
});
