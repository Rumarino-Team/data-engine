import { cachedMasks, seedMasks } from '../testing/workflow-test-helpers';
import { setAcceptedPoints } from '../testing/workflow-test-helpers';
import { VideoAddPointsResponse } from '../../services/backend.service';
import { TestBed } from '@angular/core/testing';
import { describe, beforeEach, expect, it } from 'vitest';
import { Subject, of, throwError } from 'rxjs';
import { createWorkflowFixture, makeInfo, makeResponse } from '../testing/workflow-test-helpers';

describe('point-editing workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];
  let objectOperations: ReturnType<typeof createWorkflowFixture>['objectOperations'];

  beforeEach(() => {
    ({ editor, backendMock, objectOperations } = createWorkflowFixture());
  });

  it('keeps accepted points unchanged while previewing an edit and on rejection', async () => {
    const response = new Subject<VideoAddPointsResponse>();
    backendMock.addNewPointsOrBox.mockReturnValue(response);
    const accepted = editor.store.points();
    const pending = editor.prompts.addPoint(10, 20, 1, 5);
    expect(editor.store.points()).toBe(accepted);
    expect(editor.store.previewPoints().get(5)?.get(1)).toEqual([{ x: 10, y: 20, label: 1 }]);
    response.error({ status: 422 });
    await pending;
    expect(editor.store.points()).toBe(accepted);
    expect(editor.store.pendingPointEdit()).toBeNull();
    expect(editor.store.annotationRecoveryRequested()).toBe(false);
  });

  it.each([0, 409, 500])(
    'recovers uncertain point status %s without replaying the request',
    async (status) => {
      editor.store.isInitialized.set(true);
      backendMock.getVideoInfo.mockReturnValue(of(makeInfo()));
      backendMock.addNewPointsOrBox.mockReturnValue(throwError(() => ({ status })));
      await editor.prompts.addPoint(10, 20, 1, 5);
      TestBed.tick();
      await Promise.resolve();
      expect(backendMock.addNewPointsOrBox).toHaveBeenCalledTimes(1);
      expect(backendMock.getVideoInfo).toHaveBeenCalledTimes(1);
      expect(editor.store.points().get(5)?.get(1)).toEqual([{ x: 10, y: 20, label: 1 }]);
      expect(editor.store.annotationRevision()).toBe(1);
    },
  );

  it('accepts the requested frame after navigation instead of rolling its points back', async () => {
    const response = new Subject<VideoAddPointsResponse>();
    backendMock.addNewPointsOrBox.mockReturnValue(response);
    const pending = editor.prompts.addPoint(10, 20, 1, 5);
    editor.store.displayedFrameIdx.set(8);
    response.next(makeResponse({}));
    await pending;
    expect(editor.store.displayedFrameIdx()).toBe(8);
    expect(editor.store.points().get(5)?.get(1)).toHaveLength(1);
    expect(cachedMasks(editor.framePipeline).has(5)).toBe(true);
    expect(cachedMasks(editor.framePipeline).has(8)).toBe(false);
  });

  it('recovers a rejected last-point removal using accepted state', async () => {
    setAcceptedPoints(editor.store, new Map([[5, new Map([[1, [{ x: 10, y: 20, label: 1 }]]])]]));
    editor.store.isInitialized.set(true);
    const pending = new Subject<VideoAddPointsResponse>();
    backendMock.clearAllPromptsInFrame.mockReturnValue(pending);
    const removal = editor.prompts.removePoint(1, 5, 0);
    expect(editor.store.points().get(5)?.get(1)).toHaveLength(1);
    expect(editor.store.previewPoints().size).toBe(0);
    expect(backendMock.clearAllPromptsInFrame).toHaveBeenCalledWith(5, 1, {
      session_id: 'session-1',
      expected_revision: 0,
    });
    backendMock.getVideoInfo.mockReturnValue(of(makeInfo()));
    pending.error({ status: 409 });
    await removal;
    TestBed.tick();
    await Promise.resolve();
    expect(editor.store.points().get(5)?.get(1)).toHaveLength(1);
    expect(backendMock.clearAllPromptsInFrame).toHaveBeenCalledTimes(1);
    expect(editor.store.pendingPointEdit()).toBeNull();
  });

  it('requires reload when recovery reports an unusable predictor', async () => {
    editor.store.isInitialized.set(true);
    backendMock.addNewPointsOrBox.mockReturnValue(throwError(() => ({ status: 503 })));
    backendMock.getVideoInfo.mockReturnValue(of(makeInfo({ predictor_ready: false })));
    await editor.prompts.addPoint(10, 20, 1, 5);
    TestBed.tick();
    await Promise.resolve();
    expect(editor.store.isInitialized()).toBe(false);
    expect(editor.store.sessionId()).toBe('session-1');
    expect(editor.store.isInteractionBusy()).toBe(false);
  });

  it.each([0, 409])(
    'recovers reset status %s without locally clearing accepted prompts',
    async (status) => {
      setAcceptedPoints(editor.store, new Map([[5, new Map([[1, [{ x: 10, y: 20, label: 1 }]]])]]));
      const accepted = editor.store.points();
      editor.store.isInitialized.set(true);
      backendMock.resetVideoState.mockReturnValue(throwError(() => ({ status })));
      backendMock.getVideoInfo.mockReturnValue(of(makeInfo()));
      editor.propagation.clearMasks();
      expect(editor.store.points()).toBe(accepted);
      expect(backendMock.resetVideoState).toHaveBeenCalledWith({
        session_id: 'session-1',
        expected_revision: 0,
      });
      TestBed.tick();
      await Promise.resolve();
      expect(backendMock.resetVideoState).toHaveBeenCalledTimes(1);
      expect(backendMock.getVideoInfo).toHaveBeenCalledTimes(1);
      expect(editor.store.points().get(5)?.get(1)).toHaveLength(1);
    },
  );

  it('cancels a pending preview when the editor is destroyed', async () => {
    const response = new Subject<VideoAddPointsResponse>();
    backendMock.addNewPointsOrBox.mockReturnValue(response);
    const accepted = editor.store.points();
    const pending = editor.prompts.addPoint(10, 20, 1, 5);
    TestBed.resetTestingModule();
    await pending;
    expect(editor.store.points()).toBe(accepted);
    expect(editor.store.pendingPointEdit()).toBeNull();
    expect(editor.store.isPointRequestInFlight()).toBe(false);
    expect(editor.store.annotationRecoveryRequested()).toBe(false);
  });

  it('discards a late response from a replaced session even at the same epoch', async () => {
    const response = new Subject<VideoAddPointsResponse>();
    backendMock.addNewPointsOrBox.mockReturnValue(response);
    const pending = editor.prompts.addPoint(10, 20, 1, 5);
    editor.store.applyAnnotationSnapshot(
      makeInfo({ session_id: 'replacement' }),
      backendMock.getApiUrl(),
      true,
    );
    const accepted = editor.store.annotations();
    response.next(makeResponse({}));
    await pending;
    expect(editor.store.annotations()).toBe(accepted);
    expect(editor.store.sessionId()).toBe('replacement');
    expect(editor.store.annotationRecoveryRequested()).toBe(false);
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
      setAcceptedPoints(editor.store, replacementPoints);
      seedMasks(editor, replacementMasks);
      const encodedReplacement = cachedMasks(editor.framePipeline);
      if (outcome === 'success') {
        response$.next(makeResponse({}));
        response$.complete();
      } else {
        response$.error(new Error('Old failure'));
      }
      await pending;
      expect(editor.store.stateEpoch()).toBe(4);
      expect(editor.store.points()).toEqual(replacementPoints);
      expect(cachedMasks(editor.framePipeline)).toEqual(encodedReplacement);
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
    expect(cachedMasks(editor.framePipeline).size).toBe(0);
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
      { session_id: 'session-1', expected_revision: 0 },
    );
    expect(cachedMasks(editor.framePipeline).get(5)?.get(1)).toEqual({
      size: [2, 2],
      rle: [[0, 1]],
      bbox: [0, 0, 1, 1],
    });
    expect(editor.store.lastDiscardReason()).toBeNull();
  });

  it('discards mismatched response frames without changing accepted points', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(
      of(makeResponse({ request_frame_idx: 5, frame_idx: 4 })),
    );

    await editor.prompts.addPoint(10, 20, 1, 5);

    expect(cachedMasks(editor.framePipeline).get(5)?.get(1)).toBeUndefined();
    expect(editor.store.points().get(5)?.get(1)?.length ?? 0).toBe(0);
    expect(editor.store.lastDiscardReason()).toContain('frame mismatch');
  });

  it('accepts packed masks with matching pixel metadata', async () => {
    const mask = {
      size: [2, 2] as [number, number],
      encoding: 'packed-bits' as const,
      data: 'gA==',
    };
    backendMock.addNewPointsOrBox.mockReturnValue(
      of(
        makeResponse({
          mask_data: { ...makeResponse({}).mask_data!, objects: { '1': mask } },
        }),
      ),
    );
    await editor.prompts.addPoint(12, 24, 1, 5);
    expect(cachedMasks(editor.framePipeline).get(5)?.get(1)).toBe(mask);
  });

  it.each([
    { size: [2, 2], encoding: 'packed-bits', data: 'gQ==' },
    { size: [2, 2], rle: [[3, 2]], bbox: [0, 0, 2, 2] },
  ])('recovers after malformed effective mask payloads (%j)', async (mask) => {
    const response = makeResponse({});
    response.mask_data!.objects['1'] = mask as NonNullable<
      VideoAddPointsResponse['mask_data']
    >['objects']['1'];
    backendMock.addNewPointsOrBox.mockReturnValue(of(response));
    await editor.prompts.addPoint(12, 24, 1, 5);
    expect(cachedMasks(editor.framePipeline).size).toBe(0);
    expect(editor.store.annotationRecoveryRequested()).toBe(true);
  });

  it('discards stale epoch responses and clears live masks', async () => {
    const existingMasks = new Map<number, Map<number, boolean[][]>>();
    existingMasks.set(2, new Map([[1, [[true]]]]));
    seedMasks(editor, existingMasks);

    backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({ state_epoch: 4 })));

    await editor.prompts.addPoint(14, 18, 1, 5);

    expect(editor.store.stateEpoch()).toBe(4);
    expect(cachedMasks(editor.framePipeline).size).toBe(0);
    expect(editor.store.lastDiscardReason()).toContain('epoch mismatch');
  });

  it('accepts empty interactive results without local suppression markers', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(
      of(
        makeResponse({
          mask_data: {
            ...makeResponse({}).mask_data!,
            objects: { '1': { size: [2, 2], rle: [], bbox: [0, 0, 0, 0] } },
          },
          mask_pixel_counts: { 1: 0 },
        }),
      ),
    );

    await editor.prompts.addPoint(30, 40, 1, 5);

    expect(editor.store.lastMaskPixelCount()).toBe(0);
  });

  it('keeps masks unchanged while a prompt preview is pending', async () => {
    const response$ = new Subject<VideoAddPointsResponse>();
    backendMock.addNewPointsOrBox.mockReturnValue(response$);

    const pendingRequest = editor.prompts.addPoint(30, 40, 1, 5);

    expect(editor.store.points().get(5)?.get(1)).toBeUndefined();
    expect(editor.store.previewPoints().get(5)?.get(1)?.length).toBe(1);

    response$.next(makeResponse({}));
    response$.complete();
    await pendingRequest;
  });

  it('discards the preview and requests recovery when point update conflicts', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(
      throwError(() => ({ error: { detail: 'Another operation is already running.' } })),
    );

    await editor.prompts.addPoint(30, 40, 1, 5);

    expect(editor.store.points().get(5)?.get(1)?.length ?? 0).toBe(0);
    expect(editor.store.toasts()[0].title).toBe('Point update failed');
    expect(editor.store.toasts()[0].message).toBe('Another operation is already running.');
  });
});
