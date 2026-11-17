import { of, throwError } from 'rxjs';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { HistoryStatus } from '../../services/backend.service';
import { createWorkflowFixture, objectUpdate } from '../testing/workflow-test-helpers';

describe('undo and redo', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];
  let history: ReturnType<typeof createWorkflowFixture>['history'];

  const status = (overrides: Partial<HistoryStatus> = {}): HistoryStatus => ({
    undo: 1,
    redo: 2,
    next_undo: 'Add points',
    next_redo: 'Edit object',
    ...overrides,
  });
  const answer = (message: string, next: HistoryStatus = status(), extra: object = {}) => ({
    ...objectUpdate(editor.store, editor.store.objects()),
    message,
    history: next,
    ...extra,
  });
  const series = () => ({
    point_id: 'p0',
    obj_id: 1,
    source_frame_idx: 0,
    source_x: 1,
    source_y: 2,
    tracks: [[1, 2]],
    visibility: [true],
  });
  const storedResult = () =>
    of({
      result: {
        points: [{ point_id: 'p0', obj_id: 1, source_frame_idx: 0, source_x: 1, source_y: 2 }],
        tracks: [[[1, 2]]],
        visibility: [[true]],
      },
    });

  beforeEach(() => {
    ({ editor, backendMock, history } = createWorkflowFixture());
    editor.store.isInitialized.set(true);
    editor.store.history.set(status({ undo: 2, redo: 1 }));
  });

  it('is available only when the backend has steps and the editor is free', () => {
    expect([history.canUndo(), history.canRedo()]).toEqual([true, true]);
    editor.store.history.set(status({ undo: 0, redo: 0, next_undo: null, next_redo: null }));
    expect([history.canUndo(), history.canRedo()]).toEqual([false, false]);
    editor.store.history.set(status());
    editor.store.isLoading.set(true);
    expect([history.canUndo(), history.canRedo()]).toEqual([false, false]);
    editor.store.isLoading.set(false);
    editor.store.isInitialized.set(false);
    expect([history.canUndo(), history.canRedo()]).toEqual([false, false]);
  });

  it('undoes: names session and revision, applies the answer and refreshes what is shown', async () => {
    const revision = editor.store.annotationRevision();
    backendMock.undoVideoChange.mockReturnValue(
      of(answer('Undo: Add points', status({ undo: 1, redo: 2 }))),
    );
    const clear = vi.spyOn(editor.framePipeline, 'clearMaskDataCache');
    const schedule = vi.spyOn(editor.framePipeline, 'scheduleFrameLoad');

    await expect(history.undo()).resolves.toBe(true);

    expect(backendMock.undoVideoChange).toHaveBeenCalledWith({
      session_id: 'session-1',
      expected_revision: revision,
    });
    expect(editor.store.history()).toEqual(status({ undo: 1, redo: 2 }));
    expect(editor.store.annotationRevision()).toBe(revision! + 1);
    expect(clear).toHaveBeenCalled();
    expect(schedule).toHaveBeenCalledWith(editor.store.targetFrameIdx());
    expect(editor.store.toasts()[0]).toMatchObject({
      severity: 'info',
      title: 'Undone',
      message: 'Add points',
    });
    expect(editor.store.isLoading()).toBe(false);
  });

  it('does nothing when there is nothing to do, no session is shown or work is in flight', async () => {
    editor.store.history.set(status({ undo: 0, redo: 0, next_undo: null, next_redo: null }));
    await expect(history.undo()).resolves.toBe(false);
    await expect(history.redo()).resolves.toBe(false);
    editor.store.history.set(status());
    editor.store.isLoading.set(true);
    await expect(history.undo()).resolves.toBe(false);
    editor.store.isLoading.set(false);
    editor.store.isInitialized.set(false);
    await expect(history.redo()).resolves.toBe(false);
    expect(backendMock.undoVideoChange).not.toHaveBeenCalled();
    expect(backendMock.redoVideoChange).not.toHaveBeenCalled();
  });

  describe('when the backend refuses', () => {
    it('reloads annotations after a conflict and says why', async () => {
      backendMock.undoVideoChange.mockReturnValue(
        throwError(() => ({ status: 409, error: { detail: 'Annotations changed.' } })),
      );
      await expect(history.undo()).resolves.toBe(false);
      expect(editor.store.annotationRecoveryRequested()).toBe(true);
      expect(editor.store.toasts()[0]).toMatchObject({
        severity: 'error',
        title: 'Undo failed',
        message: 'Annotations changed.',
      });
      expect(editor.store.isLoading()).toBe(false);
    });

    it('reloads when the backend cannot answer', async () => {
      backendMock.redoVideoChange.mockReturnValue(throwError(() => ({ status: 0 })));
      await expect(history.redo()).resolves.toBe(false);
      expect(editor.store.annotationRecoveryRequested()).toBe(true);
      expect(editor.store.toasts()[0].title).toBe('Redo failed');
    });
  });

  describe('tracked points', () => {
    it('loads the tracking result the restored state names, when it is a different one', async () => {
      editor.store.trackingResultId = 'track-new';
      backendMock.undoVideoChange.mockReturnValue(
        of(
          answer('Undo: Track points', status(), {
            tracking_validity: { result_id: 'track-old', point_ids: ['p0'] },
          }),
        ),
      );
      backendMock.getTrackingResult.mockReturnValue(storedResult());
      await history.undo();
      expect(backendMock.getTrackingResult).toHaveBeenCalledWith('track-old');
      expect(editor.store.trackingResultId).toBe('track-old');
      expect(editor.store.trackedPoints()).toHaveLength(1);
    });

    it('shows the overlays again when a tracking step is undone and then redone', async () => {
      editor.store.trackingResultId = 'track-1';
      editor.store.trackedPoints.set([series()]);
      backendMock.undoVideoChange.mockReturnValue(
        of(answer('Undo: Track points', status({ undo: 1, redo: 1 }), { tracking_validity: null })),
      );
      await history.undo();
      expect(editor.store.trackedPoints()).toEqual([]);

      backendMock.redoVideoChange.mockReturnValue(
        of(
          answer('Redo: Track points', status({ undo: 2, redo: 0 }), {
            tracking_validity: { result_id: 'track-1', point_ids: ['p0'] },
          }),
        ),
      );
      backendMock.getTrackingResult.mockReturnValue(storedResult());
      await history.redo();
      expect(backendMock.getTrackingResult).toHaveBeenCalledWith('track-1');
      expect(editor.store.trackedPoints()).toHaveLength(1);
    });

    it('drops the overlays when the restored state has no tracking result', async () => {
      editor.store.trackingResultId = 'track-1';
      editor.store.trackedPoints.set([series()]);
      backendMock.undoVideoChange.mockReturnValue(
        of(answer('Undo: Track points', status(), { tracking_validity: null })),
      );
      await history.undo();
      expect(editor.store.trackedPoints()).toEqual([]);
      expect(backendMock.getTrackingResult).not.toHaveBeenCalled();
    });
  });
});
