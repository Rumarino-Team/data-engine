import { beforeEach, describe, expect, it } from 'vitest';
import { createWorkflowFixture, objectUpdate } from '../testing/workflow-test-helpers';

describe('outdated mask ranges in the store', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];

  const snapshot = (outdated?: Record<string, [number, number][]>) => ({
    session_id: 'session-1',
    annotations: structuredClone(editor.store.annotations()!),
    ...(outdated === undefined ? {} : { outdated_masks: outdated }),
  });

  beforeEach(() => {
    ({ editor } = createWorkflowFixture());
  });

  it('takes the ranges from a snapshot and defaults to none when the backend sends none', () => {
    expect(editor.store.outdatedMasks()).toEqual({});
    editor.store.applyAnnotationSnapshot(snapshot({ '1': [[2, 9]] }), true);
    expect(editor.store.outdatedMasks()).toEqual({ '1': [[2, 9]] });
    editor.store.applyAnnotationSnapshot(snapshot(), true);
    expect(editor.store.outdatedMasks()).toEqual({});
  });

  it('does not share the snapshot object it was given', () => {
    const ranges: Record<string, [number, number][]> = { '1': [[2, 9]] };
    editor.store.applyAnnotationSnapshot(snapshot(ranges), true);
    ranges['1'].push([20, 30]);
    expect(editor.store.outdatedMasks()).toEqual({ '1': [[2, 9]] });
  });

  it('follows an edit: the ranges in a mutation response replace the old ones', () => {
    editor.store.outdatedMasks.set({ '1': [[0, 99]] });
    const outdated: Record<string, [number, number][]> = { '1': [[5, 7]] };
    const update = { ...objectUpdate(editor.store, editor.store.objects()), outdated_masks: outdated };
    expect(editor.store.applyAnnotationUpdate(update)).toBe(true);
    expect(editor.store.outdatedMasks()).toEqual({ '1': [[5, 7]] });
  });

  it('keeps what it has when a response says nothing about them', () => {
    editor.store.outdatedMasks.set({ '1': [[0, 99]] });
    expect(editor.store.applyAnnotationUpdate(objectUpdate(editor.store, editor.store.objects()))).toBe(true);
    expect(editor.store.outdatedMasks()).toEqual({ '1': [[0, 99]] });
  });

  it('forgets them when the session is closed', () => {
    editor.store.outdatedMasks.set({ '1': [[0, 99]] });
    editor.store.resetSession();
    expect(editor.store.outdatedMasks()).toEqual({});
  });
});

describe('undo and redo availability in the store', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  const steps = { undo: 3, redo: 1, next_undo: 'Add points', next_redo: 'Edit object' };

  beforeEach(() => {
    ({ editor } = createWorkflowFixture());
  });

  it('has nothing to undo before the backend says otherwise', () => {
    expect(editor.store.history()).toEqual({ undo: 0, redo: 0, next_undo: null, next_redo: null });
  });

  it('takes it from a snapshot, and from every mutation response that carries it', () => {
    editor.store.applyAnnotationSnapshot(
      { session_id: 'session-1', annotations: structuredClone(editor.store.annotations()!), history: steps },
      true,
    );
    expect(editor.store.history()).toEqual(steps);
    const next = { ...steps, undo: 4, redo: 0, next_redo: null };
    editor.store.applyAnnotationUpdate({ ...objectUpdate(editor.store, editor.store.objects()), history: next });
    expect(editor.store.history()).toEqual(next);
  });

  it('falls back to none for a snapshot without it, and keeps what it has for an update without it', () => {
    editor.store.history.set(steps);
    editor.store.applyAnnotationUpdate(objectUpdate(editor.store, editor.store.objects()));
    expect(editor.store.history()).toEqual(steps);
    editor.store.applyAnnotationSnapshot(
      { session_id: 'session-1', annotations: structuredClone(editor.store.annotations()!) },
      true,
    );
    expect(editor.store.history().undo).toBe(0);
  });

  it('forgets it when the session is closed', () => {
    editor.store.history.set(steps);
    editor.store.resetSession();
    expect(editor.store.history().undo).toBe(0);
  });
});

describe("tracked points follow the session's tracking result", () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  const series = (id: string) => ({
    point_id: id, obj_id: 1, source_frame_idx: 0, source_x: 1, source_y: 2, tracks: [[1, 2]], visibility: [true],
  });

  beforeEach(() => {
    ({ editor } = createWorkflowFixture());
    editor.store.trackingResultId = 'track-1';
    editor.store.trackedPoints.set([series('p0'), series('p1')]);
  });

  const update = (validity: { result_id: string; point_ids: string[] } | null) => ({
    ...objectUpdate(editor.store, editor.store.objects()),
    tracking_validity: validity,
  });

  it('keeps only the points that are still valid for the result on screen', () => {
    editor.store.applyAnnotationUpdate(update({ result_id: 'track-1', point_ids: ['p1'] }));
    expect(editor.store.trackedPoints().map((point) => point.point_id)).toEqual(['p1']);
    expect(editor.store.trackingResultId).toBe('track-1');
  });

  it('drops the overlays and forgets the result when the session no longer has one', () => {
    editor.store.applyAnnotationUpdate(update(null));
    expect(editor.store.trackedPoints()).toEqual([]);
    expect(editor.store.trackingResultId).toBeNull();
  });

  it('drops them and forgets the result when the session holds a different one', () => {
    editor.store.applyAnnotationUpdate(update({ result_id: 'track-2', point_ids: ['p0'] }));
    expect(editor.store.trackedPoints()).toEqual([]);
    expect(editor.store.trackingResultId).toBeNull();
  });
});
