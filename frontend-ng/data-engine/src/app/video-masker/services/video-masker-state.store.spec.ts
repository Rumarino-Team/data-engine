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
