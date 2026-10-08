import { describe, beforeEach, expect, it, vi } from 'vitest';
import { Subject, of, throwError } from 'rxjs';
import { createWorkflowFixture } from '../testing/workflow-test-helpers';

describe('object-operations workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];
  let objectOperations: ReturnType<typeof createWorkflowFixture>['objectOperations'];

  beforeEach(() => {
    ({ editor, backendMock, objectOperations } = createWorkflowFixture());
  });

  it('allocates unique IDs across sparse objects and removal of the highest ID', () => {
    editor.store.objects.set([
      { id: 1, name: 'One', color: '#ff0000' },
      { id: 3, name: 'Three', color: '#00ff00' },
    ]);
    objectOperations.addObject();
    expect(editor.store.selectedObjectId()).toBe(4);
    vi.spyOn(editor.framePipeline, 'scheduleFrameLoad').mockImplementation(() => {});
    objectOperations.removeObject();
    objectOperations.addObject();
    expect(editor.store.objects().map((object) => object.id)).toEqual([1, 3, 5]);
  });

  it('cleans removed objects from all frame maps without mutating previous snapshots', () => {
    vi.spyOn(editor.framePipeline, 'scheduleFrameLoad').mockImplementation(() => {});
    const removeFromCanvas = vi.spyOn(editor.framePipeline, 'removeObject');
    const points = new Map([[5, new Map([[1, [{ x: 1, y: 2, label: 1 }]]])]]);
    const masks = new Map([[5, new Map([[1, [[true]]]])]]);
    const edited = new Map([[5, new Set([1])]]);
    editor.store.points.set(points);
    editor.store.masks.set(masks);
    editor.store.liveEditedObjectFrames.set(edited);
    objectOperations.removeObject();
    expect(editor.store.points().size).toBe(0);
    expect(editor.store.masks().size).toBe(0);
    expect(editor.store.liveEditedObjectFrames().size).toBe(0);
    expect(points.get(5)?.has(1)).toBe(true);
    expect(masks.get(5)?.has(1)).toBe(true);
    expect(edited.get(5)?.has(1)).toBe(true);
    expect(removeFromCanvas).toHaveBeenCalledWith(1);
  });

  it('keeps successful removals when removing all objects fails partway through', async () => {
    vi.spyOn(editor.framePipeline, 'scheduleFrameLoad').mockImplementation(() => {});
    objectOperations.addObject();
    backendMock.removeObject
      .mockReturnValueOnce(of({}))
      .mockReturnValueOnce(throwError(() => new Error('Failed')));
    await objectOperations.removeAllObjects();
    expect(editor.store.objects().map((object) => object.id)).toEqual([2]);
    expect(editor.store.isLoading()).toBe(false);
  });

  it('blocks point requests during object removal and unlocks after failure', async () => {
    const removal$ = new Subject<unknown>();
    backendMock.removeObject.mockReturnValue(removal$);
    objectOperations.removeObject();
    await editor.prompts.addPoint(10, 20, 1, 5);
    expect(backendMock.addNewPointsOrBox).not.toHaveBeenCalled();
    removal$.error(new Error('Removal failed'));
    expect(editor.store.isInteractionBusy()).toBe(false);
    expect(editor.store.objects()).toHaveLength(1);
  });
});
