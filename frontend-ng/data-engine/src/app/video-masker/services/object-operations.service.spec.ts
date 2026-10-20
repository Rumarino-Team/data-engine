import { cachedMasks, seedMasks } from '../testing/workflow-test-helpers';
import { setAcceptedPoints } from '../testing/workflow-test-helpers';
import { describe, beforeEach, expect, it, vi } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import {
  createWorkflowFixture,
  objectSnapshot,
  makeResponse,
} from '../testing/workflow-test-helpers';
import { AnnotationSnapshot, ObjectCreationResponse } from '../../services/backend.service';

describe('object-operations workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];
  let objectOperations: ReturnType<typeof createWorkflowFixture>['objectOperations'];

  beforeEach(() => {
    ({ editor, backendMock, objectOperations } = createWorkflowFixture());
  });

  it('uses each accepted revision when removing multiple objects', async () => {
    objectOperations.addObject();
    await objectOperations.removeAllObjects();
    expect(backendMock.removeObject.mock.calls).toEqual([
      [1, { session_id: 'session-1', expected_revision: 1 }],
      [2, { session_id: 'session-1', expected_revision: 2 }],
    ]);
  });

  it.each([409, 0, 500])('recovers after status %s without replaying creation', async (status) => {
    editor.store.isInitialized.set(true);
    const recovered = objectSnapshot(editor.store, [
      ...editor.store.objects(),
      { id: 9, label: 'Accepted elsewhere', color: '#123456' },
    ]);
    backendMock.getVideoInfo.mockReturnValue(of({ ...recovered, num_frames: 10, frame_files: [] }));
    backendMock.createObject.mockReturnValue(throwError(() => ({ status })));
    objectOperations.addObject();
    TestBed.tick();
    await Promise.resolve();
    expect(backendMock.getVideoInfo).toHaveBeenCalledTimes(1);
    expect(backendMock.createObject).toHaveBeenCalledTimes(1);
    expect(editor.store.objects().map((object) => object.id)).toEqual([1, 9]);
    expect(editor.store.isRecoveringAnnotations()).toBe(false);
  });

  it('blocks editing during conflict recovery and requires reload for a replacement session', async () => {
    editor.store.isInitialized.set(true);
    const recovery = new Subject<
      ReturnType<typeof objectSnapshot> & { num_frames: number; frame_files: string[] }
    >();
    backendMock.getVideoInfo.mockReturnValue(recovery);
    backendMock.createObject.mockReturnValue(throwError(() => ({ status: 409 })));
    objectOperations.addObject();
    TestBed.tick();
    expect(editor.store.isInteractionBusy()).toBe(true);
    objectOperations.addObject();
    expect(backendMock.createObject).toHaveBeenCalledTimes(1);
    recovery.next({
      ...objectSnapshot(editor.store, []),
      session_id: 'replacement',
      num_frames: 10,
      frame_files: [],
    });
    await Promise.resolve();
    expect(editor.store.isInitialized()).toBe(false);
    expect(editor.store.sessionId()).toBe('session-1');
    expect(editor.store.isRecoveringAnnotations()).toBe(false);
  });

  it('does not refresh another backend after an old request fails', () => {
    editor.store.isInitialized.set(true);
    const pending = new Subject<ObjectCreationResponse>();
    backendMock.createObject.mockReturnValue(pending);
    objectOperations.addObject();
    backendMock.getApiUrl.mockReturnValue('http://replacement');
    pending.error({ status: 0 });
    TestBed.tick();
    expect(backendMock.getVideoInfo).not.toHaveBeenCalled();
  });

  it('uses the revision returned by point acceptance for the next object edit', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({})));
    await editor.prompts.addPoint(10, 20, 1, 5);
    objectOperations.renameObject(1, 'Hand');
    expect(backendMock.updateObject).toHaveBeenCalledWith(
      { id: 1, label: 'Hand', color: '#ff0000' },
      { session_id: 'session-1', expected_revision: 1 },
    );
    expect(editor.store.points().get(5)?.get(1)).toEqual([{ x: 10, y: 20, label: 1 }]);
    expect(backendMock.getVideoInfo).not.toHaveBeenCalled();
  });

  it.each(['clear point', 'reset'])('uses the accepted revision after %s', async (operation) => {
    backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({})));
    await editor.prompts.addPoint(10, 20, 1, 5);
    if (operation === 'reset') editor.propagation.clearMasks();
    else await editor.prompts.removePoint(1, 5, 0);
    objectOperations.addObject();
    expect(backendMock.createObject).toHaveBeenCalledWith({
      session_id: 'session-1',
      expected_revision: 2,
    });
    expect(editor.store.points().size).toBe(0);
    expect(backendMock.getVideoInfo).not.toHaveBeenCalled();
  });

  it('uses the server allocated ID even when it differs from cached allocation metadata', () => {
    const created = { id: 42, label: 'Server object', color: '#123456' };
    backendMock.createObject.mockReturnValue(
      of({
        ...objectSnapshot(editor.store, [...editor.store.objects(), created]),
        created_object_id: 42,
      }),
    );
    objectOperations.addObject();
    expect(backendMock.createObject).toHaveBeenCalledWith({
      session_id: 'session-1',
      expected_revision: 0,
    });
    expect(editor.store.selectedObjectId()).toBe(42);
    expect(editor.store.objects().at(-1)).toEqual(created);
    expect(backendMock.updateObject).not.toHaveBeenCalled();
  });

  it('applies accepted relabels and leaves accepted state unchanged on rejection', () => {
    objectOperations.renameObject(1, '  buoy ');
    expect(backendMock.updateObject).toHaveBeenLastCalledWith(
      {
        id: 1,
        label: 'buoy',
        color: '#ff0000',
      },
      expect.objectContaining({ session_id: 'session-1' }),
    );
    expect(editor.store.objects()[0].label).toBe('buoy');

    backendMock.updateObject.mockReturnValueOnce(throwError(() => new Error('nope')));
    objectOperations.renameObject(1, 'gate');
    expect(editor.store.objects()[0].label).toBe('buoy');
    expect(editor.store.toasts()[0].title).toBe('Object update failed');

    backendMock.updateObject.mockClear();
    objectOperations.renameObject(1, 'buoy');
    expect(backendMock.updateObject).not.toHaveBeenCalled();
  });

  it('requests backend object creation', () => {
    objectOperations.addObject();
    expect(backendMock.createObject).toHaveBeenCalledWith({
      session_id: 'session-1',
      expected_revision: 0,
    });
  });

  it('cleans removed objects from all frame maps without mutating previous snapshots', () => {
    vi.spyOn(editor.framePipeline, 'scheduleFrameLoad').mockImplementation(() => {});
    const removeFromCanvas = vi.spyOn(editor.framePipeline, 'clearMaskDataCache');
    const points = new Map([[5, new Map([[1, [{ x: 1, y: 2, label: 1 }]]])]]);
    const masks = new Map([[5, new Map([[1, [[true]]]])]]);
    setAcceptedPoints(editor.store, points);
    seedMasks(editor, masks);
    objectOperations.removeObject();
    expect(editor.store.points().size).toBe(0);
    expect(cachedMasks(editor.framePipeline).size).toBe(0);
    expect(points.get(5)?.has(1)).toBe(true);
    expect(masks.get(5)?.has(1)).toBe(true);
    expect(removeFromCanvas).toHaveBeenCalled();
  });

  it('keeps successful removals when removing all objects fails partway through', async () => {
    vi.spyOn(editor.framePipeline, 'scheduleFrameLoad').mockImplementation(() => {});
    objectOperations.addObject();
    backendMock.removeObject
      .mockReturnValueOnce(
        of(
          objectSnapshot(
            editor.store,
            editor.store.objects().filter((object) => object.id !== 1),
          ),
        ),
      )
      .mockReturnValueOnce(throwError(() => new Error('Failed')));
    await objectOperations.removeAllObjects();
    expect(editor.store.objects().map((object) => object.id)).toEqual([2]);
    expect(editor.store.isLoading()).toBe(false);
  });

  it('blocks point requests during object removal and unlocks after failure', async () => {
    const removal$ = new Subject<AnnotationSnapshot>();
    backendMock.removeObject.mockReturnValue(removal$);
    objectOperations.removeObject();
    await editor.prompts.addPoint(10, 20, 1, 5);
    expect(backendMock.addNewPointsOrBox).not.toHaveBeenCalled();
    removal$.error(new Error('Removal failed'));
    expect(editor.store.isInteractionBusy()).toBe(false);
    expect(editor.store.objects()).toHaveLength(1);
  });

  it('waits for accepted creation and uses the returned state without another GET', async () => {
    const pending = new Subject<ObjectCreationResponse>();
    backendMock.createObject.mockReturnValue(pending);
    objectOperations.addObject();
    expect(editor.store.objects()).toHaveLength(1);
    expect(editor.store.selectedObjectId()).toBe(1);
    expect(editor.store.isInteractionBusy()).toBe(true);
    await editor.prompts.addPoint(10, 20, 1, 5);
    expect(backendMock.addNewPointsOrBox).not.toHaveBeenCalled();
    const accepted = { id: 27, label: 'Accepted object', color: '#123456' };
    pending.next({
      ...objectSnapshot(editor.store, [...editor.store.objects(), accepted]),
      created_object_id: 27,
    });
    expect(editor.store.objects().find((object) => object.id === accepted.id)?.label).toBe(
      'Accepted object',
    );
    expect(editor.store.selectedObjectId()).toBe(accepted.id);
    expect(editor.store.annotationRevision()).toBe(1);
    expect(editor.store.isInteractionBusy()).toBe(false);
    expect(backendMock.getVideoInfo).not.toHaveBeenCalled();
  });

  it('does not leave a phantom object when creation fails', () => {
    const previous = editor.store.objects();
    const accepted = editor.store.annotations();
    backendMock.createObject.mockReturnValue(throwError(() => new Error('rejected')));
    objectOperations.addObject();
    expect(editor.store.objects()).toBe(previous);
    expect(editor.store.annotations()).toBe(accepted);
    expect(editor.store.selectedObjectId()).toBe(1);
    expect(editor.store.pendingObjectUpdates()).toBe(0);
  });

  it('keeps the accepted label while a rename is pending and blocks overlapping object edits', () => {
    const pending = new Subject<AnnotationSnapshot>();
    backendMock.updateObject.mockReturnValue(pending);
    const previous = editor.store.objects();
    objectOperations.renameObject(1, 'Draft');
    expect(editor.store.objects()).toBe(previous);
    objectOperations.renameObject(1, 'Later');
    expect(backendMock.updateObject).toHaveBeenCalledTimes(1);
    pending.error(new Error('rejected'));
    expect(editor.store.objects()).toBe(previous);
  });

  it('ignores an older accepted response without rolling back a newer snapshot', () => {
    const pending = new Subject<AnnotationSnapshot>();
    backendMock.updateObject.mockReturnValue(pending);
    const older = objectSnapshot(editor.store, [{ id: 1, label: 'Older', color: '#ff0000' }]);
    objectOperations.renameObject(1, 'Older');
    const newer = objectSnapshot(editor.store, [{ id: 1, label: 'Newer', color: '#ff0000' }]);
    newer.annotations!.revision = 10;
    editor.store.applyAnnotationSnapshot(newer, backendMock.getApiUrl());
    pending.next(older);
    expect(editor.store.objects()[0].label).toBe('Newer');
    expect(editor.store.annotationRevision()).toBe(10);
    expect(editor.store.isInteractionBusy()).toBe(false);
  });

  it('does not apply a response after the backend URL changes', () => {
    const pending = new Subject<AnnotationSnapshot>();
    backendMock.updateObject.mockReturnValue(pending);
    const response = objectSnapshot(editor.store, [
      { id: 1, label: 'Old backend', color: '#ff0000' },
    ]);
    objectOperations.renameObject(1, 'Old backend');
    backendMock.getApiUrl.mockReturnValue('http://other-backend');
    pending.next(response);
    expect(editor.store.objects()[0].label).toBe('Object 1');
    expect(editor.store.annotationRevision()).toBe(0);
  });

  it('does not apply a response to a replacement session', () => {
    const pending = new Subject<AnnotationSnapshot>();
    backendMock.updateObject.mockReturnValue(pending);
    const response = objectSnapshot(editor.store, [
      { id: 1, label: 'Old session', color: '#ff0000' },
    ]);
    objectOperations.renameObject(1, 'Old session');
    const replacement = objectSnapshot(editor.store, [
      { id: 1, label: 'Replacement', color: '#00ff00' },
    ]);
    replacement.session_id = 'replacement-session';
    editor.store.applyAnnotationSnapshot(replacement, backendMock.getApiUrl(), true);
    pending.next(response);
    expect(editor.store.sessionId()).toBe('replacement-session');
    expect(editor.store.objects()[0].label).toBe('Replacement');
  });

  it('takes remaining prompts from the deletion response instead of cleaning local point maps', () => {
    objectOperations.addObject();
    editor.store.selectedObjectId.set(1);
    const accepted = objectSnapshot(
      editor.store,
      editor.store.objects().filter((object) => object.id !== 1),
    );
    accepted.annotations!.prompts = [
      { frame_idx: 9, obj_id: 2, points: [[3, 4]], labels: [1], clear_old_points: true, box: null },
    ];
    backendMock.removeObject.mockReturnValue(of(accepted));
    objectOperations.removeObject();
    expect(editor.store.points().get(9)?.get(2)).toEqual([{ x: 3, y: 4, label: 1 }]);
    expect(editor.store.selectedObjectId()).toBe(2);
    expect(editor.store.annotations()).toEqual(accepted.annotations);
    expect(backendMock.getVideoInfo).not.toHaveBeenCalled();
  });
});
