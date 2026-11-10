import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { describe, expect, it } from 'vitest';
import {
  createWorkflowFixture,
  makeInfo,
  makeResponse,
  objectUpdate,
} from '../testing/workflow-test-helpers';

type Fixture = ReturnType<typeof createWorkflowFixture>;
type Started = { pending: Subject<any>; success: unknown; done: Promise<unknown> };

/** Each workflow starts one backend call that stays pending and knows what the backend would answer. */
const operations: Record<string, (fixture: Fixture) => Started> = {
  'rename object': ({ backendMock, editor, objectOperations }) => {
    const pending = new Subject<any>();
    backendMock.updateObject.mockReturnValue(pending);
    objectOperations.renameObject(1, 'Late');
    const success = objectUpdate(editor.store, [{ id: 1, label: 'Late', color: '#ff0000' }]);
    return { pending, success, done: Promise.resolve() };
  },
  'create object': ({ backendMock, editor, objectOperations }) => {
    const pending = new Subject<any>();
    backendMock.createObject.mockReturnValue(pending);
    objectOperations.addObject();
    const created = { id: 7, label: 'Late', color: '#00ff00' };
    const success = {
      ...objectUpdate(editor.store, [...editor.store.objects(), created]),
      created_object_id: 7,
    };
    return { pending, success, done: Promise.resolve() };
  },
  'add point': ({ backendMock, editor }) => {
    const pending = new Subject<any>();
    backendMock.addNewPointsOrBox.mockReturnValue(pending);
    return { pending, success: makeResponse({}), done: editor.prompts.addPoint(10, 20, 1, 5) };
  },
  'reset masks': ({ backendMock, editor }) => {
    const pending = new Subject<any>();
    backendMock.resetVideoState.mockReturnValue(pending);
    editor.propagation.clearMasks();
    return {
      pending,
      success: objectUpdate(editor.store, editor.store.objects()),
      done: Promise.resolve(),
    };
  },
};

/** Ways the editor can move on while a request is in flight. */
const contexts: Record<string, (fixture: Fixture) => void> = {
  'the session is replaced': ({ backendMock, editor }) =>
    editor.store.applyAnnotationSnapshot(
      makeInfo({ session_id: 'replacement' }),
      backendMock.getApiUrl(),
      true,
    ),
  'the backend is switched': ({ backendMock }) =>
    backendMock.getApiUrl.mockReturnValue('http://other-backend'),
  'the editor is destroyed': () => TestBed.resetTestingModule(),
};

describe('late backend responses', () => {
  const cases = Object.keys(operations).flatMap((operation) =>
    Object.keys(contexts).flatMap((context) =>
      (['success', 'failure'] as const).map((outcome) => [operation, outcome, context] as const),
    ),
  );

  it.each(cases)('%s: a late %s is ignored once %s', async (operation, outcome, context) => {
    const fixture = createWorkflowFixture();
    const { editor } = fixture;
    const call = operations[operation](fixture);
    contexts[context](fixture);
    const accepted = editor.store.annotations();

    if (outcome === 'success') {
      call.pending.next(call.success);
      call.pending.complete();
    } else {
      call.pending.error({ status: 500 });
    }
    await call.done;
    await Promise.resolve();

    expect(editor.store.annotations()).toBe(accepted);
    expect(editor.store.annotationRecoveryRequested()).toBe(false);
    expect(editor.store.toasts()).toHaveLength(0);
    expect(editor.store.pendingPointEdit()).toBeNull();
    expect(editor.store.isInteractionBusy()).toBe(false);
  });
});
