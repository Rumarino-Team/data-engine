import { Subject } from 'rxjs';
import { AnnotationDelta } from '../../services/backend.service';
import { createWorkflowFixture, makeResponse, objectUpdate } from './workflow-test-helpers';

export type Fixture = ReturnType<typeof createWorkflowFixture>;
export type StartedEdit = {
  /** Resolves or rejects the backend call the workflow is waiting on. */
  pending: Subject<any>;
  /** What the backend would answer if the call succeeded. */
  success: { annotation_delta: AnnotationDelta };
  done: Promise<unknown>;
};

const pendingCall = () => new Subject<any>();

/** Each editing workflow starts one backend call that stays pending. */
export const editOperations: Record<string, (fixture: Fixture) => StartedEdit> = {
  'rename object': ({ backendMock, editor, objectOperations }) => {
    const pending = pendingCall();
    backendMock.updateObject.mockReturnValue(pending);
    objectOperations.renameObject(1, 'Late');
    const success = objectUpdate(editor.store, [{ id: 1, label: 'Late', color: '#ff0000' }]);
    return { pending, success, done: Promise.resolve() };
  },
  'create object': ({ backendMock, editor, objectOperations }) => {
    const pending = pendingCall();
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
    const pending = pendingCall();
    backendMock.addNewPointsOrBox.mockReturnValue(pending);
    return { pending, success: makeResponse({}), done: editor.prompts.addPoint(10, 20, 1, 5) };
  },
  'reset masks': ({ backendMock, editor }) => {
    const pending = pendingCall();
    backendMock.resetVideoState.mockReturnValue(pending);
    editor.propagation.clearMasks();
    return {
      pending,
      success: objectUpdate(editor.store, editor.store.objects()),
      done: Promise.resolve(),
    };
  },
};
