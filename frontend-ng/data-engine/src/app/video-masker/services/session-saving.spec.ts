import { of, throwError } from 'rxjs';
import { beforeEach, describe, expect, it } from 'vitest';
import { SavedState, VideoSaveResponse } from '../../services/backend.service';
import {
  createWorkflowFixture,
  makeInfo,
  mockInitJob,
  updateWithSavedState,
} from '../testing/workflow-test-helpers';

const saved = (overrides: Partial<SavedState> = {}): SavedState => ({
  checkpoint_id: 'cp-1',
  display_name: 'Bedroom',
  saved_at: 'then',
  dirty: false,
  ...overrides,
});

const saveResponse = (overrides: Partial<VideoSaveResponse> = {}): VideoSaveResponse => ({
  message: 'Session saved successfully',
  name: 'Bedroom',
  checkpoint_id: 'cp-1',
  saved_path: 'C:/saved/bedroom-1234',
  saved_state: saved(),
  ...overrides,
});

describe('saving workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];
  let objectOperations: ReturnType<typeof createWorkflowFixture>['objectOperations'];

  beforeEach(() => {
    ({ editor, backendMock, objectOperations } = createWorkflowFixture());
    editor.store.isInitialized.set(true);
  });

  it('reports a never-saved session as new and clean until its first edit', () => {
    editor.store.savedState.set(saved({ checkpoint_id: null, display_name: null, saved_at: null }));
    expect(editor.store.saveStatus()).toBe('new');
    expect(editor.store.hasUnsavedChanges()).toBe(false);

    editor.store.savedState.set(saved({ checkpoint_id: null, dirty: true }));
    expect(editor.store.saveStatus()).toBe('unsaved');
    expect(editor.store.hasUnsavedChanges()).toBe(true);
  });

  it('follows the backend dirty flag through snapshots and mutation responses', () => {
    editor.store.applyAnnotationSnapshot(makeInfo({ saved_state: saved() }), true);
    expect(editor.store.saveStatus()).toBe('saved');
    expect(editor.store.hasUnsavedChanges()).toBe(false);

    backendMock.updateObject.mockImplementation((object) =>
      of(updateWithSavedState(editor.store, object, saved({ dirty: true }))),
    );
    objectOperations.renameObject(1, 'Left hand');

    expect(editor.store.saveStatus()).toBe('unsaved');
    expect(editor.store.hasUnsavedChanges()).toBe(true);
  });

  it('keeps the previous saved state when a response does not carry one', () => {
    editor.store.savedState.set(saved({ dirty: true }));
    objectOperations.renameObject(1, 'Left hand');
    expect(editor.store.savedState()).toEqual(saved({ dirty: true }));
  });

  it('forgets saved state when the session is reset', () => {
    editor.store.savedState.set(saved({ dirty: true }));
    editor.store.resetSession();
    expect(editor.store.savedState()).toBeNull();
    expect(editor.store.hasUnsavedChanges()).toBe(false);
  });

  it('requires a name for the first save and for Save As', async () => {
    editor.store.savedState.set(saved({ checkpoint_id: null, dirty: true }));
    editor.store.saveName.set('   ');
    await expect(editor.sessions.save()).resolves.toBe(false);

    editor.store.savedState.set(saved({ dirty: true }));
    editor.store.saveName.set('');
    await expect(editor.sessions.save(true)).resolves.toBe(false);

    expect(backendMock.saveVideoSession).not.toHaveBeenCalled();
    expect(editor.store.toasts()[0].title).toBe('Missing save name');
  });

  it.each([
    {
      case: 'a first save sends the typed name',
      state: saved({ checkpoint_id: null, display_name: null, dirty: true }),
      typed: '  Bedroom  ',
      saveAs: false,
      request: { name: 'Bedroom', saveAs: false },
    },
    {
      case: 'a re-save with an unchanged name sends no name',
      state: saved({ dirty: true }),
      typed: 'Bedroom',
      saveAs: false,
      request: { name: undefined, saveAs: false },
    },
    {
      case: 'a re-save with a new name renames',
      state: saved({ dirty: true }),
      typed: 'Kitchen',
      saveAs: false,
      request: { name: 'Kitchen', saveAs: false },
    },
    {
      case: 'Save As sends the name with save_as',
      state: saved(),
      typed: 'Bedroom copy',
      saveAs: true,
      request: { name: 'Bedroom copy', saveAs: true },
    },
  ])('$case and adopts the accepted checkpoint', async ({ state, typed, saveAs, request }) => {
    editor.store.savedState.set(state);
    editor.store.saveName.set(typed);
    const name = request.name ?? 'Bedroom';
    const response = saveResponse({
      name,
      checkpoint_id: saveAs ? 'cp-2' : 'cp-1',
      saved_state: saved({ checkpoint_id: saveAs ? 'cp-2' : 'cp-1', display_name: name }),
    });
    backendMock.saveVideoSession.mockReturnValue(of(response));

    await expect(editor.sessions.save(saveAs)).resolves.toBe(true);

    expect(backendMock.saveVideoSession.mock.calls[0][0]).toEqual(request);
    expect(editor.store.savedState()).toEqual(response.saved_state);
    expect(editor.store.saveName()).toBe(name);
  });

  it('keeps the unsaved state and reports a failed save', async () => {
    editor.store.savedState.set(saved({ dirty: true }));
    editor.store.saveName.set('Bedroom');
    backendMock.saveVideoSession.mockReturnValue(throwError(() => new Error('disk full')));

    await expect(editor.sessions.save()).resolves.toBe(false);

    expect(editor.store.savedState()).toEqual(saved({ dirty: true }));
    expect(editor.store.toasts()[0].title).toBe('Save failed');
    expect(editor.store.isLoading()).toBe(false);
  });

  it('blocks saving while another workflow is busy', async () => {
    editor.store.savedState.set(saved({ dirty: true }));
    editor.store.isLoading.set(true);
    await expect(editor.sessions.save()).resolves.toBe(false);
    expect(backendMock.saveVideoSession).not.toHaveBeenCalled();
  });

  it('adopts the saved name and state of a loaded session', async () => {
    mockInitJob(backendMock, { num_frames: 10, source_type: 'saved_session' });
    backendMock.getVideoInfo.mockReturnValue(of(makeInfo({ saved_state: saved() })));
    editor.store.isInitialized.set(false);
    editor.store.saveName.set('stale');
    editor.store.videoDir.set('C:/saved/bedroom');

    await editor.sessions.initVideo();

    expect(editor.store.isInitialized()).toBe(true);
    expect(editor.store.saveName()).toBe('Bedroom');
    expect(editor.store.saveStatus()).toBe('saved');
  });
});
