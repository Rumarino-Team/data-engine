import { Subject, of, throwError } from 'rxjs';
import { beforeEach, describe, expect, it } from 'vitest';
import { SavedState, VideoSaveResponse } from '../../services/backend.service';
import {
  createWorkflowFixture,
  makeInfo,
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
  state_epoch: 3,
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
    editor.store.applyAnnotationSnapshot(
      makeInfo({ saved_state: saved() }),
      'http://127.0.0.1:8000',
      true,
    );
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

  it('sends the typed name on the first save and records the checkpoint', async () => {
    editor.store.savedState.set(saved({ checkpoint_id: null, display_name: null, dirty: true }));
    editor.store.saveName.set('  Bedroom  ');
    backendMock.saveVideoSession.mockReturnValue(of(saveResponse()));

    await expect(editor.sessions.save()).resolves.toBe(true);

    expect(backendMock.saveVideoSession.mock.calls[0][0]).toEqual({
      name: 'Bedroom',
      saveAs: false,
    });
    expect(editor.store.savedState()).toEqual(saved());
    expect(editor.store.saveStatus()).toBe('saved');
    expect(editor.store.saveName()).toBe('Bedroom');
  });

  it('re-saves without a name and renames only when the name changed', async () => {
    editor.store.savedState.set(saved({ dirty: true }));
    editor.store.saveName.set('Bedroom');
    backendMock.saveVideoSession.mockReturnValue(of(saveResponse()));

    await editor.sessions.save();
    expect(backendMock.saveVideoSession.mock.calls[0][0]).toEqual({
      name: undefined,
      saveAs: false,
    });

    editor.store.saveName.set('Kitchen');
    backendMock.saveVideoSession.mockReturnValue(
      of(saveResponse({ name: 'Kitchen', saved_state: saved({ display_name: 'Kitchen' }) })),
    );
    await editor.sessions.save();
    expect(backendMock.saveVideoSession.mock.calls[1][0]).toEqual({
      name: 'Kitchen',
      saveAs: false,
    });
    expect(editor.store.saveName()).toBe('Kitchen');
  });

  it('Save As sends the name with save_as and adopts the new checkpoint', async () => {
    editor.store.savedState.set(saved());
    editor.store.saveName.set('Bedroom copy');
    backendMock.saveVideoSession.mockReturnValue(
      of(
        saveResponse({
          name: 'Bedroom copy',
          checkpoint_id: 'cp-2',
          saved_state: saved({ checkpoint_id: 'cp-2', display_name: 'Bedroom copy' }),
        }),
      ),
    );

    await editor.sessions.save(true);

    expect(backendMock.saveVideoSession.mock.calls[0][0]).toEqual({
      name: 'Bedroom copy',
      saveAs: true,
    });
    expect(editor.store.savedState()?.checkpoint_id).toBe('cp-2');
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

  it('does not apply a save response to a session that was replaced meanwhile', async () => {
    editor.store.savedState.set(saved({ dirty: true }));
    editor.store.saveName.set('Bedroom');
    const response = new Subject<VideoSaveResponse>();
    backendMock.saveVideoSession.mockReturnValue(response);

    const pending = editor.sessions.save();
    editor.store.sessionId.set('replacement');
    editor.store.savedState.set(saved({ checkpoint_id: null, dirty: false }));
    response.next(saveResponse());
    response.complete();
    await pending;

    expect(editor.store.savedState()?.checkpoint_id).toBeNull();
    expect(editor.store.toasts()).toHaveLength(0);
  });

  it('blocks saving while another workflow is busy', async () => {
    editor.store.savedState.set(saved({ dirty: true }));
    editor.store.isLoading.set(true);
    await expect(editor.sessions.save()).resolves.toBe(false);
    expect(backendMock.saveVideoSession).not.toHaveBeenCalled();
  });

  it('adopts the saved name and state of a loaded session', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({ job_id: 'job-1', status: 'queued', operation: 'video_init', message: 'queued' }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-1',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 10,
          total: 10,
          window_index: null,
          window_count: null,
          frame_idx: null,
          stage_history: [],
          message: 'done',
          error: null,
          started_at: 'now',
          updated_at: 'now',
          completed_at: 'now',
          result: {
            message: 'Video state initialized successfully',
            num_frames: 10,
            resolved_video_frames_dir: 'C:/work/frames',
            online_mode: true,
            batch_size: 1,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 3,
            source_type: 'saved_session',
          },
        },
      }),
    );
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
