import { of, throwError } from 'rxjs';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { SavedState } from '../../services/backend.service';
import { createWorkflowFixture } from '../testing/workflow-test-helpers';

const saved = (overrides: Partial<SavedState> = {}): SavedState => ({
  checkpoint_id: 'cp-1',
  display_name: 'Bedroom',
  saved_at: 'then',
  dirty: true,
  ...overrides,
});

describe('SessionLeaveService', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];
  let confirm: ReturnType<typeof createWorkflowFixture>['confirm'];
  let leave: ReturnType<typeof createWorkflowFixture>['leave'];

  beforeEach(() => {
    ({ editor, backendMock, confirm, leave } = createWorkflowFixture());
    editor.store.isInitialized.set(true);
    editor.store.savedState.set(saved());
    editor.store.saveName.set('Bedroom');
  });

  it('lets a clean session go without asking', async () => {
    editor.store.savedState.set(saved({ dirty: false }));
    await expect(leave.confirmLeave('close it')).resolves.toBe(true);
    expect(confirm.request()).toBeNull();
  });

  it('asks about unsaved changes with save, discard and cancel choices', async () => {
    const pending = leave.confirmLeave('close it');
    const request = confirm.request();
    expect(request?.title).toBe('Unsaved changes');
    expect(request?.message).toContain('before you close it');
    expect(request?.actions.map((action) => action.value)).toEqual(['save', 'discard']);
    confirm.answer(null);
    await expect(pending).resolves.toBe(false);
  });

  it('saves first when asked to, then lets the session go', async () => {
    backendMock.saveVideoSession.mockReturnValue(
      of({
        message: 'ok',
        name: 'Bedroom',
        checkpoint_id: 'cp-1',
        saved_path: 'C:/saved/bedroom',
        state_epoch: 3,
        saved_state: saved({ dirty: false }),
      }),
    );
    const pending = leave.confirmLeave('close it');
    confirm.answer('save');
    await expect(pending).resolves.toBe(true);
    expect(backendMock.saveVideoSession).toHaveBeenCalledTimes(1);
    expect(editor.store.hasUnsavedChanges()).toBe(false);
  });

  it('keeps the session when the requested save fails', async () => {
    backendMock.saveVideoSession.mockReturnValue(throwError(() => new Error('disk full')));
    const pending = leave.confirmLeave('close it');
    confirm.answer('save');
    await expect(pending).resolves.toBe(false);
    expect(editor.store.hasUnsavedChanges()).toBe(true);
  });

  it('keeps the session when a first save has no name', async () => {
    editor.store.savedState.set(saved({ checkpoint_id: null }));
    editor.store.saveName.set('');
    const pending = leave.confirmLeave('close it');
    confirm.answer('save');
    await expect(pending).resolves.toBe(false);
    expect(backendMock.saveVideoSession).not.toHaveBeenCalled();
  });

  it('discards on request without saving', async () => {
    const pending = leave.confirmLeave('close it');
    confirm.answer('discard');
    await expect(pending).resolves.toBe(true);
    expect(backendMock.saveVideoSession).not.toHaveBeenCalled();
  });

  it('closes a discarded session and clears editor state and caches', async () => {
    const clear = vi.spyOn(editor.framePipeline, 'clearFrameCaches');
    const pending = leave.closeSession();
    confirm.answer('discard');
    await expect(pending).resolves.toBe(true);
    expect(editor.store.isInitialized()).toBe(false);
    expect(editor.store.sessionId()).toBeNull();
    expect(editor.store.savedState()).toBeNull();
    expect(clear).toHaveBeenCalled();
  });

  it('closes a clean session immediately', async () => {
    editor.store.savedState.set(saved({ dirty: false }));
    await expect(leave.closeSession()).resolves.toBe(true);
    expect(editor.store.isInitialized()).toBe(false);
    expect(confirm.request()).toBeNull();
  });

  it('keeps the session open when closing is cancelled', async () => {
    const pending = leave.closeSession();
    confirm.answer(null);
    await expect(pending).resolves.toBe(false);
    expect(editor.store.isInitialized()).toBe(true);
    expect(editor.store.sessionId()).toBe('session-1');
  });

  it('refuses to close while work is in flight or nothing is open', async () => {
    editor.store.isLoading.set(true);
    await expect(leave.closeSession()).resolves.toBe(false);
    expect(confirm.request()).toBeNull();
    editor.store.isLoading.set(false);
    editor.store.isInitialized.set(false);
    await expect(leave.closeSession()).resolves.toBe(false);
  });

  it('does not close if work starts while the question is open', async () => {
    const pending = leave.closeSession();
    editor.store.isLoading.set(true);
    confirm.answer('discard');
    await expect(pending).resolves.toBe(false);
    expect(editor.store.sessionId()).toBe('session-1');
  });
});
