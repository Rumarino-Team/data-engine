import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { describe, expect, it } from 'vitest';
import { Fixture, editOperations } from '../testing/edit-operations';
import {
  createWorkflowFixture,
  makeInfo,
  mockJob,
  setAcceptedPoints,
} from '../testing/workflow-test-helpers';

/**
 * The editing contract between the frontend and the backend: the editor refuses to start work while
 * busy, applies the backend's answer, and recovers from the backend's snapshot when it cannot.
 */
describe('edit protocol', () => {
  describe('while the editor is busy, nothing is sent', () => {
    const entries: [string, (f: Fixture) => unknown, (f: Fixture) => { mock: unknown }][] = [
      [
        'rename object',
        (f) => f.objectOperations.renameObject(1, 'X'),
        (f) => f.backendMock.updateObject,
      ],
      ['add object', (f) => f.objectOperations.addObject(), (f) => f.backendMock.createObject],
      [
        'remove object',
        (f) => f.objectOperations.removeObject(),
        (f) => f.backendMock.removeObject,
      ],
      [
        'remove all objects',
        (f) => f.objectOperations.removeAllObjects(),
        (f) => f.backendMock.removeObject,
      ],
      [
        'add point',
        (f) => f.editor.prompts.addPoint(10, 20, 1, 5),
        (f) => f.backendMock.addNewPointsOrBox,
      ],
      [
        'remove point',
        (f) => {
          setAcceptedPoints(
            f.editor.store,
            new Map([[5, new Map([[1, [{ x: 1, y: 2, label: 1 }]]])]]),
          );
          return f.editor.prompts.removePoint(1, 5, 0);
        },
        (f) => f.backendMock.clearAllPromptsInFrame,
      ],
      [
        'clear masks',
        (f) => f.editor.propagation.clearMasks(),
        (f) => f.backendMock.resetVideoState,
      ],
      ['propagate', (f) => f.editor.propagation.propagate(), (f) => f.backendMock.propagateInVideo],
      [
        'track points',
        (f) => f.editor.tracking.runTracking(),
        (f) => f.backendMock.trackPromptPoints,
      ],
      [
        'save',
        (f) => {
          f.editor.store.savedState.set({
            checkpoint_id: 'cp-1',
            display_name: 'Bedroom',
            saved_at: 'then',
            dirty: true,
          });
          return f.editor.sessions.save();
        },
        (f) => f.backendMock.saveVideoSession,
      ],
      [
        'load video',
        (f) => {
          f.editor.store.videoDir.set('C:/other');
          return f.editor.sessions.initVideo();
        },
        (f) => f.backendMock.initVideoState,
      ],
      [
        'refresh annotations',
        (f) => f.editor.sessions.refreshAnnotations(),
        (f) => f.backendMock.getVideoInfo,
      ],
      [
        'switch backend',
        (f) => {
          f.editor.store.apiUrlInput.set('http://other-backend');
          return f.connection.applyApiUrl();
        },
        (f) => f.backendMock.setApiUrl,
      ],
    ];
    const reasons: [string, (f: Fixture) => void][] = [
      ['a job or load is running', (f) => f.editor.store.isLoading.set(true)],
      ['a point request is in flight', (f) => f.editor.store.isPointRequestInFlight.set(true)],
      ['an object edit is pending', (f) => f.editor.store.pendingObjectUpdates.set(1)],
      ['the snapshot is being recovered', (f) => f.editor.store.isRecoveringAnnotations.set(true)],
    ];
    const cases = entries.flatMap(([name, start, backend]) =>
      reasons.map(([reason, busy]) => [name, reason, start, backend, busy] as const),
    );

    it.each(cases)('%s is refused while %s', async (_name, _reason, start, backend, busy) => {
      const fixture = createWorkflowFixture();
      fixture.editor.store.isInitialized.set(true);
      busy(fixture);
      await start(fixture);
      expect(backend(fixture).mock).toMatchObject({ calls: [] });
      expect(fixture.editor.store.sessionId()).toBe('session-1');
    });
  });

  describe('when a job fails', () => {
    const failed = {
      status: 'failed',
      error: { code: 'backend_error', message: 'Job failed', detail: null },
    };
    const starts: [string, (f: Fixture) => Promise<unknown>][] = [
      [
        'propagation',
        (f) => {
          mockJob(f.backendMock.propagateInVideo, f.backendMock.getJob, 'mask_propagation', failed);
          return f.editor.propagation.propagate();
        },
      ],
      [
        'tracking',
        (f) => {
          mockJob(f.backendMock.trackPromptPoints, f.backendMock.getJob, 'prompt_tracking', failed);
          return f.editor.tracking.runTracking();
        },
      ],
      [
        'loading a replacement video',
        (f) => {
          mockJob(f.backendMock.initVideoState, f.backendMock.getJob, 'video_init', failed);
          f.editor.store.videoDir.set('C:/replacement');
          return f.editor.sessions.initVideo();
        },
      ],
    ];

    it.each(starts)('%s reports it and asks for a snapshot refresh', async (_name, start) => {
      const fixture = createWorkflowFixture();
      fixture.editor.store.isInitialized.set(true);
      await start(fixture);
      expect(fixture.editor.store.annotationRecoveryRequested()).toBe(true);
      expect(fixture.editor.store.toasts()).toHaveLength(1);
      expect(fixture.editor.store.isInteractionBusy()).toBe(false);
    });
  });

  it('waits until the editor is idle before refreshing the snapshot', () => {
    const fixture = createWorkflowFixture();
    const { store } = fixture.editor;
    store.isInitialized.set(true);
    fixture.backendMock.getVideoInfo.mockReturnValue(of(makeInfo()));
    store.isPointRequestInFlight.set(true);
    store.annotationRecoveryRequested.set(true);
    TestBed.tick();
    expect(fixture.backendMock.getVideoInfo).not.toHaveBeenCalled();
    store.isPointRequestInFlight.set(false);
    TestBed.tick();
    expect(fixture.backendMock.getVideoInfo).toHaveBeenCalledTimes(1);
    expect(store.annotationRecoveryRequested()).toBe(false);
  });

  describe('when the backend cannot answer', () => {
    const operations = Object.keys(editOperations);

    it.each(operations.flatMap((name) => [0, 409, 500].map((status) => [name, status] as const)))(
      '%s recovers from the backend snapshot after status %s',
      async (name, status) => {
        const fixture = createWorkflowFixture();
        const call = editOperations[name](fixture);
        const accepted = fixture.editor.store.annotations();
        call.pending.error({ status });
        await call.done;
        expect(fixture.editor.store.annotationRecoveryRequested()).toBe(true);
        expect(fixture.editor.store.annotations()).toBe(accepted);
        expect(fixture.editor.store.toasts()).toHaveLength(1);
        expect(fixture.editor.store.isInteractionBusy()).toBe(false);
      },
    );

    it.each(operations.flatMap((name) => [400, 422].map((status) => [name, status] as const)))(
      '%s reports a rejected request (status %s) without recovery',
      async (name, status) => {
        const fixture = createWorkflowFixture();
        const call = editOperations[name](fixture);
        const accepted = fixture.editor.store.annotations();
        call.pending.error({ status });
        await call.done;
        expect(fixture.editor.store.annotationRecoveryRequested()).toBe(false);
        expect(fixture.editor.store.annotations()).toBe(accepted);
        expect(fixture.editor.store.toasts()).toHaveLength(1);
        expect(fixture.editor.store.isInteractionBusy()).toBe(false);
      },
    );

    it.each(operations)('%s recovers when the returned delta does not apply', async (name) => {
      const fixture = createWorkflowFixture();
      const call = editOperations[name](fixture);
      const accepted = fixture.editor.store.annotations();
      const delta = { ...call.success.annotation_delta, base_revision: 99 };
      call.pending.next({ ...call.success, annotation_delta: delta });
      call.pending.complete();
      await call.done;
      expect(fixture.editor.store.annotationRecoveryRequested()).toBe(true);
      expect(fixture.editor.store.annotations()).toBe(accepted);
      expect(fixture.editor.store.isInteractionBusy()).toBe(false);
    });

    it.each(operations)('%s does nothing once the editor is destroyed', async (name) => {
      const fixture = createWorkflowFixture();
      const call = editOperations[name](fixture);
      const accepted = fixture.editor.store.annotations();
      TestBed.resetTestingModule();
      call.pending.next(call.success);
      call.pending.complete();
      await call.done;
      expect(fixture.editor.store.annotations()).toBe(accepted);
      expect(fixture.editor.store.annotationRecoveryRequested()).toBe(false);
      expect(fixture.editor.store.toasts()).toHaveLength(0);
    });
  });
});
