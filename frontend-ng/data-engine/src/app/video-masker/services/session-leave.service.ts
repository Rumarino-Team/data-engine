import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { BackendService } from '../../services/backend.service';
import { ConfirmService } from './confirm.service';
import { FramePipelineService } from './frame-pipeline.service';
import { SessionService } from './session.service';
import { ToastService } from './toast.service';
import { VideoMaskerStateStore } from './video-masker-state.store';
import { getErrorMessage } from '../video-masker.util';

/** Guards actions that drop the open session, so unsaved working changes are never lost silently. */
@Injectable()
export class SessionLeaveService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly confirm = inject(ConfirmService);
  private readonly sessions = inject(SessionService);
  private readonly framePipeline = inject(FramePipelineService);
  private readonly backend = inject(BackendService);
  private readonly toast = inject(ToastService);

  /**
   * Resolves true when the caller may drop the current session: nothing is unsaved, the user saved
   * it, or the user chose to discard. `reason` completes "Save them before you ...?".
   */
  async confirmLeave(reason: string): Promise<boolean> {
    if (!this.store.hasUnsavedChanges()) return true;
    const choice = await this.confirm.ask({
      title: 'Unsaved changes',
      message: `This session has changes that are not saved. Save them before you ${reason}?`,
      actions: [
        { value: 'save', label: 'Save', tone: 'primary' },
        { value: 'discard', label: 'Discard changes', tone: 'danger' },
      ],
    });
    if (choice === 'save') return this.sessions.save();
    return choice === 'discard';
  }

  /**
   * Closes the session after confirmation: the backend frees it (loaded video, annotations, working copy)
   * and only then does this tab drop it. If the backend refuses, the session stays open; if it cannot be
   * reached at all the tab still closes, since nothing here can be saved or edited without it.
   */
  async closeSession(): Promise<boolean> {
    if (!this.store.isInitialized() || this.store.isInteractionBusy()) return false;
    if (!(await this.confirmLeave('close it'))) return false;
    if (this.store.isInteractionBusy()) return false;
    const sessionId = this.store.sessionId();
    const revision = this.store.annotationRevision();
    if (sessionId && revision !== null) {
      const closed = await this.closeOnBackend(sessionId, revision);
      if (closed.outcome === 'refused') {
        // 409: the session changed elsewhere or a job is running. Reload what the backend holds.
        if (closed.conflict) this.store.annotationRecoveryRequested.set(true);
        return false;
      }
      if (closed.outcome === 'unreachable') {
        this.toast.show(
          'warning',
          'Backend unreachable',
          'The session was closed here only; the backend still holds it until it is replaced.',
        );
      }
    }
    this.store.resetSession();
    this.framePipeline.clearFrameCaches();
    return true;
  }

  /**
   * Closes a session the backend still holds that this tab is not showing (see `openBackendSession`).
   * Unsaved changes in it exist nowhere else, so discarding them takes an explicit answer.
   */
  async closeOpenBackendSession(): Promise<boolean> {
    const open = this.store.openBackendSession();
    if (!open || this.store.isInitialized() || this.store.isInteractionBusy()) return false;
    if (open.dirty) {
      const choice = await this.confirm.ask({
        title: 'Unsaved changes',
        message:
          'The session still open on the backend has changes that were never saved. Close it and lose them?',
        actions: [{ value: 'discard', label: 'Close and discard', tone: 'danger' }],
      });
      if (choice !== 'discard') return false;
      if (this.store.isInitialized() || this.store.isInteractionBusy()) return false;
    }
    const closed = await this.closeOnBackend(open.sessionId, open.revision);
    if (closed.outcome === 'closed') {
      this.store.openBackendSession.set(null);
      return true;
    }
    if (closed.outcome === 'unreachable') {
      this.toast.show(
        'warning',
        'Backend unreachable',
        'The session could not be closed because the backend did not answer.',
      );
    } else {
      // It changed meanwhile (another tab edited or replaced it): show what is there now.
      void this.sessions.checkForOpenSession();
    }
    return false;
  }

  private async closeOnBackend(
    sessionId: string,
    revision: number,
  ): Promise<
    { outcome: 'closed' } | { outcome: 'unreachable' } | { outcome: 'refused'; conflict: boolean }
  > {
    this.store.isLoading.set(true);
    try {
      await firstValueFrom(
        this.backend.closeVideoSession({ session_id: sessionId, expected_revision: revision }),
      );
      return { outcome: 'closed' };
    } catch (error: unknown) {
      console.error(error);
      const status = (error as { status?: number })?.status;
      if (status === 0) return { outcome: 'unreachable' };
      this.toast.show('error', 'Close failed', getErrorMessage(error, 'Unable to close the session.'));
      return { outcome: 'refused', conflict: status === 409 };
    } finally {
      this.store.isLoading.set(false);
    }
  }
}
