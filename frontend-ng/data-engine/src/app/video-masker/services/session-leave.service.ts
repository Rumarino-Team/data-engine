import { Injectable, inject } from '@angular/core';
import { ConfirmService } from './confirm.service';
import { FramePipelineService } from './frame-pipeline.service';
import { SessionService } from './session.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

/** Guards actions that drop the open session, so unsaved working changes are never lost silently. */
@Injectable()
export class SessionLeaveService {
  private readonly store = inject(VideoMaskerStateStore);
  private readonly confirm = inject(ConfirmService);
  private readonly sessions = inject(SessionService);
  private readonly framePipeline = inject(FramePipelineService);

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

  /** Closes the editor session after confirmation; the backend keeps it until another one replaces it. */
  async closeSession(): Promise<boolean> {
    if (!this.store.isInitialized() || this.store.isInteractionBusy()) return false;
    if (!(await this.confirmLeave('close it'))) return false;
    if (this.store.isInteractionBusy()) return false;
    this.store.resetSession();
    this.framePipeline.clearFrameCaches();
    return true;
  }
}
