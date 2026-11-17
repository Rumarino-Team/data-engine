import { describe, beforeEach, expect, it, vi } from 'vitest';
import { createVideoMaskerFixture } from './testing/component-test-helpers';

describe('VideoMaskerComponent UI', () => {
  let component: Awaited<ReturnType<typeof createVideoMaskerFixture>>['component'];
  let fixture: Awaited<ReturnType<typeof createVideoMaskerFixture>>['fixture'];
  let desktopBridgeMock: Awaited<ReturnType<typeof createVideoMaskerFixture>>['desktopBridgeMock'];

  beforeEach(async () => {
    ({ component, fixture, desktopBridgeMock } = await createVideoMaskerFixture());
  });

  it('submits image coordinates from the viewport on the displayed frame', () => {
    component.store.isInitialized.set(true);
    component.store.targetFrameIdx.set(6);
    component.store.interactionMode.set('negative');
    vi.spyOn(component.framePipeline, 'currentBaseImage', 'get').mockReturnValue(new Image());
    const addPoint = vi.spyOn(component.prompts, 'addPoint').mockResolvedValue();
    component.onCanvasPoint({ x: 200, y: 100 });
    expect(addPoint).toHaveBeenCalledWith(200, 100, 0, 5);
    component.store.isPointRequestInFlight.set(true);
    component.onCanvasPoint({ x: 300, y: 100 });
    expect(addPoint).toHaveBeenCalledTimes(1);
  });

  it('offers the three frame loading modes and falls back to lazy on anything else', () => {
    expect(component.store.frameLoadingMode()).toBe('lazy');

    component.onFrameLoadingChange('preload');
    expect(component.store.frameLoadingMode()).toBe('preload');
    component.onFrameLoadingChange('eager');
    expect(component.store.frameLoadingMode()).toBe('eager');
    component.onFrameLoadingChange('streaming');
    expect(component.store.frameLoadingMode()).toBe('lazy');
  });

  describe('the notice about a session the backend still holds', () => {
    const banner = (): HTMLElement | null =>
      fixture.nativeElement.querySelector('.open-session-banner');
    const buttons = (): string[] =>
      Array.from(banner()?.querySelectorAll('button') ?? []).map((b) => b.textContent!.trim());
    const offer = (overrides: Partial<{ dirty: boolean; resumable: boolean }> = {}) =>
      component.store.openBackendSession.set({
        sessionId: 'held-1',
        revision: 4,
        numFrames: 924,
        dirty: false,
        resumable: true,
        ...overrides,
      });

    it('says what is open and offers to resume it or close it', () => {
      offer({ dirty: true });
      fixture.detectChanges();
      expect(banner()?.textContent).toContain('924 frames, with unsaved changes');
      expect(buttons()).toEqual(['Resume', 'Close it']);
    });

    it('offers only closing when the session cannot be resumed', () => {
      offer({ resumable: false });
      fixture.detectChanges();
      expect(buttons()).toEqual(['Close it']);
    });

    it('wires its buttons to resuming and closing', () => {
      const resume = vi.spyOn(component.sessions, 'resumeOpenSession').mockResolvedValue(true);
      const close = vi.spyOn(component.leave, 'closeOpenBackendSession').mockResolvedValue(true);
      offer();
      fixture.detectChanges();
      const [resumeButton, closeButton] = Array.from(banner()!.querySelectorAll('button'));
      resumeButton.click();
      closeButton.click();
      expect(resume).toHaveBeenCalledTimes(1);
      expect(close).toHaveBeenCalledTimes(1);
    });
  });

  describe('undo and redo', () => {
    const buttons = (): { undo: HTMLButtonElement; redo: HTMLButtonElement } => {
      const [undo, redo] = Array.from(
        fixture.nativeElement.querySelectorAll(
          '.canvas-toolbar button[aria-label="Undo"], .canvas-toolbar button[aria-label="Redo"]',
        ),
      ) as HTMLButtonElement[];
      return { undo, redo };
    };
    const steps = { undo: 2, redo: 1, next_undo: 'Add points', next_redo: 'Edit object' };
    const press = (init: KeyboardEventInit, target: EventTarget = window) =>
      target.dispatchEvent(
        new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }),
      );

    it('are disabled until the backend has steps', () => {
      component.store.isInitialized.set(true);
      fixture.detectChanges();
      expect(buttons().undo.disabled).toBe(true);
      expect(buttons().redo.disabled).toBe(true);
      expect(buttons().undo.title).toBe('Nothing to undo');
    });

    it('enable with the next step in their tooltip and call the history service', () => {
      const undo = vi.spyOn(component.changes, 'undo').mockResolvedValue(true);
      const redo = vi.spyOn(component.changes, 'redo').mockResolvedValue(true);
      component.store.isInitialized.set(true);
      component.store.history.set(steps);
      fixture.detectChanges();
      expect(buttons().undo.title).toBe('Undo: Add points (Ctrl+Z)');
      expect(buttons().redo.title).toBe('Redo: Edit object (Ctrl+Shift+Z)');
      buttons().undo.click();
      buttons().redo.click();
      expect(undo).toHaveBeenCalledTimes(1);
      expect(redo).toHaveBeenCalledTimes(1);
    });

    it('answer Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y (and the Cmd versions)', () => {
      const undo = vi.spyOn(component.changes, 'undo').mockResolvedValue(true);
      const redo = vi.spyOn(component.changes, 'redo').mockResolvedValue(true);
      press({ key: 'z', ctrlKey: true });
      press({ key: 'z', metaKey: true });
      expect(undo).toHaveBeenCalledTimes(2);
      press({ key: 'z', ctrlKey: true, shiftKey: true });
      press({ key: 'y', ctrlKey: true });
      press({ key: 'Z', metaKey: true, shiftKey: true });
      expect(redo).toHaveBeenCalledTimes(3);
    });

    it('leave Ctrl+Z to a text field while the user is typing', () => {
      const undo = vi.spyOn(component.changes, 'undo').mockResolvedValue(true);
      const field = document.createElement('input');
      field.type = 'text';
      document.body.appendChild(field);
      expect(press({ key: 'z', ctrlKey: true }, field)).toBe(true); // not prevented: the field undoes its text
      field.remove();
      expect(undo).not.toHaveBeenCalled();
    });
  });

  describe('the timeline tint for masks from an earlier propagation', () => {
    const segments = (): HTMLElement[] =>
      Array.from(fixture.nativeElement.querySelectorAll('.timeline-outdated-segment'));

    it('draws each stretch where it is on the timeline, with a plain tooltip', () => {
      const label = component.store.objects()[0].label;
      component.store.isInitialized.set(true);
      component.store.numFrames.set(101);
      component.store.outdatedMasks.set({ [String(component.store.objects()[0].id)]: [[0, 49]] });
      fixture.detectChanges();
      const [segment] = segments();
      expect(segments()).toHaveLength(1);
      expect(segment.style.left).toBe('0%');
      expect(segment.style.width).toBe('49%');
      expect(segment.getAttribute('title')).toBe(
        `Frames 0\u201349: masks for ${label} come from an earlier propagation. They are still shown and included in exports.`,
      );
    });
  });

  it('uses the native Tauri video picker when video mode is selected', async () => {
    desktopBridgeMock.isTauri.mockReturnValue(true);
    desktopBridgeMock.pickVideoFile.mockResolvedValue('C:/videos/example.mp4');
    component.store.loadSourceMode.set('video_file');

    await component.browseSelectedSource();

    expect(desktopBridgeMock.pickVideoFile).toHaveBeenCalled();
    expect(component.store.videoDir()).toBe('C:/videos/example.mp4');
  });

  it('falls back to browser video input when Tauri runtime is unavailable in video mode', async () => {
    const pickerSpy = vi
      .spyOn(component, 'openVideoFilePicker')
      .mockImplementation(() => undefined);
    component.store.loadSourceMode.set('video_file');

    await component.browseSelectedSource();

    expect(pickerSpy).toHaveBeenCalled();
  });

  it('sets propagation frame bounds from the displayed frame while keeping the range valid', () => {
    component.store.displayedFrameIdx.set(8);
    component.store.propagationStartFrameIdx.set(0);
    component.store.propagationEndFrameIdx.set(5);

    component.setPropagationStartFrame();

    expect(component.store.propagationStartFrameIdx()).toBe(8);
    expect(component.store.propagationEndFrameIdx()).toBe(8);

    component.store.displayedFrameIdx.set(3);
    component.setPropagationEndFrame();

    expect(component.store.propagationStartFrameIdx()).toBe(3);
    expect(component.store.propagationEndFrameIdx()).toBe(3);
  });

  describe('saving and unsaved changes', () => {
    const savedState = (overrides = {}) => ({
      checkpoint_id: 'cp-1',
      display_name: 'Bedroom',
      saved_at: 'then',
      dirty: false,
      ...overrides,
    });
    const status = () =>
      (
        fixture.nativeElement.querySelector('[data-testid="save-status"]') as HTMLElement
      ).textContent?.trim();
    const buttonLabelled = (label: string) =>
      (Array.from(fixture.nativeElement.querySelectorAll('button')) as HTMLButtonElement[]).find(
        (button) => button.textContent?.trim() === label,
      );

    beforeEach(() => {
      component.store.isInitialized.set(true);
    });

    it.each([
      [null, 'Not saved yet'],
      [{ checkpoint_id: null, dirty: false }, 'Not saved yet'],
      [{ checkpoint_id: null, dirty: true }, 'Unsaved changes'],
      [{ dirty: true }, 'Unsaved changes'],
      [{ dirty: false }, 'Saved'],
    ])('shows the save status for %j as "%s"', (state, label) => {
      component.store.savedState.set(state === null ? null : savedState(state));
      fixture.detectChanges();
      expect(status()).toBe(label);
    });

    it('needs a name for the first save and offers Save As only after a checkpoint exists', () => {
      component.store.savedState.set(savedState({ checkpoint_id: null, dirty: true }));
      component.store.saveName.set('');
      expect(component.canSave()).toBe(false);
      component.store.saveName.set('Bedroom');
      expect(component.canSave()).toBe(true);
      expect(component.canSaveAs()).toBe(false);

      component.store.savedState.set(savedState({ dirty: true }));
      expect(component.canSaveAs()).toBe(true);
      component.store.saveName.set('');
      expect(component.canSaveAs()).toBe(false);
    });

    it('wires Save and Save As to the session service', () => {
      component.store.savedState.set(savedState({ dirty: true }));
      component.store.saveName.set('Bedroom copy');
      const save = vi.spyOn(component.sessions, 'save').mockResolvedValue(true);
      fixture.detectChanges();
      buttonLabelled('Save')?.click();
      buttonLabelled('Save As')?.click();
      expect(save.mock.calls).toEqual([[], [true]]);
    });

    it('shows Close Session only while a session is open and closes it through the guard', () => {
      fixture.detectChanges();
      expect(buttonLabelled('Close Session')).toBeDefined();
      const close = vi.spyOn(component.leave, 'closeSession').mockResolvedValue(true);
      buttonLabelled('Close Session')?.click();
      expect(close).toHaveBeenCalledOnce();
      component.store.isInitialized.set(false);
      fixture.detectChanges();
      expect(buttonLabelled('Close Session')).toBeUndefined();
    });

    it('renders the unsaved-changes question in the page dialog', () => {
      component.store.savedState.set(savedState({ dirty: true }));
      void component.leave.confirmLeave('close it');
      fixture.detectChanges();
      expect(fixture.nativeElement.querySelector('[role="alertdialog"]')?.textContent).toContain(
        'Unsaved changes',
      );
      component.confirm.answer(null);
      fixture.detectChanges();
      expect(fixture.nativeElement.querySelector('[role="alertdialog"]')).toBeNull();
    });

    it('warns before the window closes only while changes are unsaved', () => {
      component.store.savedState.set(savedState({ dirty: true }));
      const unsaved = new Event('beforeunload', { cancelable: true }) as BeforeUnloadEvent;
      component.warnBeforeUnload(unsaved);
      expect(unsaved.defaultPrevented).toBe(true);

      component.store.savedState.set(savedState());
      const clean = new Event('beforeunload', { cancelable: true }) as BeforeUnloadEvent;
      component.warnBeforeUnload(clean);
      expect(clean.defaultPrevented).toBe(false);
    });
  });
});
