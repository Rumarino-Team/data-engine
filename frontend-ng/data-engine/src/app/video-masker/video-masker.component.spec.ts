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

  it('hides the API URL scheme while the input is not focused', () => {
    component.store.apiUrlInput.set('http://127.0.0.1:8000');

    expect(component.getApiUrlInputValue()).toBe('127.0.0.1:8000');

    component.onApiUrlFocus();

    expect(component.getApiUrlInputValue()).toBe('http://127.0.0.1:8000');
  });

  it('adds the default API URL scheme when focusing a host-only value', () => {
    component.store.apiUrlInput.set('127.0.0.1:8000');

    component.onApiUrlFocus();

    expect(component.store.apiUrlInput()).toBe('http://127.0.0.1:8000');
    expect(component.getApiUrlInputValue()).toBe('http://127.0.0.1:8000');
  });

  it('uses the native Tauri video picker when video mode is selected', async () => {
    desktopBridgeMock.isTauri.mockReturnValue(true);
    desktopBridgeMock.pickVideoFile.mockResolvedValue('C:/videos/example.mp4');
    component.store.loadSourceMode.set('video_file');

    await component.browseSelectedSource();

    expect(desktopBridgeMock.pickVideoFile).toHaveBeenCalled();
    expect(component.store.videoDir()).toBe('C:/videos/example.mp4');
  });

  it('uses the native Tauri directory picker for saved-session mode', async () => {
    desktopBridgeMock.isTauri.mockReturnValue(true);
    desktopBridgeMock.pickFramesDirectory.mockResolvedValue('C:/sessions/saved1');
    component.store.loadSourceMode.set('saved_session_dir');

    await component.browseSelectedSource();

    expect(desktopBridgeMock.pickFramesDirectory).toHaveBeenCalled();
    expect(component.store.videoDir()).toBe('C:/sessions/saved1');
  });

  it('falls back to browser video input when Tauri runtime is unavailable in video mode', async () => {
    const pickerSpy = vi
      .spyOn(component, 'openVideoFilePicker')
      .mockImplementation(() => undefined);
    component.store.loadSourceMode.set('video_file');

    await component.browseSelectedSource();

    expect(pickerSpy).toHaveBeenCalled();
  });

  it('falls back to browser directory input when Tauri runtime is unavailable in frames mode', async () => {
    const pickerSpy = vi
      .spyOn(component, 'openFramesDirPicker')
      .mockImplementation(() => undefined);
    component.store.loadSourceMode.set('frames_dir');

    await component.browseSelectedSource();

    expect(pickerSpy).toHaveBeenCalled();
  });

  it('updates load placeholder and browse label based on selected load mode', () => {
    component.store.loadSourceMode.set('frames_dir');
    expect(component.getBrowseLabel()).toBe('Browse Frames');
    expect(component.getLoadPathPlaceholder()).toContain('frames directory');

    component.store.loadSourceMode.set('video_file');
    expect(component.getBrowseLabel()).toBe('Browse Video');
    expect(component.getLoadPathPlaceholder()).toContain('video file');

    component.store.loadSourceMode.set('saved_session_dir');
    expect(component.getBrowseLabel()).toBe('Browse Saved Session');
    expect(component.getLoadPathPlaceholder()).toContain('saved session directory');
    expect(component.getLoadPathPlaceholder()).toContain('session.json');
    expect(component.getLoadPathPlaceholder()).toContain('masks/');
  });

  it('labels the prompt tracking action as Track Prompt Points', () => {
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Track Prompt Points');
    expect(fixture.nativeElement.textContent).not.toContain('Run CoTracker');
  });

  it('renders displayed frame text without target text next to the frame scrubber', () => {
    component.store.isInitialized.set(true);
    component.store.numFrames.set(12);
    component.store.targetFrameIdx.set(4);
    component.store.displayedFrameIdx.set(4);

    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).not.toContain('Target:');
    expect(fixture.nativeElement.textContent).toContain('Displayed: 4 / 11');
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

  it('renders the propagation range markers on the timeline', () => {
    component.store.isInitialized.set(true);
    component.store.numFrames.set(100);
    component.store.propagationStartFrameIdx.set(20);
    component.store.propagationEndFrameIdx.set(80);

    fixture.detectChanges();

    const timeline = fixture.nativeElement.querySelector('.timeline') as HTMLElement;
    expect(parseFloat(timeline.style.getPropertyValue('--range-start'))).toBeCloseTo(
      (20 / 99) * 100,
    );
    expect(parseFloat(timeline.style.getPropertyValue('--range-end'))).toBeCloseTo((80 / 99) * 100);
    expect(timeline.textContent).toContain('Start 20');
    expect(timeline.textContent).toContain('End 80');
  });

  it('does not render a tracking model selector', () => {
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).not.toContain('Tracking Model');
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

    it('enables re-save for changes or a new name, and nothing while busy', () => {
      component.store.savedState.set(savedState());
      component.store.saveName.set('Bedroom');
      expect(component.canSave()).toBe(false);
      component.store.saveName.set('Kitchen');
      expect(component.canSave()).toBe(true);
      component.store.saveName.set('Bedroom');
      component.store.savedState.set(savedState({ dirty: true }));
      expect(component.canSave()).toBe(true);
      component.store.isLoading.set(true);
      expect(component.canSave()).toBe(false);
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
