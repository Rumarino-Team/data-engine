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
});
