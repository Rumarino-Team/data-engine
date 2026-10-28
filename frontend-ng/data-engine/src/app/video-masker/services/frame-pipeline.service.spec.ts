import { setAcceptedPoints, initializeEditorStore } from '../testing/workflow-test-helpers';
import { TestBed } from '@angular/core/testing';
import { Subject, of } from 'rxjs';
import { vi } from 'vitest';
import { BackendService, VideoMaskDataResponse } from '../../services/backend.service';
import { FramePipelineService } from './frame-pipeline.service';
import { CanvasViewportService } from './canvas-viewport.service';
import { VideoMaskerStateStore } from './video-masker-state.store';

describe('FramePipelineService', () => {
  let service: FramePipelineService;
  let store: VideoMaskerStateStore;
  let images: FakeImage[];
  let backend: {
    getApiUrl: ReturnType<typeof vi.fn>;
    getVideoFrameUrl: ReturnType<typeof vi.fn>;
    getVideoMaskData: ReturnType<typeof vi.fn>;
  };
  let viewport: {
    attach: ReturnType<typeof vi.fn>;
    detach: ReturnType<typeof vi.fn>;
    reset: ReturnType<typeof vi.fn>;
    render: ReturnType<typeof vi.fn>;
  };

  class FakeImage {
    width = 100;
    height = 50;
    complete = true;
    src = '';
    onload: (() => Promise<void> | void) | null = null;
    onerror: (() => void) | null = null;
    removeAttribute(name: string): void {
      if (name === 'src') this.src = '';
    }
    constructor() {
      images.push(this);
    }
  }

  const maskResponse = (frameIdx: number): VideoMaskDataResponse => ({
    session_id: 'session-1',
    annotation_revision: store.annotationRevision()!,
    state_epoch: 3,
    sources: { '1': 'manifest' },
    frame_idx: frameIdx,
    objects: { '1': { size: [1, 2], rle: [[0, 1]], bbox: [0, 0, 1, 1] } },
  });

  beforeEach(() => {
    images = [];
    viewport = { attach: vi.fn(), detach: vi.fn(), reset: vi.fn(), render: vi.fn() };
    vi.stubGlobal('Image', FakeImage);
    backend = {
      getApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
      getVideoFrameUrl: vi.fn((idx: number) => `/frame/${idx}`),
      getVideoMaskData: vi.fn((idx: number) => of(maskResponse(idx))),
    };
    TestBed.configureTestingModule({
      providers: [
        FramePipelineService,
        VideoMaskerStateStore,
        { provide: CanvasViewportService, useValue: viewport },
        { provide: BackendService, useValue: backend },
      ],
    });
    service = TestBed.inject(FramePipelineService);
    store = TestBed.inject(VideoMaskerStateStore);
    initializeEditorStore(store);
    store.displayedFrameIdx.set(-1);
    store.selectedObjectId.set(null);
    service.attach(document.createElement('div'), () => {});
    store.objects.set([{ id: 1, label: 'Object 1', color: '#ff0000' }]);
    store.hasManifestMasks.set(true);
  });

  afterEach(() => {
    service.detach();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  async function startFrame(idx: number): Promise<FakeImage> {
    await service.loadFrame(idx);
    return images[images.length - 1];
  }

  it('passes frame, mask data, and prompt coordinates to the Konva viewport', async () => {
    setAcceptedPoints(
      store,
      new Map([
        [
          0,
          new Map([
            [
              1,
              [
                { x: 12, y: 24, label: 1 },
                { x: 30, y: 40, label: 0 },
              ],
            ],
          ]),
        ],
      ]),
    );
    const image = await startFrame(0);
    await image.onload?.();
    const rendered = viewport.render.mock.lastCall!;
    expect(rendered[0]).toBe(image);
    expect(rendered[1]).toEqual([
      { objectId: 1, color: '#ff0000', source: maskResponse(0).objects['1'] },
    ]);

    expect(rendered[6]).toBe(0);
    expect(rendered[2]).toEqual([
      { x: 12, y: 24, label: 1 },
      { x: 30, y: 40, label: 0 },
    ]);
    expect(store.lastMaskSource()).toBe('none');
  });

  it.each(['live', 'override', 'manifest'] as const)(
    'renders the backend-selected %s source without local precedence',
    async (source) => {
      const response = maskResponse(0);
      response.sources = { '1': source };
      backend.getVideoMaskData.mockReturnValue(of(response));
      store.selectedObjectId.set(1);
      const image = await startFrame(0);
      await image.onload?.();
      expect(viewport.render.mock.lastCall![1]).toEqual([
        { objectId: 1, color: '#ff0000', source: response.objects['1'] },
      ]);
      expect(store.lastMaskSource()).toBe(source);
    },
  );

  it('retains a backend-selected empty override on refetch', async () => {
    const response = maskResponse(0);
    response.objects['1'] = { size: [1, 2], rle: [], bbox: [0, 0, 0, 0] };
    response.sources['1'] = 'override';
    backend.getVideoMaskData.mockReturnValue(of(response));
    const image = await startFrame(0);
    await image.onload?.();
    service.clearMaskDataCache();
    await service.loadFrame(0);
    expect(viewport.render.mock.lastCall![1][0].source).toEqual(response.objects['1']);
  });

  it('rejects cached and late masks after an accepted deletion', async () => {
    const old = maskResponse(0);
    const image = await startFrame(0);
    await image.onload?.();
    store.annotations.set({ ...store.annotations()!, revision: 2, objects: {} });
    store.objects.set([]);
    service.redraw();
    expect(service.acceptMaskData(old, 0)).toBe(false);
    expect(viewport.render.mock.lastCall![1]).toEqual([]);
  });

  it.each(['session', 'revision', 'epoch', 'frame', 'backend'])(
    'rejects masks with stale %s identity',
    async (kind) => {
      const response = maskResponse(0);
      if (kind === 'session') response.session_id = 'other';
      if (kind === 'revision') response.annotation_revision++;
      if (kind === 'epoch') response.state_epoch++;
      if (kind === 'frame') response.frame_idx++;
      if (kind === 'backend') backend.getApiUrl.mockReturnValue('other');
      expect(service.acceptMaskData(response, 0)).toBe(false);
    },
  );

  it('ignores a mask response invalidated while it was in flight', async () => {
    const pending = new Subject<VideoMaskDataResponse>();
    backend.getVideoMaskData.mockReturnValue(pending);
    const image = await startFrame(0);
    const loaded = image.onload?.();
    service.clearMaskDataCache();
    pending.next(maskResponse(0));
    pending.complete();
    await loaded;
    service.redraw();
    expect(viewport.render.mock.lastCall![1]).toEqual([]);
  });

  it('does not let a previous frame mask failure erase the current mask', async () => {
    const oldRequest = new Subject<VideoMaskDataResponse>();
    backend.getVideoMaskData.mockReturnValueOnce(oldRequest);
    const oldImage = await startFrame(0);
    const oldLoad = oldImage.onload?.();
    const currentImage = await startFrame(1);
    await currentImage.onload?.();
    oldRequest.error(new Error('Old frame failed'));
    await oldLoad;

    service.redraw();
    expect(viewport.render.mock.lastCall![1]).toHaveLength(1);
    expect(store.displayedFrameIdx()).toBe(1);
  });

  it('ignores frame preloads from a previous session', async () => {
    store.numFrames.set(2);
    const image = await startFrame(0);
    await image.onload?.();
    const oldPreload = images.find((entry) => entry.src === '/frame/1')!;
    service.clearFrameCaches();
    await oldPreload.onload?.();
    const count = images.length;
    await service.loadFrame(1);
    expect(images).toHaveLength(count + 1);
    expect(service.currentBaseImage).toBeNull();
  });

  it('ignores an older image finishing after a newer frame', async () => {
    const oldImage = await startFrame(0);
    const currentImage = await startFrame(1);
    await currentImage.onload?.();
    await oldImage.onload?.();
    expect(service.currentBaseImage).toBe(currentImage);
    expect(store.displayedFrameIdx()).toBe(1);
  });

  it('does not apply an image load after detach', async () => {
    const image = await startFrame(0);
    service.detach();
    await image.onload?.();
    expect(service.currentBaseImage).toBeNull();
    expect(store.displayedFrameIdx()).toBe(-1);
    expect(viewport.render).not.toHaveBeenCalled();
  });

  it('shares image loads when the same frame is requested repeatedly', async () => {
    await service.loadFrame(0);
    await service.loadFrame(0);
    expect(images).toHaveLength(1);
    expect(backend.getVideoFrameUrl).toHaveBeenCalledTimes(1);
    await images[0].onload?.();
    expect(store.displayedFrameIdx()).toBe(0);
  });

  it('promotes a pending neighbor preload to the displayed frame without another request', async () => {
    store.numFrames.set(3);
    const first = await startFrame(0);
    await first.onload?.();
    const preload = images.find((image) => image.src === '/frame/1')!;
    const count = images.length;
    await service.loadFrame(1);
    expect(images).toHaveLength(count);
    await preload.onload?.();
    expect(service.currentBaseImage).toBe(preload);
    expect(store.displayedFrameIdx()).toBe(1);
    expect(backend.getVideoFrameUrl.mock.calls.filter(([idx]) => idx === 1)).toHaveLength(1);
  });

  it('cancels images outside the new frame neighborhood and ignores saved callbacks', async () => {
    const old = await startFrame(0);
    const oldCallback = old.onload;
    await service.loadFrame(10);
    expect(old.src).toBe('');
    expect(old.onload).toBeNull();
    await oldCallback?.();
    expect(service.currentBaseImage).toBeNull();
  });

  it('unsubscribes obsolete mask requests when navigating to another frame', async () => {
    const response = new Subject<VideoMaskDataResponse>();
    backend.getVideoMaskData.mockReturnValueOnce(response);
    const image = await startFrame(0);
    const loaded = image.onload?.();
    expect(response.observed).toBe(true);
    await service.loadFrame(5);
    expect(response.observed).toBe(false);
    await loaded;
    expect(viewport.render).toHaveBeenCalledTimes(1);
  });

  it('shares a pending mask request when the current frame is requested again', async () => {
    const response = new Subject<VideoMaskDataResponse>();
    backend.getVideoMaskData.mockReturnValueOnce(response);
    const image = await startFrame(0);
    const firstLoad = image.onload?.();
    const secondLoad = service.loadFrame(0);
    expect(backend.getVideoMaskData).toHaveBeenCalledTimes(1);
    response.next(maskResponse(0));
    await Promise.all([firstLoad, secondLoad]);
    expect(viewport.render.mock.lastCall![1]).toHaveLength(1);
  });

  it('renders cached masks in one pass without refetching', async () => {
    const image = await startFrame(0);
    await image.onload?.();
    viewport.render.mockClear();
    await service.loadFrame(0);
    expect(viewport.render).toHaveBeenCalledTimes(1);
    expect(viewport.render.mock.lastCall![1]).toHaveLength(1);
    expect(backend.getVideoMaskData).toHaveBeenCalledTimes(1);
  });

  it('loads masks alongside the image and renders once when masks arrive first', async () => {
    const image = await startFrame(0);
    expect(backend.getVideoMaskData).toHaveBeenCalledTimes(1);
    expect(viewport.render).not.toHaveBeenCalled();
    await image.onload?.();
    expect(viewport.render).toHaveBeenCalledTimes(1);
    expect(viewport.render.mock.lastCall![1]).toHaveLength(1);
  });

  it('does not redraw a frame for an empty saved-mask response', async () => {
    backend.getVideoMaskData.mockReturnValue(of({ frame_idx: 0, objects: {} }));
    const image = await startFrame(0);
    await image.onload?.();
    expect(viewport.render).toHaveBeenCalledTimes(1);
  });

  it('keeps recently accessed images and evicts older images at the byte budget', async () => {
    store.hasManifestMasks.set(false);
    for (const index of [0, 1]) {
      const image = await startFrame(index);
      image.width = image.height = 4096; // 64 MiB each: two fill the 128 MiB budget.
      await image.onload?.();
    }
    await service.loadFrame(0); // Refresh frame 0; frame 1 is now least recently used.
    const third = await startFrame(2);
    third.width = third.height = 4096;
    await third.onload?.();
    const count = images.length;
    await service.loadFrame(0);
    expect(images).toHaveLength(count);
    await service.loadFrame(1);
    expect(images).toHaveLength(count + 1);
  });

  it('retains the displayed masks while the next image is loading', async () => {
    const image = await startFrame(0);
    await image.onload?.();
    await service.loadFrame(5);
    service.redraw();
    expect(viewport.render.mock.lastCall![0]).toBe(image);
    expect(viewport.render.mock.lastCall![1]).toHaveLength(1);
  });

  it('prefetches neighbor masks after the current pair is ready and promotes the pending request', async () => {
    store.numFrames.set(3);
    const responses = new Map(
      [0, 1, 2].map((frame) => [frame, new Subject<VideoMaskDataResponse>()]),
    );
    backend.getVideoMaskData.mockImplementation((frame) => responses.get(frame));
    const image = await startFrame(1);
    const loaded = image.onload?.();
    expect(backend.getVideoMaskData.mock.calls).toEqual([[1]]);
    responses.get(1)!.next(maskResponse(1));
    await loaded;
    expect(backend.getVideoMaskData.mock.calls).toEqual([[1], [2], [0]]);
    const neighbor = images.find((entry) => entry.src === '/frame/2')!;
    await service.loadFrame(2);
    expect(responses.get(2)!.observed).toBe(true);
    expect(responses.get(0)!.observed).toBe(false);
    const nextLoaded = neighbor.onload?.();
    responses.get(2)!.next(maskResponse(2));
    await nextLoaded;
    expect(backend.getVideoMaskData.mock.calls.filter(([frame]) => frame === 2)).toHaveLength(1);
    expect(viewport.render.mock.lastCall![6]).toBe(2);
    expect(viewport.render.mock.lastCall![1]).toHaveLength(1);
  });

  it('cancels distant image and mask work immediately and coalesces scheduled navigation', async () => {
    let animation: FrameRequestCallback | undefined;
    vi.stubGlobal(
      'requestAnimationFrame',
      vi.fn((callback) => {
        animation = callback;
        return 1;
      }),
    );
    const masks = new Subject<VideoMaskDataResponse>();
    backend.getVideoMaskData.mockReturnValueOnce(masks);
    const image = await startFrame(0);
    const callback = image.onload;
    service.scheduleFrameLoad(5);
    service.scheduleFrameLoad(6);
    expect(image.src).toBe('');
    expect(masks.observed).toBe(false);
    await callback?.();
    expect(viewport.render).not.toHaveBeenCalled();
    animation!(0);
    expect(backend.getVideoFrameUrl).toHaveBeenCalledWith(6);
    expect(backend.getVideoFrameUrl).not.toHaveBeenCalledWith(5);
    expect(requestAnimationFrame).toHaveBeenCalledOnce();
  });

  it('retains useful neighbor work without allowing it to paint before scheduled navigation', async () => {
    vi.stubGlobal(
      'requestAnimationFrame',
      vi.fn(() => 1),
    );
    const image = await startFrame(0);
    service.scheduleFrameLoad(1);
    expect(image.src).toBe('/frame/0');
    await image.onload?.();
    expect(service.currentBaseImage).toBeNull();
    expect(viewport.render).not.toHaveBeenCalled();
    await service.loadFrame(0);
    expect(service.currentBaseImage).toBe(image);
    expect(backend.getVideoFrameUrl.mock.calls.filter(([frame]) => frame === 0)).toHaveLength(1);
  });

  it('does not repaint obsolete masks between navigation intent and the next animation tick', async () => {
    vi.stubGlobal(
      'requestAnimationFrame',
      vi.fn(() => 1),
    );
    const response = new Subject<VideoMaskDataResponse>();
    backend.getVideoMaskData.mockReturnValueOnce(response);
    const image = await startFrame(0);
    const loaded = image.onload?.();
    expect(viewport.render).toHaveBeenCalledOnce();
    service.scheduleFrameLoad(1);
    response.next(maskResponse(0));
    await loaded;
    expect(viewport.render).toHaveBeenCalledOnce();
  });

  it('keeps compressed masks when decoded images are evicted', async () => {
    for (const frame of [0, 1, 2]) {
      const image = await startFrame(frame);
      image.width = image.height = 4096;
      await image.onload?.();
    }
    expect(backend.getVideoMaskData).toHaveBeenCalledTimes(3);
    const imageCount = images.length;
    await service.loadFrame(0);
    expect(images).toHaveLength(imageCount + 1);
    await images.at(-1)!.onload?.();
    expect(backend.getVideoMaskData).toHaveBeenCalledTimes(3);
    expect(viewport.render.mock.lastCall![1]).toHaveLength(1);
  });
});
