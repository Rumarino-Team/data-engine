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
    frame_idx: frameIdx,
    objects: { '1': { size: [1, 2], rle: [[0, 1]], bbox: [0, 0, 1, 1] } },
  });

  beforeEach(() => {
    images = [];
    viewport = { attach: vi.fn(), detach: vi.fn(), reset: vi.fn(), render: vi.fn() };
    vi.stubGlobal('Image', FakeImage);
    backend = {
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
    service.attach(document.createElement('div'), () => {});
    store.objects.set([{ id: 1, name: 'Object 1', color: '#ff0000' }]);
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
    store.points.set(
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

  it('suppresses a saved mask when the live edit explicitly returns an empty mask', async () => {
    store.selectedObjectId.set(1);
    store.liveEditedObjectFrames.set(new Map([[0, new Set([1])]]));
    store.masks.set(new Map([[0, new Map([[1, [[false, false]]]])]]));
    const image = await startFrame(0);
    await image.onload?.();
    expect(viewport.render.mock.lastCall![1]).toEqual([]);
    expect(store.lastMaskSource()).toBe('none');
  });

  it('does not paint deleted objects from cached or refetched manifests', async () => {
    const image = await startFrame(0);
    await image.onload?.();
    expect(viewport.render.mock.lastCall![1]).toHaveLength(1);
    service.removeObject(1);
    store.objects.set([]);

    service.redraw();
    await service.loadFrame(0);
    expect(backend.getVideoMaskData).toHaveBeenCalledTimes(2);
    expect(viewport.render.mock.lastCall![1]).toEqual([]);
  });

  it('renders encoded live masks directly and marks their source as live', async () => {
    const mask = {
      size: [1, 2] as [number, number],
      rle: [[1, 1]],
      bbox: [1, 0, 1, 1] as [number, number, number, number],
    };
    store.selectedObjectId.set(1);
    store.liveEditedObjectFrames.set(new Map([[0, new Set([1])]]));
    store.masks.set(new Map([[0, new Map([[1, mask]])]]));
    const image = await startFrame(0);
    await image.onload?.();
    expect(viewport.render.mock.lastCall![1]).toEqual([
      { objectId: 1, source: mask, color: '#ff0000' },
    ]);
    expect(viewport.render.mock.lastCall![1][0].source).toBe(mask);
    expect(store.lastMaskSource()).toBe('live');
  });

  it('suppresses saved foreground when an encoded live edit is empty', async () => {
    store.liveEditedObjectFrames.set(new Map([[0, new Set([1])]]));
    store.masks.set(
      new Map([
        [
          0,
          new Map([
            [
              1,
              {
                size: [1, 2] as [number, number],
                rle: [],
                bbox: [0, 0, 0, 0] as [number, number, number, number],
              },
            ],
          ]),
        ],
      ]),
    );
    const image = await startFrame(0);
    await image.onload?.();
    expect(viewport.render.mock.lastCall![1]).toEqual([]);
  });

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
});
