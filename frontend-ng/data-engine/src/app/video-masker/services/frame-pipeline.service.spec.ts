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
});
