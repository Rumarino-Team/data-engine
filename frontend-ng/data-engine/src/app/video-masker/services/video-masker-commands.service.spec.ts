import { of, throwError } from 'rxjs';
import { describe, expect, it, vi } from 'vitest';
import { BackendService } from '../../services/backend.service';
import { VideoMaskerCommandsService } from './video-masker-commands.service';

describe('VideoMaskerCommandsService', () => {
  it('applies returned masks on success', async () => {
    const backend = {
      addNewPointsOrBox: vi.fn(() =>
        of({
          request_frame_idx: 1,
          frame_idx: 1,
          frame_file: '00001.jpg',
          state_epoch: 2,
          out_obj_ids: [1],
          out_masks: [[[true]]],
          mask_pixel_counts: { 1: 1 },
          single_frame_fallback_used: false,
        }),
      ),
    } as unknown as BackendService;

    let points = new Map<number, Map<number, any[]>>();
    let masks = new Map<number, Map<number, boolean[][]>>();
    const service = new VideoMaskerCommandsService();

    await service.addPoint({
      backend,
      getObjects: () => [{ id: 1, name: 'Object 1', color: '#ff0000' }],
      frameIdx: 1,
      label: 1,
      objId: 1,
      x: 1,
      y: 2,
      expectedEpoch: 2,
      displayedFrameIdx: () => 1,
      getPointsMap: () => points as any,
      setPointsMap: (next) => {
        points = next as any;
      },
      getMasksMap: () => masks,
      setMasksMap: (next) => {
        masks = next;
      },
      setPointRequestInFlight: () => undefined,
      onBeforeRequest: () => undefined,
      getMaskPixelCount: () => 1,
      setResponseEpoch: () => undefined,
      setResponseFrame: () => undefined,
      setMaskDebug: () => undefined,
      setDiscardReason: () => undefined,
      onEpochMismatch: () => undefined,
      markObjectAsLiveEdited: () => undefined,
      wasLiveEditedBeforeRequest: () => false,
      unmarkObjectAsLiveEdited: () => undefined,
      redraw: () => undefined,
      onFailure: () => undefined,
    });

    expect(masks.get(1)?.get(1)).toEqual([[true]]);
  });

  it('calls failure callback on backend error', async () => {
    const backend = {
      addNewPointsOrBox: vi.fn(() => throwError(() => new Error('fail'))),
    } as unknown as BackendService;

    const onFailure = vi.fn();
    const service = new VideoMaskerCommandsService();

    await service.addPoint({
      backend,
      getObjects: () => [{ id: 1, name: 'Object 1', color: '#ff0000' }],
      frameIdx: 1,
      label: 1,
      objId: 1,
      x: 1,
      y: 2,
      expectedEpoch: 2,
      displayedFrameIdx: () => 1,
      getPointsMap: () => new Map(),
      setPointsMap: () => undefined,
      getMasksMap: () => new Map(),
      setMasksMap: () => undefined,
      setPointRequestInFlight: () => undefined,
      onBeforeRequest: () => undefined,
      getMaskPixelCount: () => 0,
      setResponseEpoch: () => undefined,
      setResponseFrame: () => undefined,
      setMaskDebug: () => undefined,
      setDiscardReason: () => undefined,
      onEpochMismatch: () => undefined,
      markObjectAsLiveEdited: () => undefined,
      wasLiveEditedBeforeRequest: () => false,
      unmarkObjectAsLiveEdited: () => undefined,
      redraw: () => undefined,
      onFailure,
    });

    expect(onFailure).toHaveBeenCalled();
  });

  it('does not rollback points for an object removed while a point request was in flight', async () => {
    const backend = {
      addNewPointsOrBox: vi.fn(() => throwError(() => new Error('stale'))),
    } as unknown as BackendService;
    const objects: Array<{ id: number; name: string; color: string }> = [];
    let points = new Map<number, Map<number, any[]>>([
      [1, new Map([[1, [{ x: 9, y: 9, label: 1 }]]])],
    ]);
    const service = new VideoMaskerCommandsService();

    await service.addPoint({
      backend,
      getObjects: () => objects,
      frameIdx: 1,
      label: 1,
      objId: 1,
      x: 1,
      y: 2,
      expectedEpoch: 2,
      displayedFrameIdx: () => 1,
      getPointsMap: () => points as any,
      setPointsMap: (next) => {
        points = next as any;
      },
      getMasksMap: () => new Map(),
      setMasksMap: () => undefined,
      setPointRequestInFlight: () => undefined,
      onBeforeRequest: () => undefined,
      getMaskPixelCount: () => 0,
      setResponseEpoch: () => undefined,
      setResponseFrame: () => undefined,
      setMaskDebug: () => undefined,
      setDiscardReason: () => undefined,
      onEpochMismatch: () => undefined,
      markObjectAsLiveEdited: () => undefined,
      wasLiveEditedBeforeRequest: () => false,
      unmarkObjectAsLiveEdited: () => undefined,
      redraw: () => undefined,
      onFailure: () => undefined,
    });

    expect(points.has(1)).toBe(false);
  });

  it('adds new objects with an unused id after deletion gaps', () => {
    const service = new VideoMaskerCommandsService();
    let objects = [
      { id: 1, name: 'Object 1', color: '#111111' },
      { id: 3, name: 'Object 3', color: '#333333' },
    ];
    let selectedObjectId: number | null = null;

    service.addObject({
      backend: {} as BackendService,
      getObjects: () => objects,
      setObjects: (next) => {
        objects = next;
      },
      getSelectedObjectId: () => selectedObjectId,
      setSelectedObjectId: (value) => {
        selectedObjectId = value;
      },
      getMasksMap: () => new Map(),
      setMasksMap: () => undefined,
      getPointsMap: () => new Map(),
      setPointsMap: () => undefined,
      getTrackedPoints: () => [],
      setTrackedPoints: () => undefined,
      getLiveEditedObjectFrames: () => new Map(),
      setLiveEditedObjectFrames: () => undefined,
      randomColor: () => '#444444',
      redraw: () => undefined,
      showError: () => undefined,
    });

    expect(objects.map((entry) => entry.id)).toEqual([1, 3, 4]);
    expect(selectedObjectId).toBe(4);
  });

  it('removes selected object state and refreshes manifest masks', () => {
    const backend = {
      removeObject: vi.fn(() => of({ state_epoch: 4 })),
    } as unknown as BackendService;
    const service = new VideoMaskerCommandsService();
    let objects = [
      { id: 1, name: 'Object 1', color: '#111111' },
      { id: 2, name: 'Object 2', color: '#222222' },
    ];
    let selectedObjectId: number | null = 1;
    let masks = new Map<number, Map<number, boolean[][]>>([
      [5, new Map([[1, [[true]]], [2, [[false]]]])],
    ]);
    let points = new Map<number, Map<number, any[]>>([
      [5, new Map([[1, [{ x: 1, y: 1, label: 1 }]], [2, [{ x: 2, y: 2, label: 1 }]]])],
    ]);
    let liveEdited = new Map<number, Set<number>>([[5, new Set([1, 2])]]);
    const refreshManifestMasks = vi.fn();
    const updateStateEpoch = vi.fn();

    service.removeObject({
      backend,
      getObjects: () => objects,
      setObjects: (next) => {
        objects = next;
      },
      getSelectedObjectId: () => selectedObjectId,
      setSelectedObjectId: (value) => {
        selectedObjectId = value;
      },
      getMasksMap: () => masks,
      setMasksMap: (next) => {
        masks = next;
      },
      getPointsMap: () => points as any,
      setPointsMap: (next) => {
        points = next as any;
      },
      getTrackedPoints: () => [{ obj_id: 1 }, { obj_id: 2 }],
      setTrackedPoints: () => undefined,
      getLiveEditedObjectFrames: () => liveEdited,
      setLiveEditedObjectFrames: (next) => {
        liveEdited = next;
      },
      updateStateEpoch,
      refreshManifestMasks,
      randomColor: () => '#333333',
      redraw: () => undefined,
      showError: () => undefined,
    });

    expect(objects.map((entry) => entry.id)).toEqual([2]);
    expect(selectedObjectId).toBe(2);
    expect(masks.get(5)?.has(1)).toBe(false);
    expect(points.get(5)?.has(1)).toBe(false);
    expect(liveEdited.get(5)).toEqual(new Set([2]));
    expect(updateStateEpoch).toHaveBeenCalledWith(4, 'remove object');
    expect(refreshManifestMasks).toHaveBeenCalled();
  });

  it('applies successful removals when remove all fails partway through', async () => {
    const backend = {
      removeObject: vi
        .fn()
        .mockReturnValueOnce(of({ state_epoch: 4 }))
        .mockReturnValueOnce(throwError(() => new Error('remove failed'))),
    } as unknown as BackendService;
    const service = new VideoMaskerCommandsService();
    let objects = [
      { id: 1, name: 'Object 1', color: '#111111' },
      { id: 2, name: 'Object 2', color: '#222222' },
    ];
    let selectedObjectId: number | null = 1;
    let masks = new Map<number, Map<number, boolean[][]>>([
      [5, new Map([[1, [[true]]], [2, [[false]]]])],
    ]);
    const showError = vi.fn();
    const refreshManifestMasks = vi.fn();

    await service.removeAllObjects({
      backend,
      getObjects: () => objects,
      setObjects: (next) => {
        objects = next;
      },
      getSelectedObjectId: () => selectedObjectId,
      setSelectedObjectId: (value) => {
        selectedObjectId = value;
      },
      getMasksMap: () => masks,
      setMasksMap: (next) => {
        masks = next;
      },
      getPointsMap: () => new Map(),
      setPointsMap: () => undefined,
      getTrackedPoints: () => [],
      setTrackedPoints: () => undefined,
      getLiveEditedObjectFrames: () => new Map(),
      setLiveEditedObjectFrames: () => undefined,
      updateStateEpoch: () => undefined,
      refreshManifestMasks,
      randomColor: () => '#333333',
      redraw: () => undefined,
      showError,
    });

    expect(objects.map((entry) => entry.id)).toEqual([2]);
    expect(selectedObjectId).toBe(2);
    expect(masks.get(5)?.has(1)).toBe(false);
    expect(masks.get(5)?.has(2)).toBe(true);
    expect(refreshManifestMasks).toHaveBeenCalled();
    expect(showError).toHaveBeenCalled();
  });
});
