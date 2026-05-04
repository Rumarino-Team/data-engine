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
});
