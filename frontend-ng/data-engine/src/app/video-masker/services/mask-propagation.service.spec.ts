import { describe, beforeEach, expect, it, vi } from 'vitest';
import { of } from 'rxjs';
import { createWorkflowFixture } from '../testing/workflow-test-helpers';

describe('mask-propagation workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];

  beforeEach(() => {
    ({ editor, backendMock } = createWorkflowFixture());
  });

  it('loads the propagated manifest while preserving live masks outside the requested range', async () => {
    backendMock.propagateInVideo.mockReturnValue(
      of({
        job_id: 'job-propagate',
        status: 'queued',
        operation: 'mask_propagation',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-propagate',
          operation: 'mask_propagation',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 2,
          total: 2,
          window_index: null,
          window_count: null,
          frame_idx: null,
          stage_history: [],
          message: 'done',
          error: null,
          started_at: 'now',
          updated_at: 'now',
          completed_at: 'now',
          result: {
            video_segments: {},
            saved_mask_paths: {},
            video_segments_total_frames: 2,
            video_segments_returned_frames: 0,
            video_segments_returned_mask_values: 0,
            video_segments_truncated: false,
            'state.mask_manifest_path': '/tmp/session/masks/manifest.json',
            state_epoch: 3,
          },
        },
      }),
    );
    editor.store.isInitialized.set(true);
    editor.store.numFrames.set(5);
    editor.store.stateEpoch.set(2);
    editor.store.propagationStartFrameIdx.set(1);
    editor.store.propagationEndFrameIdx.set(2);
    const originalMask = [[true, false]];
    editor.store.masks.set(
      new Map([0, 1, 2, 4].map((frame) => [frame, new Map([[1, originalMask]])])),
    );
    editor.store.liveEditedObjectFrames.set(
      new Map([0, 1, 2, 4].map((frame) => [frame, new Set([1])])),
    );

    await editor.propagation.propagate();

    expect(editor.store.hasManifestMasks()).toBe(true);
    expect(editor.store.stateEpoch()).toBe(3);
    expect([...editor.store.masks().keys()]).toEqual([0, 4]);
    expect(editor.store.masks().get(4)?.get(1)).toBe(originalMask);
    expect([...editor.store.liveEditedObjectFrames().keys()]).toEqual([0, 4]);
    expect(backendMock.propagateInVideo).toHaveBeenCalledWith({
      start_frame_idx: 1,
      max_frame_num_to_track: 2,
      include_masks_in_response: false,
      include_saved_mask_paths: false,
    });
  });
});
