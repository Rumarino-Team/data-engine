import { cachedMasks, seedMasks } from '../testing/workflow-test-helpers';
import { describe, beforeEach, expect, it } from 'vitest';
import { createWorkflowFixture, mockJob } from '../testing/workflow-test-helpers';

describe('mask-propagation workflows', () => {
  let editor: ReturnType<typeof createWorkflowFixture>['editor'];
  let backendMock: ReturnType<typeof createWorkflowFixture>['backendMock'];

  beforeEach(() => {
    ({ editor, backendMock } = createWorkflowFixture());
  });

  it('invalidates rendering caches and refetches backend-selected masks after propagation', async () => {
    mockJob(
      backendMock.propagateInVideo,
      backendMock.getJob,
      'mask_propagation',
      {
        result: {
          video_segments: {},
          saved_mask_paths: {},
          video_segments_total_frames: 2,
          video_segments_returned_frames: 0,
          video_segments_returned_mask_values: 0,
          video_segments_truncated: false,
          'state.mask_manifest_path': '/tmp/session/masks/manifest.json',
          state_epoch: 3,
          saved_state: {
            checkpoint_id: 'cp-1',
            display_name: 'Bedroom',
            saved_at: 'then',
            dirty: true,
          },
        },
      },
      'job-propagate',
    );
    editor.store.isInitialized.set(true);
    editor.store.numFrames.set(5);
    editor.store.stateEpoch.set(2);
    editor.store.propagationStartFrameIdx.set(1);
    editor.store.propagationEndFrameIdx.set(2);
    const originalMask = [[true, false]];
    seedMasks(editor, new Map([0, 1, 2, 4].map((frame) => [frame, new Map([[1, originalMask]])])));

    await editor.propagation.propagate();

    expect(editor.store.hasManifestMasks()).toBe(true);
    expect(editor.store.stateEpoch()).toBe(3);
    expect(editor.store.savedState()?.dirty).toBe(true);
    expect([...cachedMasks(editor.framePipeline).keys()]).toEqual([]);
    expect(backendMock.propagateInVideo).toHaveBeenCalledWith(
      {
        start_frame_idx: 1,
        max_frame_num_to_track: 2,
        include_masks_in_response: false,
        include_saved_mask_paths: false,
      },
      { session_id: 'session-1', expected_revision: editor.store.annotationRevision() },
    );
  });
});
