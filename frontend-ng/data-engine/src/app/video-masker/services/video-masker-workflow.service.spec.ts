import { of } from 'rxjs';
import { describe, expect, it, vi } from 'vitest';
import { BackendService } from '../../services/backend.service';
import { VideoMaskerWorkflowService } from './video-masker-workflow.service';

describe('VideoMaskerWorkflowService', () => {
  it('propagate updates manifest and schedules frame', async () => {
    const backend = {
      propagateInVideo: vi.fn(() => of({ job_id: 'job-1' })),
    } as unknown as BackendService;

    const service = new VideoMaskerWorkflowService();
    const setHasManifestMasks = vi.fn();
    const scheduleFrameLoad = vi.fn();

    await service.propagate({
      backend,
      runBackendJob: vi.fn(async () => ({ state_epoch: 3, mask_manifest_path: 'm.json' } as any)),
      updateStateEpoch: vi.fn(),
      setHasManifestMasks,
      clearMaskDataCache: vi.fn(),
      scheduleFrameLoad,
      targetFrameIdx: () => 4,
      setTrackedPoints: vi.fn(),
      drawCurrentFrame: vi.fn(),
      showToast: vi.fn(),
      getErrorMessage: vi.fn(() => 'err'),
      getLastCompletedJobId: vi.fn(() => null),
      trackingUseSupportGrid: vi.fn(() => false),
      setIsLoading: vi.fn(),
      getSaveName: vi.fn(() => 's'),
      setSaveName: vi.fn(),
      buildInteractiveStateSnapshot: vi.fn(() => ({ version: 1 } as any)),
      resetInteractiveMaps: vi.fn(),
    });

    expect(setHasManifestMasks).toHaveBeenCalledWith(true);
    expect(scheduleFrameLoad).toHaveBeenCalledWith(4);
  });

  it('propagate warns when tracking guidance is skipped', async () => {
    const backend = {
      propagateInVideo: vi.fn(() => of({ job_id: 'job-1' })),
    } as unknown as BackendService;

    const service = new VideoMaskerWorkflowService();
    const showToast = vi.fn();

    await service.propagate({
      backend,
      runBackendJob: vi.fn(async () => ({
        state_epoch: 3,
        mask_manifest_path: 'm.json',
        tracked_points_skipped_reason: 'No CoTracker result available for propagation.',
      } as any)),
      updateStateEpoch: vi.fn(),
      setHasManifestMasks: vi.fn(),
      clearMaskDataCache: vi.fn(),
      scheduleFrameLoad: vi.fn(),
      targetFrameIdx: () => 4,
      setTrackedPoints: vi.fn(),
      drawCurrentFrame: vi.fn(),
      showToast,
      getErrorMessage: vi.fn(() => 'err'),
      getLastCompletedJobId: vi.fn(() => null),
      trackingUseSupportGrid: vi.fn(() => false),
      setIsLoading: vi.fn(),
      getSaveName: vi.fn(() => 's'),
      setSaveName: vi.fn(),
      buildInteractiveStateSnapshot: vi.fn(() => ({ version: 1 } as any)),
      resetInteractiveMaps: vi.fn(),
    });

    expect(showToast).toHaveBeenCalledWith(
      'warning',
      'Tracking guidance skipped',
      'No CoTracker result available for propagation.',
    );
  });
});
