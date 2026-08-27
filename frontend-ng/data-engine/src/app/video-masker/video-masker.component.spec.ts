import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import { vi } from 'vitest';
import { BackendService, VideoAddPointsResponse } from '../services/backend.service';
import { DesktopBridgeService } from '../services/desktop-bridge.service';
import { VideoMaskerComponent } from './video-masker.component';

describe('VideoMaskerComponent sync contract', () => {
  let component: VideoMaskerComponent;
  let fixture: ComponentFixture<VideoMaskerComponent>;
  let backendMock: {
    addNewPointsOrBox: ReturnType<typeof vi.fn>;
    health: ReturnType<typeof vi.fn>;
    initVideoState: ReturnType<typeof vi.fn>;
    getJob: ReturnType<typeof vi.fn>;
    trackPromptPoints: ReturnType<typeof vi.fn>;
    propagateInVideo: ReturnType<typeof vi.fn>;
    getTrackingResult: ReturnType<typeof vi.fn>;
    clearJobResult: ReturnType<typeof vi.fn>;
    getApiUrl: ReturnType<typeof vi.fn>;
    setApiUrl: ReturnType<typeof vi.fn>;
    resetApiUrl: ReturnType<typeof vi.fn>;
    saveVideoSession: ReturnType<typeof vi.fn>;
  };
  let desktopBridgeMock: {
    isTauri: ReturnType<typeof vi.fn>;
    pickVideoFile: ReturnType<typeof vi.fn>;
    pickFramesDirectory: ReturnType<typeof vi.fn>;
  };

  const makeResponse = (overrides: Partial<VideoAddPointsResponse>): VideoAddPointsResponse => ({
    request_frame_idx: 5,
    frame_idx: 5,
    frame_file: '00005.jpg',
    out_obj_ids: [1],
    out_masks: [
      [
        [true, false],
        [false, false],
      ],
    ],
    mask_pixel_counts: { 1: 1 },
    mask_shapes: { 1: [2, 2] },
    state_epoch: 3,
    ...overrides,
  });

  beforeEach(async () => {
    backendMock = {
      addNewPointsOrBox: vi.fn(),
      health: vi.fn(() => of({ status: 'ok' })),
      initVideoState: vi.fn(),
      getJob: vi.fn(),
      trackPromptPoints: vi.fn(),
      propagateInVideo: vi.fn(),
      getTrackingResult: vi.fn(),
      clearJobResult: vi.fn(() => of({ cleared: true })),
      getApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
      setApiUrl: vi.fn((value: string) => value),
      resetApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
      saveVideoSession: vi.fn(),
    };
    desktopBridgeMock = {
      isTauri: vi.fn(() => false),
      pickVideoFile: vi.fn(),
      pickFramesDirectory: vi.fn(),
    };

    await TestBed.configureTestingModule({
      imports: [VideoMaskerComponent],
      providers: [
        {
          provide: BackendService,
          useValue: {
            addNewPointsOrBox: backendMock.addNewPointsOrBox,
            health: backendMock.health,
            initVideoState: backendMock.initVideoState,
            getJob: backendMock.getJob,
            trackPromptPoints: backendMock.trackPromptPoints,
            propagateInVideo: backendMock.propagateInVideo,
            getTrackingResult: backendMock.getTrackingResult,
            clearJobResult: backendMock.clearJobResult,
            getApiUrl: backendMock.getApiUrl,
            setApiUrl: backendMock.setApiUrl,
            resetApiUrl: backendMock.resetApiUrl,
            saveVideoSession: backendMock.saveVideoSession,
          },
        },
        {
          provide: DesktopBridgeService,
          useValue: desktopBridgeMock,
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(VideoMaskerComponent);
    component = fixture.componentInstance;
    component.store.selectedObjectId.set(1);
    component.store.objects.set([{ id: 1, name: 'Object 1', color: '#ff0000' }]);
    component.store.stateEpoch.set(3);
    component.store.displayedFrameIdx.set(5);
  });

  it('uses displayed frame index in request and stores mask on that frame', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({})));

    await component.addPoint(12, 24, 1, 5);

    expect(backendMock.addNewPointsOrBox).toHaveBeenCalledWith(
      expect.objectContaining({ frame_idx: 5 }),
    );
    expect(component.store.masks().get(5)?.get(1)).toEqual([
      [true, false],
      [false, false],
    ]);
    expect(component.store.lastDiscardReason()).toBeNull();
  });

  it('discards mismatched response frame and rolls back optimistic point', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(
      of(makeResponse({ request_frame_idx: 5, frame_idx: 4 })),
    );

    await component.addPoint(10, 20, 1, 5);

    expect(component.store.masks().get(5)?.get(1)).toBeUndefined();
    expect(component.store.points().get(5)?.get(1)?.length ?? 0).toBe(0);
    expect(component.store.lastDiscardReason()).toContain('frame mismatch');
  });

  it('discards stale epoch responses and clears live masks', async () => {
    const existingMasks = new Map<number, Map<number, boolean[][]>>();
    existingMasks.set(2, new Map([[1, [[true]]]]));
    component.store.masks.set(existingMasks);
    component.store.liveEditedObjectFrames.set(new Map([[2, new Set([1])]]));

    backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({ state_epoch: 4 })));

    await component.addPoint(14, 18, 1, 5);

    expect(component.store.stateEpoch()).toBe(4);
    expect(component.store.masks().size).toBe(0);
    expect(component.store.liveEditedObjectFrames().size).toBe(0);
    expect(component.store.lastDiscardReason()).toContain('epoch mismatch');
  });

  it('keeps frame/object marked as live-edited even when returned mask is empty', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(
      of(
        makeResponse({
          out_masks: [
            [
              [false, false],
              [false, false],
            ],
          ],
          mask_pixel_counts: { 1: 0 },
        }),
      ),
    );

    await component.addPoint(30, 40, 1, 5);

    expect(component.store.liveEditedObjectFrames().get(5)?.has(1)).toBe(true);
    expect(component.store.lastMaskPixelCount()).toBe(0);
  });

  it('does not mark an object live-edited until the point mask response returns', async () => {
    const response$ = new Subject<VideoAddPointsResponse>();
    backendMock.addNewPointsOrBox.mockReturnValue(response$);

    const pendingRequest = component.addPoint(30, 40, 1, 5);

    expect(component.store.points().get(5)?.get(1)?.length).toBe(1);
    expect(component.store.liveEditedObjectFrames().get(5)?.has(1)).toBeFalsy();

    response$.next(makeResponse({}));
    response$.complete();
    await pendingRequest;

    expect(component.store.liveEditedObjectFrames().get(5)?.has(1)).toBe(true);
  });

  it('rolls back optimistic point and shows a toast when point update conflicts', async () => {
    backendMock.addNewPointsOrBox.mockReturnValue(
      throwError(() => ({ error: { detail: 'Another operation is already running.' } })),
    );

    await component.addPoint(30, 40, 1, 5);

    expect(component.store.points().get(5)?.get(1)?.length ?? 0).toBe(0);
    expect(component.store.toasts()[0].title).toBe('Point update failed');
    expect(component.store.toasts()[0].message).toBe('Another operation is already running.');
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

  it('starts a video init job, polls completion, and applies the result', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-1',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-1',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 12,
          total: 12,
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
            message: 'Video state initialized successfully',
            num_frames: 12,
            resolved_video_frames_dir: 'C:/frames',
            source_video_path: null,
            online_mode: true,
            batch_size: 32,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 7,
          },
        },
      }),
    );
    component.store.videoDir.set('C:/frames');

    await component.initVideo();

    expect(backendMock.initVideoState).toHaveBeenCalledWith('C:/frames');
    expect(backendMock.getJob).toHaveBeenCalledWith('job-1');
    expect(component.store.isInitialized()).toBe(true);
    expect(component.store.numFrames()).toBe(12);
    expect(component.store.stateEpoch()).toBe(7);
  });

  it('uses the same init endpoint path flow for saved-session mode', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-saved',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-saved',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 8,
          total: 8,
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
            message: 'Video state initialized successfully',
            num_frames: 8,
            resolved_video_frames_dir: 'C:/backend/saved/review-run/frames',
            source_video_path: null,
            online_mode: true,
            batch_size: 32,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 11,
          },
        },
      }),
    );
    component.store.loadSourceMode.set('saved_session_dir');
    component.store.videoDir.set('C:/backend/saved/review-run');

    await component.initVideo();

    expect(backendMock.initVideoState).toHaveBeenCalledWith('C:/backend/saved/review-run');
    expect(component.store.isInitialized()).toBe(true);
  });

  it('restores interactive state from saved-session init result', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-restore',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-restore',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 10,
          total: 10,
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
            message: 'Video state initialized successfully',
            num_frames: 10,
            resolved_video_frames_dir: 'C:/backend/saved/review-run/frames',
            source_video_path: null,
            online_mode: true,
            batch_size: 32,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 12,
            source_type: 'saved_session',
            restored_session: {
              session_meta: { schema_version: 2 },
              has_mask_manifest: true,
              interactive_state: {
                version: 1,
                objects: [{ id: 1, name: 'Object 1', color: '#ff6600' }],
                selected_object_id: 1,
                interaction_mode: 'negative',
                current_frame_idx: 4,
                points: [{ frame_idx: 4, obj_id: 1, x: 10, y: 20, label: 1 }],
                live_masks: [{ frame_idx: 4, obj_id: 1, height: 2, width: 2, counts: [0, 1, 3] }],
              },
            },
          },
        },
      }),
    );
    component.store.loadSourceMode.set('saved_session_dir');
    component.store.videoDir.set('C:/backend/saved/review-run');

    await component.initVideo();

    expect(component.store.hasManifestMasks()).toBe(true);
    expect(component.store.interactionMode()).toBe('negative');
    expect(component.store.targetFrameIdx()).toBe(4);
    expect(component.store.points().get(4)?.get(1)?.length).toBe(1);
    expect(component.store.masks().get(4)?.get(1)).toEqual([
      [true, false],
      [false, false],
    ]);
  });

  it('restores persisted tracking result from saved-session init result', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-restore-track',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-restore-track',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 10,
          total: 10,
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
            message: 'Video state initialized successfully',
            num_frames: 10,
            resolved_video_frames_dir: 'C:/backend/saved/review-run/frames',
            source_video_path: null,
            online_mode: true,
            batch_size: 32,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 12,
            source_type: 'saved_session',
            restored_session: {
              session_meta: { schema_version: 2 },
              has_mask_manifest: true,
              tracking_result: {
                result_id: 'track-restored',
                summary: { num_points: 1 },
              },
            },
          },
        },
      }),
    );
    backendMock.getTrackingResult.mockReturnValue(
      of({
        result: {
          version: 1,
          result_id: 'track-restored',
          model_name: 'cotracker3_online',
          num_points: 1,
          num_frames: 2,
          add_support_grid_used: false,
          tracking_mode: 'streaming',
          streaming_frame_threshold: 256,
          tracks: [
            [
              [10, 20],
              [11, 21],
            ],
          ],
          visibility: [[true, true]],
          points: [
            {
              point_id: 'p0_0',
              obj_id: 1,
              source_frame_idx: 0,
              source_x: 10,
              source_y: 20,
            },
          ],
        },
      }),
    );
    component.store.loadSourceMode.set('saved_session_dir');
    component.store.videoDir.set('C:/backend/saved/review-run');

    await component.initVideo();

    expect(backendMock.getTrackingResult).toHaveBeenCalledWith('track-restored');
    expect(component.store.trackedPoints()[0].tracks).toEqual([
      [10, 20],
      [11, 21],
    ]);
  });

  it('keeps saved-session init successful when restored tracking result is unavailable', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-restore-track-missing',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-restore-track-missing',
          operation: 'video_init',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 10,
          total: 10,
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
            message: 'Video state initialized successfully',
            num_frames: 10,
            resolved_video_frames_dir: 'C:/backend/saved/review-run/frames',
            source_video_path: null,
            online_mode: true,
            batch_size: 32,
            offload_video_to_cpu: true,
            offload_state_to_cpu: true,
            state_epoch: 12,
            source_type: 'saved_session',
            restored_session: {
              session_meta: { schema_version: 2 },
              has_mask_manifest: true,
              tracking_result: { result_id: 'missing-track' },
            },
          },
        },
      }),
    );
    backendMock.getTrackingResult.mockReturnValue(
      throwError(() => ({ error: { detail: 'missing' } })),
    );
    component.store.loadSourceMode.set('saved_session_dir');
    component.store.videoDir.set('C:/backend/saved/review-run');

    await component.initVideo();

    expect(component.store.isInitialized()).toBe(true);
    expect(component.store.toasts()[0].title).toBe('Tracking result unavailable');
  });

  it('runs CoTracker without sending a model selector value', async () => {
    backendMock.trackPromptPoints.mockReturnValue(
      of({
        job_id: 'job-track',
        status: 'queued',
        operation: 'prompt_tracking',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-track',
          operation: 'prompt_tracking',
          status: 'completed',
          stage: 'completed',
          stage_label: 'Completed',
          progress: 1,
          current: 1,
          total: 1,
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
            message: 'Prompt-point tracking completed',
            model_name: 'cotracker3_online',
            num_points: 1,
            num_frames: 2,
            add_support_grid_used: true,
            tracking_mode: 'streaming',
            streaming_frame_threshold: 256,
            tracking_result_id: 'track-1',
            state_epoch: 3,
          },
        },
      }),
    );
    backendMock.getTrackingResult.mockReturnValue(
      of({
        result: {
          version: 1,
          result_id: 'track-1',
          message: 'Prompt-point tracking completed',
          model_name: 'cotracker3_online',
          num_points: 1,
          num_frames: 2,
          add_support_grid_used: true,
          tracking_mode: 'streaming',
          streaming_frame_threshold: 256,
          tracks: [
            [
              [10, 20],
              [11, 21],
            ],
          ],
          visibility: [[true, true]],
          points: [
            {
              point_id: 'p0_0',
              obj_id: 1,
              source_frame_idx: 0,
              source_x: 10,
              source_y: 20,
            },
          ],
        },
      }),
    );
    component.store.trackingUseSupportGrid.set(true);

    await component.runTracking();

    expect(backendMock.trackPromptPoints).toHaveBeenCalledWith({ add_support_grid: true });
    expect(backendMock.trackPromptPoints.mock.calls[0][0]).not.toHaveProperty('model_name');
    expect(backendMock.getTrackingResult).toHaveBeenCalledWith('track-1');
    expect(backendMock.clearJobResult).toHaveBeenCalledWith('job-track');
    expect(component.store.trackedPoints()[0].tracks).toEqual([
      [10, 20],
      [11, 21],
    ]);
  });

  it('enables manifest mask loading after propagation with legacy manifest key', async () => {
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
    component.store.isInitialized.set(true);
    component.store.numFrames.set(2);

    await component.propagate();

    expect(component.store.hasManifestMasks()).toBe(true);
    expect(backendMock.propagateInVideo).toHaveBeenCalledWith({
      include_masks_in_response: false,
      include_saved_mask_paths: false,
    });
  });

  it('does not render a tracking model selector', () => {
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).not.toContain('Tracking Model');
  });

  it('creates an error toast when a job fails', async () => {
    backendMock.initVideoState.mockReturnValue(
      of({
        job_id: 'job-2',
        status: 'queued',
        operation: 'video_init',
        message: 'queued',
      }),
    );
    backendMock.getJob.mockReturnValue(
      of({
        job: {
          job_id: 'job-2',
          operation: 'video_init',
          status: 'failed',
          stage: 'initializing_state',
          stage_label: 'Initializing video state',
          progress: 0.5,
          current: null,
          total: null,
          window_index: null,
          window_count: null,
          frame_idx: null,
          stage_history: [],
          message: 'failed',
          error: { code: 'validation_error', message: 'Path not found', detail: null },
          started_at: 'now',
          updated_at: 'now',
          completed_at: 'now',
          result: null,
        },
      }),
    );
    component.store.videoDir.set('C:/missing');

    await component.initVideo();

    expect(component.store.isInitialized()).toBe(false);
    expect(component.store.toasts()[0].title).toBe('Loading video');
    expect(component.store.toasts()[0].message).toBe('Path not found');
  });

  it('saves the current session with the typed name', async () => {
    backendMock.saveVideoSession.mockReturnValue(
      of({
        message: 'Session saved successfully',
        name: 'review-run',
        saved_path: 'C:/project/backend/saved/review-run',
        state_epoch: 3,
      }),
    );
    component.store.isInitialized.set(true);
    component.store.saveName.set('review-run');

    component.save();
    await Promise.resolve();

    expect(backendMock.saveVideoSession).toHaveBeenCalledWith(
      'review-run',
      expect.objectContaining({
        version: 1,
        objects: expect.any(Array),
        points: expect.any(Array),
        live_masks: expect.any(Array),
      }),
    );
    expect(component.store.toasts()[0].title).toBe('Session saved');
  });
});
