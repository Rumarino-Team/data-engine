import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { vi } from 'vitest';
import { BackendService, VideoAddPointsResponse } from '../services/backend.service';
import { DesktopBridgeService } from '../services/desktop-bridge.service';
import { VideoMaskerComponent } from './video-masker.component';

describe('VideoMaskerComponent sync contract', () => {
	let component: VideoMaskerComponent;
	let backendMock: {
		addNewPointsOrBox: ReturnType<typeof vi.fn>;
		health: ReturnType<typeof vi.fn>;
		initVideoState: ReturnType<typeof vi.fn>;
		getJob: ReturnType<typeof vi.fn>;
		getApiUrl: ReturnType<typeof vi.fn>;
		setApiUrl: ReturnType<typeof vi.fn>;
		resetApiUrl: ReturnType<typeof vi.fn>;
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
		out_masks: [[[true, false], [false, false]]],
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
			getApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
			setApiUrl: vi.fn((value: string) => value),
			resetApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
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
						getApiUrl: backendMock.getApiUrl,
						setApiUrl: backendMock.setApiUrl,
						resetApiUrl: backendMock.resetApiUrl,
					},
				},
				{
					provide: DesktopBridgeService,
					useValue: desktopBridgeMock,
				},
			],
		}).compileComponents();

		const fixture = TestBed.createComponent(VideoMaskerComponent);
		component = fixture.componentInstance;
		component.selectedObjectId.set(1);
		component.objects.set([{ id: 1, name: 'Object 1', color: '#ff0000' }]);
		component.stateEpoch.set(3);
		component.displayedFrameIdx.set(5);
	});

	it('uses displayed frame index in request and stores mask on that frame', async () => {
		backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({})));

		await component.addPoint(12, 24, 1, 5);

		expect(backendMock.addNewPointsOrBox).toHaveBeenCalledWith(
			expect.objectContaining({ frame_idx: 5 }),
		);
		expect(component.masks().get(5)?.get(1)).toEqual([[true, false], [false, false]]);
		expect(component.lastDiscardReason()).toBeNull();
	});

	it('discards mismatched response frame and rolls back optimistic point', async () => {
		backendMock.addNewPointsOrBox.mockReturnValue(
			of(makeResponse({ request_frame_idx: 5, frame_idx: 4 })),
		);

		await component.addPoint(10, 20, 1, 5);

		expect(component.masks().get(5)?.get(1)).toBeUndefined();
		expect(component.points().get(5)?.get(1)?.length ?? 0).toBe(0);
		expect(component.lastDiscardReason()).toContain('frame mismatch');
	});

	it('discards stale epoch responses and clears live masks', async () => {
		const existingMasks = new Map<number, Map<number, boolean[][]>>();
		existingMasks.set(2, new Map([[1, [[true]]]]));
		component.masks.set(existingMasks);
		component.liveEditedObjectFrames.set(new Map([[2, new Set([1])]]));

		backendMock.addNewPointsOrBox.mockReturnValue(of(makeResponse({ state_epoch: 4 })));

		await component.addPoint(14, 18, 1, 5);

		expect(component.stateEpoch()).toBe(4);
		expect(component.masks().size).toBe(0);
		expect(component.liveEditedObjectFrames().size).toBe(0);
		expect(component.lastDiscardReason()).toContain('epoch mismatch');
	});

	it('keeps frame/object marked as live-edited even when returned mask is empty', async () => {
		backendMock.addNewPointsOrBox.mockReturnValue(
			of(
				makeResponse({
					out_masks: [[[false, false], [false, false]]],
					mask_pixel_counts: { 1: 0 },
				}),
			),
		);

		await component.addPoint(30, 40, 1, 5);

		expect(component.liveEditedObjectFrames().get(5)?.has(1)).toBe(true);
		expect(component.lastMaskPixelCount()).toBe(0);
	});

	it('uses the native Tauri video picker when desktop runtime is available', async () => {
		desktopBridgeMock.isTauri.mockReturnValue(true);
		desktopBridgeMock.pickVideoFile.mockResolvedValue('C:/videos/example.mp4');

		await component.browseVideo();

		expect(desktopBridgeMock.pickVideoFile).toHaveBeenCalled();
		expect(component.videoDir()).toBe('C:/videos/example.mp4');
	});

	it('falls back to browser file input when Tauri runtime is unavailable', async () => {
		const pickerSpy = vi.spyOn(component, 'openVideoFilePicker').mockImplementation(() => undefined);

		await component.browseVideo();

		expect(pickerSpy).toHaveBeenCalled();
	});

	it('starts a video init job, polls completion, and applies the result', async () => {
		backendMock.initVideoState.mockReturnValue(of({
			job_id: 'job-1',
			status: 'queued',
			operation: 'video_init',
			message: 'queued',
		}));
		backendMock.getJob.mockReturnValue(of({
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
		}));
		component.videoDir.set('C:/frames');

		await component.initVideo();

		expect(backendMock.initVideoState).toHaveBeenCalledWith('C:/frames');
		expect(backendMock.getJob).toHaveBeenCalledWith('job-1');
		expect(component.isInitialized()).toBe(true);
		expect(component.numFrames()).toBe(12);
		expect(component.stateEpoch()).toBe(7);
	});

	it('creates an error toast when a job fails', async () => {
		backendMock.initVideoState.mockReturnValue(of({
			job_id: 'job-2',
			status: 'queued',
			operation: 'video_init',
			message: 'queued',
		}));
		backendMock.getJob.mockReturnValue(of({
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
		}));
		component.videoDir.set('C:/missing');

		await component.initVideo();

		expect(component.isInitialized()).toBe(false);
		expect(component.toasts()[0].title).toBe('Loading video');
		expect(component.toasts()[0].message).toBe('Path not found');
	});
});
