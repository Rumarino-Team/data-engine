import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { vi } from 'vitest';
import { BackendService, VideoAddPointsResponse } from '../services/backend.service';
import { VideoMaskerComponent } from './video-masker.component';

describe('VideoMaskerComponent sync contract', () => {
	let component: VideoMaskerComponent;
	let backendMock: {
		addNewPointsOrBox: ReturnType<typeof vi.fn>;
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
		};

		await TestBed.configureTestingModule({
			imports: [VideoMaskerComponent],
			providers: [
				{
					provide: BackendService,
					useValue: {
						addNewPointsOrBox: backendMock.addNewPointsOrBox,
					},
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
});
