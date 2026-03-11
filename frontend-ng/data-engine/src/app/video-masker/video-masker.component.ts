import { Component, ElementRef, ViewChild, signal, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';
import { BackendService, VideoAddPointsOrBoxRequest } from '../services/backend.service';

interface MaskObject {
	id: number;
	name: string;
	color: string;
}

interface Point {
	x: number;
	y: number;
	label: number; // 1 for positive, 0 for negative
}

@Component({
	selector: 'app-video-masker',
	standalone: true,
	imports: [CommonModule, FormsModule],
	templateUrl: './video-masker.component.html',
	styleUrls: ['./video-masker.component.css']
})
export class VideoMaskerComponent {
	@ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;

	videoDir = signal<string>('');
	isInitialized = signal<boolean>(false);
	numFrames = signal<number>(0);
	currentFrameIdx = signal<number>(0);

	objects = signal<MaskObject[]>([]);
	selectedObjectId = signal<number | null>(null);

	// Interaction mode
	interactionMode = signal<'positive' | 'negative'>('positive');

	// State for visualization
	// frameIdx -> objId -> mask (boolean[][])
	masks = signal<Map<number, Map<number, boolean[][]>>>(new Map());

	// frameIdx -> objId -> points
	points = signal<Map<number, Map<number, Point[]>>>(new Map());
	useSavedMaskFrames = signal<boolean>(false);

	isLoading = signal<boolean>(false);

	constructor(private backend: BackendService) {
		effect(() => {
			if (this.isInitialized()) {
				this.loadFrame(this.currentFrameIdx());
			}
		});
	}

	async initVideo() {
		const enteredPath = this.videoDir().trim().replace(/^['\"]|['\"]$/g, '');
		if (!enteredPath) {
			alert('Please enter a valid video frames directory path.');
			return;
		}

		this.videoDir.set(enteredPath);
		this.isLoading.set(true);
		try {
			const res = await firstValueFrom(this.backend.initVideoState(enteredPath));
			this.numFrames.set(res.num_frames);
			this.isInitialized.set(true);
			this.currentFrameIdx.set(0);
			this.useSavedMaskFrames.set(false);
			this.objects.set([{ id: 1, name: 'Object 1', color: this.getRandomColor() }]);
			this.selectedObjectId.set(1);
		} catch (err: any) {
			console.error(err);
			const errorMessage = err?.error?.detail || err?.error?.error || 'Failed to initialize video';
			alert(errorMessage);
		} finally {
			this.isLoading.set(false);
		}
	}

	loadFrame(frameIdx: number) {
		const ctx = this.canvasRef.nativeElement.getContext('2d');
		if (!ctx) return;

		const img = new Image();
		const frameUrl = this.backend.getVideoFrameUrl(frameIdx);
		const maskFrameUrl = this.backend.getVideoMaskFrameUrl(frameIdx);
		let triedFallbackToRawFrame = false;

		img.onerror = () => {
			if (this.useSavedMaskFrames() && !triedFallbackToRawFrame) {
				triedFallbackToRawFrame = true;
				img.src = frameUrl;
			}
		};

		img.onload = () => {
			this.canvasRef.nativeElement.width = img.width;
			this.canvasRef.nativeElement.height = img.height;
			this.draw(img);
		};

		img.src = this.useSavedMaskFrames() ? maskFrameUrl : frameUrl;
	}

	draw(img: HTMLImageElement) {
		const canvas = this.canvasRef.nativeElement;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		// Clear canvas
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		// Draw image
		ctx.drawImage(img, 0, 0);

		// Draw masks
		const frameMasks = this.masks().get(this.currentFrameIdx());
		if (frameMasks) {
			frameMasks.forEach((mask, objId) => {
				const obj = this.objects().find(o => o.id === objId);
				if (obj) {
					this.drawMask(ctx, mask, obj.color);
				}
			});
		}

		// Draw points
		const framePoints = this.points().get(this.currentFrameIdx());
		if (framePoints) {
			framePoints.forEach((points, objId) => {
				const obj = this.objects().find(o => o.id === objId);
				if (obj) {
					points.forEach(p => {
						this.drawPoint(ctx, p, obj.color);
					});
				}
			});
		}
	}

	drawMask(ctx: CanvasRenderingContext2D, mask: boolean[][], color: string) {
		const width = mask[0].length;
		const height = mask.length;

		// Create an ImageData object
		const imageData = ctx.createImageData(width, height);
		const data = imageData.data;

		const [r, g, b] = this.hexToRgb(color);

		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				if (mask[y][x]) {
					const index = (y * width + x) * 4;
					data[index] = r;     // R
					data[index + 1] = g; // G
					data[index + 2] = b; // B
					data[index + 3] = 128; // Alpha (0-255)
				}
			}
		}

		// Create a temporary canvas to put the image data
		const tempCanvas = document.createElement('canvas');
		tempCanvas.width = width;
		tempCanvas.height = height;
		tempCanvas.getContext('2d')!.putImageData(imageData, 0, 0);

		// Draw the temporary canvas onto the main canvas
		// We need to scale it if the main canvas size is different from mask size (should be same)
		ctx.drawImage(tempCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
	}

	drawPoint(ctx: CanvasRenderingContext2D, point: Point, color: string) {
		ctx.beginPath();
		ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
		ctx.fillStyle = point.label === 1 ? color : 'red'; // Positive: object color, Negative: red (or maybe white/black?)
		// Actually, usually positive is green, negative is red. But here we have multiple objects.
		// Let's say positive is object color, negative is black with object color border?
		// For simplicity: Positive = Green, Negative = Red.
		ctx.fillStyle = point.label === 1 ? '#00FF00' : '#FF0000';
		ctx.fill();
		ctx.strokeStyle = 'white';
		ctx.lineWidth = 2;
		ctx.stroke();
	}

	onCanvasClick(event: MouseEvent) {
		if (!this.isInitialized() || this.selectedObjectId() === null) return;

		const rect = this.canvasRef.nativeElement.getBoundingClientRect();
		const scaleX = this.canvasRef.nativeElement.width / rect.width;
		const scaleY = this.canvasRef.nativeElement.height / rect.height;

		const x = (event.clientX - rect.left) * scaleX;
		const y = (event.clientY - rect.top) * scaleY;

		const label = this.interactionMode() === 'positive' ? 1 : 0;

		this.addPoint(x, y, label);
	}

	async addPoint(x: number, y: number, label: number) {
		const objId = this.selectedObjectId();
		if (objId === null) return;

		// Update local points state
		const currentPointsMap = this.points();
		let framePointsMap = currentPointsMap.get(this.currentFrameIdx());
		if (!framePointsMap) {
			framePointsMap = new Map();
			currentPointsMap.set(this.currentFrameIdx(), framePointsMap);
		}
		let objPoints = framePointsMap.get(objId);
		if (!objPoints) {
			objPoints = [];
			framePointsMap.set(objId, objPoints);
		}
		objPoints.push({ x, y, label });
		this.points.set(new Map(currentPointsMap)); // Trigger signal update

		// Call backend
		// We need to send ALL points for this object on this frame, or just the new one?
		// The API `add_new_points_or_box` takes `points` list.
		// Usually SAM2 expects all points for the current interaction?
		// The API has `clear_old_points=True` by default.
		// If we want to accumulate points, we should probably send all of them, or set `clear_old_points=False`.
		// Let's try sending just the new point with `clear_old_points=False`.

		const request: VideoAddPointsOrBoxRequest = {
			frame_idx: this.currentFrameIdx(),
			obj_id: objId,
			points: [[x, y]],
			labels: [label],
			clear_old_points: false
		};

		try {
			const res = await firstValueFrom(this.backend.addNewPointsOrBox(request));
			if (res) {
				// Update masks
				const currentMasksMap = this.masks();
				let frameMasksMap = currentMasksMap.get(this.currentFrameIdx());
				if (!frameMasksMap) {
					frameMasksMap = new Map();
					currentMasksMap.set(this.currentFrameIdx(), frameMasksMap);
				}

				// res.out_masks is a list of masks corresponding to out_obj_ids
				res.out_obj_ids.forEach((id, index) => {
					frameMasksMap!.set(id, res.out_masks[index]);
				});

				this.masks.set(new Map(currentMasksMap));

				// Redraw
				// We need to reload the image to clear and redraw everything
				this.loadFrame(this.currentFrameIdx());
			}
		} catch (err) {
			console.error(err);
		}
	}

	addObject() {
		const newId = this.objects().length + 1;
		const newObj: MaskObject = {
			id: newId,
			name: `Object ${newId}`,
			color: this.getRandomColor()
		};
		this.objects.update(objs => [...objs, newObj]);
		this.selectedObjectId.set(newId);
	}

	removeObject() {
		const id = this.selectedObjectId();
		if (id === null) return;

		this.backend.removeObject(id).subscribe(() => {
			this.objects.update(objs => objs.filter(o => o.id !== id));
			if (this.objects().length > 0) {
				this.selectedObjectId.set(this.objects()[0].id);
			} else {
				this.selectedObjectId.set(null);
			}
			// Also clear masks and points for this object
			// ... implementation omitted for brevity, but should be done
		});
	}

	async propagate() {
		this.isLoading.set(true);
		try {
			const res = await firstValueFrom(this.backend.propagateInVideo({
				include_masks_in_response: false
			}));
			if (res && res.video_segments) {
				// Update all masks
				const currentMasksMap = this.masks();
				const entries = Object.entries(res.video_segments);

				for (const [frameIdxStr, objMasks] of entries) {
					const frameIdx = parseInt(frameIdxStr);
					let frameMasksMap = currentMasksMap.get(frameIdx);
					if (!frameMasksMap) {
						frameMasksMap = new Map();
						currentMasksMap.set(frameIdx, frameMasksMap);
					}

					for (const [objIdStr, mask] of Object.entries(objMasks)) {
						const objId = parseInt(objIdStr);
						frameMasksMap.set(objId, mask as boolean[][]);
					}
				}

				const hasSavedMaskFrames = Object.keys(res.saved_mask_paths || {}).length > 0;
				const useSavedFrames = hasSavedMaskFrames && entries.length === 0;
				this.useSavedMaskFrames.set(useSavedFrames);

				if (useSavedFrames) {
					this.masks.set(new Map());
				} else {
					this.masks.set(new Map(currentMasksMap));
				}

				this.loadFrame(this.currentFrameIdx()); // Redraw current frame
			}
		} catch (err) {
			console.error(err);
			alert('Propagation failed');
		} finally {
			this.isLoading.set(false);
		}
	}

	clearMasks() {
		// This should probably call reset_state on backend
		this.backend.resetVideoState().subscribe(() => {
			this.useSavedMaskFrames.set(false);
			this.masks.set(new Map());
			this.points.set(new Map());
			this.loadFrame(this.currentFrameIdx());
		});
	}

	save() {
		alert('Save functionality not implemented yet.');
	}

	getRandomColor() {
		const letters = '0123456789ABCDEF';
		let color = '#';
		for (let i = 0; i < 6; i++) {
			color += letters[Math.floor(Math.random() * 16)];
		}
		return color;
	}

	hexToRgb(hex: string) {
		const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
		return result ? [
			parseInt(result[1], 16),
			parseInt(result[2], 16),
			parseInt(result[3], 16)
		] : [0, 0, 0];
	}
}
