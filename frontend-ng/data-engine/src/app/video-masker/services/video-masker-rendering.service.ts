import { Injectable } from '@angular/core';
import { VideoMaskObjectData } from '../../services/backend.service';
import { MaskObject, Point, TrackedPointSeries } from './video-masker-state.store';

@Injectable({ providedIn: 'root' })
export class VideoMaskerRenderingService {
  drawMask(ctx: CanvasRenderingContext2D, mask: boolean[][], color: string): void {
    const normalizedMask = this.normalizeMask2d(mask);
    if (!normalizedMask?.length || !normalizedMask[0]?.length) {
      return;
    }
    const width = normalizedMask[0].length;
    const height = normalizedMask.length;
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
    const [r, g, b] = this.hexToRgb(color);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (!normalizedMask[y][x]) {
          continue;
        }
        const index = (y * width + x) * 4;
        data[index] = r;
        data[index + 1] = g;
        data[index + 2] = b;
        data[index + 3] = 120;
      }
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
    ctx.drawImage(tempCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  normalizeMask2d(mask: unknown): boolean[][] | null {
    let candidate: unknown = mask;
    while (
      Array.isArray(candidate) &&
      candidate.length > 0 &&
      Array.isArray(candidate[0]) &&
      Array.isArray((candidate[0] as unknown[])[0])
    ) {
      candidate = candidate[0];
    }
    if (!Array.isArray(candidate) || candidate.length === 0 || !Array.isArray(candidate[0])) {
      return null;
    }
    return candidate as boolean[][];
  }

  maskHasForeground(mask: boolean[][]): boolean {
    for (const row of mask) {
      for (const value of row) {
        if (value) {
          return true;
        }
      }
    }
    return false;
  }

  drawMaskFromRle(ctx: CanvasRenderingContext2D, maskData: VideoMaskObjectData, color: string): void {
    const size = maskData.size;
    if (!Array.isArray(size) || size.length !== 2) {
      return;
    }
    const height = Number(size[0]);
    const width = Number(size[1]);
    if (!Number.isFinite(height) || !Number.isFinite(width) || height <= 0 || width <= 0) {
      return;
    }

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
    const [r, g, b] = this.hexToRgb(color);

    for (const run of maskData.rle || []) {
      if (!Array.isArray(run) || run.length !== 2) {
        continue;
      }
      const start = Math.max(0, Number(run[0]) | 0);
      const length = Math.max(0, Number(run[1]) | 0);
      const end = Math.min(width * height, start + length);
      for (let index = start; index < end; index++) {
        const pixelOffset = index * 4;
        data[pixelOffset] = r;
        data[pixelOffset + 1] = g;
        data[pixelOffset + 2] = b;
        data[pixelOffset + 3] = 120;
      }
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
    ctx.drawImage(tempCanvas, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  drawPoint(ctx: CanvasRenderingContext2D, point: Point): void {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = point.label === 1 ? '#00ff00' : '#ff0000';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  drawTrackingOverlay(
    ctx: CanvasRenderingContext2D,
    frameIdx: number,
    style: 'point' | 'short' | 'full',
    trackedPoints: TrackedPointSeries[],
    objects: MaskObject[],
  ): void {
    if (!trackedPoints.length) {
      return;
    }

    for (const pointSeries of trackedPoints) {
      if (frameIdx < 0 || frameIdx >= pointSeries.tracks.length) {
        continue;
      }

      const obj = objects.find((candidate) => candidate.id === pointSeries.obj_id);
      const color = obj?.color || '#ffd54f';
      const visibleNow = pointSeries.visibility[frameIdx] !== false;

      if (style !== 'point') {
        const trailStart =
          style === 'short'
            ? Math.max(pointSeries.source_frame_idx, frameIdx - 20)
            : Math.max(pointSeries.source_frame_idx, 0);

        ctx.beginPath();
        let started = false;
        for (let idx = trailStart; idx <= frameIdx; idx++) {
          if (idx < 0 || idx >= pointSeries.tracks.length || pointSeries.visibility[idx] === false) {
            continue;
          }
          const [x, y] = pointSeries.tracks[idx];
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = style === 'short' ? 0.75 : 0.55;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      if (!visibleNow) {
        continue;
      }

      const [currentX, currentY] = pointSeries.tracks[frameIdx];
      ctx.beginPath();
      ctx.arc(currentX, currentY, 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  private hexToRgb(hex: string): [number, number, number] {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
      : [0, 0, 0];
  }
}
