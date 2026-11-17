import { Injectable, signal } from '@angular/core';
import { Stage } from 'konva/lib/Stage';
import { Layer } from 'konva/lib/Layer';
import { Group } from 'konva/lib/Group';
import { Image as CanvasImage } from 'konva/lib/shapes/Image';
import { Circle } from 'konva/lib/shapes/Circle';
import { Line } from 'konva/lib/shapes/Line';
import { MaskObject, Point, TrackedPointSeries } from './video-masker-state.store';
import { CanvasViewMode, TrackingOverlayStyle } from '../state/video-masker-ui.types';
import { createMaskTexture, MaskOverlay } from './mask-texture';

interface Position {
  x: number;
  y: number;
}

/** A prompt point as drawn; one that says which object and position in its list it is can be edited. */
export type CanvasPoint = Point & { objId?: number; index?: number };

/** What the user did to an existing prompt point on the canvas. */
export type PointEdit =
  | { action: 'move'; objId: number; index: number; x: number; y: number }
  | { action: 'delete' | 'flip'; objId: number; index: number };

/** How close, in screen pixels, the pointer has to be to a point to grab it. */
const GRAB_RADIUS = 9;

/** Konva scene and viewport. All scene geometry stays in source-image pixels. */
@Injectable()
export class CanvasViewportService {
  readonly zoom = signal(1);
  readonly ready = signal(false);
  readonly panMode = signal(false);
  readonly spacePressed = signal(false);
  readonly panning = signal(false);
  readonly viewMode = signal<CanvasViewMode>('full');
  private stage: Stage | null = null;
  private scene: Group | null = null;
  private frame: CanvasImage | null = null;
  private masks: Group | null = null;
  private maskTextures = new WeakMap<
    MaskOverlay['source'],
    Map<string, HTMLCanvasElement | null>
  >();
  private annotations: Group | null = null;
  private observer: ResizeObserver | null = null;
  private listeners: AbortController | null = null;
  private width = 0;
  private height = 0;
  private fitMode = true;
  private gesture: {
    id: number;
    start: Position;
    origin: Position;
    pan: boolean;
    moved: boolean;
    button: number;
    point: { objId: number; index: number } | null;
  } | null = null;

  attach(
    host: HTMLDivElement,
    onPoint: (point: Position) => void,
    onPointEdit: (edit: PointEdit) => void = () => {},
  ): void {
    this.detach();
    this.stage = new Stage({
      container: host,
      width: host.clientWidth,
      height: host.clientHeight,
    });
    const layer = new Layer({ listening: false });
    this.scene = new Group();
    this.frame = new CanvasImage({ image: undefined, visible: this.viewMode() === 'full' });
    this.masks = new Group({ name: 'masks' });
    this.annotations = new Group({ visible: this.viewMode() !== 'masks' });
    this.scene.add(this.frame, this.masks, this.annotations);
    layer.add(this.scene);
    this.stage.add(layer);
    this.listeners = new AbortController();
    const options = { signal: this.listeners.signal };
    host.addEventListener(
      'wheel',
      (event) => {
        if (!this.ready()) return;
        event.preventDefault();
        const delta =
          event.deltaY *
          (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? host.clientHeight : 1);
        this.zoomBy(
          Math.exp(-Math.max(-100, Math.min(100, delta)) * 0.002),
          this.clientToStage(event.clientX, event.clientY),
        );
      },
      { ...options, passive: false },
    );
    host.addEventListener(
      'pointerdown',
      (event) => {
        if (!this.ready() || this.gesture || event.button < 0 || event.button > 2) return;
        const pan = event.button === 1 || this.panMode() || this.spacePressed();
        const start = this.clientToStage(event.clientX, event.clientY);
        // An existing point under the pointer is edited: dragged, or removed with the right button.
        const point = pan ? null : this.promptAt(start);
        if (event.button === 2 && !point) return;
        host.focus({ preventScroll: true });
        event.preventDefault();
        this.gesture = {
          id: event.pointerId,
          start,
          origin: this.scene!.position(),
          pan,
          moved: false,
          button: event.button,
          point,
        };
        this.panning.set(pan);
        host.setPointerCapture(event.pointerId);
      },
      options,
    );
    host.addEventListener(
      'pointermove',
      (event) => {
        const gesture = this.gesture;
        if (!gesture) {
          const over = this.ready() && !this.panMode() && !this.spacePressed();
          host.style.cursor =
            over && this.promptAt(this.clientToStage(event.clientX, event.clientY)) ? 'grab' : '';
          return;
        }
        if (gesture.id !== event.pointerId) return;
        const point = this.clientToStage(event.clientX, event.clientY);
        const dx = point.x - gesture.start.x;
        const dy = point.y - gesture.start.y;
        gesture.moved ||= Math.hypot(dx, dy) > 3;
        if (gesture.point && gesture.button === 0 && gesture.moved) {
          // The point follows the pointer; the mask is asked for once, on release.
          this.promptCircle(gesture.point)?.position(this.stageToImage(point));
          this.stage!.batchDraw();
        }
        if (gesture.pan) {
          this.fitMode = false;
          this.scene!.position({ x: gesture.origin.x + dx, y: gesture.origin.y + dy });
          this.stage!.batchDraw();
        }
      },
      options,
    );
    host.addEventListener(
      'pointerup',
      (event) => {
        const gesture = this.gesture;
        if (!gesture || gesture.id !== event.pointerId) return;
        const point = this.clientToImage(event.clientX, event.clientY);
        this.cancelGesture();
        if (gesture.point) {
          if (gesture.button === 2) onPointEdit({ action: 'delete', ...gesture.point });
          else if (gesture.moved)
            onPointEdit({
              action: 'move',
              ...gesture.point,
              ...this.stageToImage(this.clientToStage(event.clientX, event.clientY)),
            });
          else if (event.shiftKey) onPointEdit({ action: 'flip', ...gesture.point });
          return;
        }
        if (!gesture.pan && !gesture.moved && point) onPoint(point);
      },
      options,
    );
    // The right button removes a point, so the browser menu stays closed over one.
    host.addEventListener(
      'contextmenu',
      (event) => {
        if (this.ready() && this.promptAt(this.clientToStage(event.clientX, event.clientY)))
          event.preventDefault();
      },
      options,
    );
    host.addEventListener('pointercancel', () => this.cancelGesture(), options);
    host.addEventListener('lostpointercapture', () => this.cancelGesture(), options);
    host.addEventListener(
      'keydown',
      (event) => {
        if (event.code === 'Space') {
          event.preventDefault();
          this.spacePressed.set(true);
        }
        if (event.key === '+' || event.key === '=') {
          event.preventDefault();
          this.zoomBy(1.2);
        }
        if (event.key === '-') {
          event.preventDefault();
          this.zoomBy(1 / 1.2);
        }
        if (event.key === '0') {
          event.preventDefault();
          this.fit();
        }
      },
      options,
    );
    host.addEventListener(
      'keyup',
      (event) => {
        if (event.code === 'Space') {
          event.preventDefault();
          this.spacePressed.set(false);
        }
      },
      options,
    );
    host.addEventListener(
      'blur',
      () => {
        this.spacePressed.set(false);
        this.cancelGesture();
      },
      options,
    );
    this.observer = new ResizeObserver(() => this.resize(host.clientWidth, host.clientHeight));
    this.observer.observe(host);
  }

  detach(): void {
    this.cancelGesture();
    this.listeners?.abort();
    this.observer?.disconnect();
    this.stage?.destroy();
    this.stage = null;
    this.scene = null;
    this.frame = null;
    this.masks = null;
    this.annotations = null;
    this.reset();
  }

  /** Switch to `mode`, or back to the full view when it is already the one shown. */
  toggleViewMode(mode: Exclude<CanvasViewMode, 'full'>): void {
    this.viewMode.update((current) => (current === mode ? 'full' : mode));
    this.frame?.visible(this.viewMode() === 'full');
    this.annotations?.visible(this.viewMode() !== 'masks');
    this.stage?.batchDraw();
  }

  reset(): void {
    this.cancelGesture();
    this.width = this.height = 0;
    this.fitMode = true;
    this.ready.set(false);
    this.zoom.set(1);
    this.panMode.set(false);
    this.spacePressed.set(false);
    this.frame?.image(undefined);
    this.masks?.destroyChildren();
    this.maskTextures = new WeakMap();
    this.annotations?.destroyChildren();
    this.stage?.batchDraw();
  }

  render(
    image: HTMLImageElement,
    masks: MaskOverlay[],
    points: CanvasPoint[],
    tracks: TrackedPointSeries[],
    objects: MaskObject[],
    style: TrackingOverlayStyle,
    frameIdx: number,
  ): void {
    if (!this.scene || !this.annotations) return;
    const changedSize = this.width !== image.width || this.height !== image.height;
    this.width = image.width;
    this.height = image.height;
    this.scene.clip({ x: 0, y: 0, width: this.width, height: this.height });
    this.frame!.setAttrs({ image, width: this.width, height: this.height });
    this.masks!.destroyChildren();
    for (const mask of masks) {
      let colors = this.maskTextures.get(mask.source);
      if (!colors) {
        colors = new Map();
        this.maskTextures.set(mask.source, colors);
      }
      if (!colors.has(mask.color)) colors.set(mask.color, createMaskTexture(mask));
      const texture = colors.get(mask.color);
      if (texture)
        this.masks!.add(
          new CanvasImage({
            image: texture,
            width: this.width,
            height: this.height,
            name: 'mask',
            objectId: mask.objectId,
          }),
        );
    }
    this.annotations.destroyChildren();
    for (const point of points) {
      this.addPoint(point, point.label === 1 ? '#00ff00' : '#ff0000', 5, {
        name: 'prompt',
        objId: point.objId,
        index: point.index,
      });
    }
    for (const track of tracks) {
      if (frameIdx < 0 || frameIdx >= track.tracks.length) continue;
      const color = objects.find((object) => object.id === track.obj_id)?.color ?? '#ffd54f';
      if (style !== 'point') {
        const start = Math.max(track.source_frame_idx, style === 'short' ? frameIdx - 20 : 0);
        let segment: number[] = [];
        const flush = () => {
          if (segment.length > 2)
            this.annotations!.add(
              new Line({
                points: segment,
                stroke: color,
                strokeWidth: 2,
                strokeScaleEnabled: false,
                opacity: style === 'short' ? 0.75 : 0.55,
              }),
            );
          segment = [];
        };
        for (let i = start; i <= frameIdx; i++) {
          if (track.visibility[i] === false) {
            flush();
            continue;
          }
          segment.push(...track.tracks[i]);
        }
        flush();
      }
      // Before the frame a point was placed on there is no track, only the placed position.
      if (frameIdx >= track.source_frame_idx && track.visibility[frameIdx] !== false) {
        const [x, y] = track.tracks[frameIdx];
        this.addPoint({ x, y }, color, 4.5);
      }
    }
    this.ready.set(true);
    if (changedSize) this.fit();
    else this.updateMarkerSizes();
    this.stage!.batchDraw();
  }

  private addPoint(point: Position, color: string, radius: number, attrs: object = {}): void {
    this.annotations!.add(
      new Circle({
        ...attrs,
        x: point.x,
        y: point.y,
        radius,
        screenRadius: radius,
        fill: color,
        stroke: 'white',
        strokeWidth: 1.5,
        strokeScaleEnabled: false,
      }),
    );
  }

  private get fitScale(): number {
    if (!this.stage || !this.width || !this.height) return 1;
    return Math.min(
      Math.max(1, this.stage.width() - 24) / this.width,
      Math.max(1, this.stage.height() - 24) / this.height,
      1,
    );
  }

  fit(): void {
    if (!this.stage || !this.scene || !this.ready()) return;
    this.fitMode = true;
    const scale = this.fitScale;
    this.setTransform(scale, {
      x: (this.stage.width() - this.width * scale) / 2,
      y: (this.stage.height() - this.height * scale) / 2,
    });
  }

  zoomBy(factor: number, anchor?: Position): void {
    if (!this.stage || !this.scene || !this.ready() || !Number.isFinite(factor) || factor <= 0)
      return;
    const center = anchor ?? { x: this.stage.width() / 2, y: this.stage.height() / 2 };
    const imagePoint = this.scene.getTransform().copy().invert().point(center);
    const scale = Math.max(Math.min(0.05, this.fitScale), Math.min(32, this.zoom() * factor));
    this.fitMode = false;
    this.setTransform(scale, {
      x: center.x - imagePoint.x * scale,
      y: center.y - imagePoint.y * scale,
    });
  }

  resize(width: number, height: number): void {
    if (!this.stage || !this.scene || width <= 0 || height <= 0) return;
    const dx = (width - this.stage.width()) / 2;
    const dy = (height - this.stage.height()) / 2;
    this.stage.size({ width, height });
    if (this.fitMode) this.fit();
    else this.setTransform(this.zoom(), { x: this.scene.x() + dx, y: this.scene.y() + dy });
  }

  clientToImage(clientX: number, clientY: number): Position | null {
    if (!this.scene || !this.ready()) return null;
    const point = this.scene
      .getTransform()
      .copy()
      .invert()
      .point(this.clientToStage(clientX, clientY));
    return point.x >= 0 && point.y >= 0 && point.x < this.width && point.y < this.height
      ? point
      : null;
  }

  /** The editable prompt point nearest to a stage position, if one is within reach. */
  private promptAt(stagePoint: Position): { objId: number; index: number } | null {
    if (!this.annotations?.isVisible()) return null;
    let nearest: { objId: number; index: number } | null = null;
    let reach = GRAB_RADIUS;
    for (const circle of this.annotations.find<Circle>('.prompt')) {
      const objId = circle.getAttr('objId');
      const index = circle.getAttr('index');
      if (typeof objId !== 'number' || typeof index !== 'number') continue;
      const at = circle.getAbsolutePosition();
      const distance = Math.hypot(at.x - stagePoint.x, at.y - stagePoint.y);
      if (distance <= reach) {
        reach = distance;
        nearest = { objId, index };
      }
    }
    return nearest;
  }

  private promptCircle(point: { objId: number; index: number }): Circle | undefined {
    return this.annotations
      ?.find<Circle>('.prompt')
      .find(
        (circle) =>
          circle.getAttr('objId') === point.objId && circle.getAttr('index') === point.index,
      );
  }

  /** A stage position in image pixels, kept inside the image. */
  private stageToImage(stagePoint: Position): Position {
    const point = this.scene!.getTransform().copy().invert().point(stagePoint);
    return {
      x: Math.min(Math.max(point.x, 0), Math.max(0, this.width - 1)),
      y: Math.min(Math.max(point.y, 0), Math.max(0, this.height - 1)),
    };
  }

  private clientToStage(clientX: number, clientY: number): Position {
    const rect = this.stage!.container().getBoundingClientRect();
    return {
      x: ((clientX - rect.left) * this.stage!.width()) / (rect.width || 1),
      y: ((clientY - rect.top) * this.stage!.height()) / (rect.height || 1),
    };
  }

  private setTransform(scale: number, position: Position): void {
    this.scene!.scale({ x: scale, y: scale });
    this.scene!.position(position);
    this.zoom.set(scale);
    this.updateMarkerSizes();
    this.stage!.batchDraw();
  }

  private updateMarkerSizes(): void {
    this.annotations
      ?.find<Circle>('Circle')
      .forEach((circle) => circle.radius(circle.getAttr('screenRadius') / this.zoom()));
  }

  private cancelGesture(): void {
    const gesture = this.gesture;
    this.gesture = null;
    const host = this.stage?.container();
    if (gesture && host?.hasPointerCapture(gesture.id)) host.releasePointerCapture(gesture.id);
    this.panning.set(false);
  }
}
