import { TestBed } from '@angular/core/testing';
import Konva from 'konva';
import { vi } from 'vitest';
import { CanvasViewportService } from './canvas-viewport.service';
import { MaskOverlay } from './mask-texture';

describe('CanvasViewportService', () => {
  let viewport: CanvasViewportService;
  let host: HTMLDivElement;
  let stage: Konva.Stage;
  let onPoint = vi.fn<(point: { x: number; y: number }) => void>();
  let image: HTMLImageElement;
  let masks: MaskOverlay[];
  let disconnect: ReturnType<typeof vi.fn>;
  let resizeCallback: () => void;
  let width: number;
  let height: number;

  beforeEach(() => {
    // Exercise real Konva nodes and transforms; only the browser raster API is stubbed.
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
      this: HTMLCanvasElement,
    ) {
      return new Proxy(
        {
          canvas: this,
          createImageData: (width: number, height: number) => ({
            data: new Uint8ClampedArray(width * height * 4),
          }),
          getImageData: (_x: number, _y: number, width: number, height: number) => ({
            data: new Uint8ClampedArray(width * height * 4),
          }),
        },
        {
          get(target, key) {
            return key in target ? target[key as keyof typeof target] : vi.fn();
          },
        },
      ) as unknown as CanvasRenderingContext2D;
    });
    disconnect = vi.fn();
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: () => void) {
          resizeCallback = callback;
        }
        observe = vi.fn();
        disconnect = disconnect;
      },
    );
    TestBed.configureTestingModule({ providers: [CanvasViewportService] });
    viewport = TestBed.inject(CanvasViewportService);
    host = document.createElement('div');
    host.tabIndex = 0;
    document.body.appendChild(host);
    width = 1024;
    height = 524;
    Object.defineProperties(host, {
      clientWidth: { get: () => width },
      clientHeight: { get: () => height },
    });
    vi.spyOn(host, 'getBoundingClientRect').mockImplementation(
      () => ({ left: 20, top: 30, width, height }) as DOMRect,
    );
    host.setPointerCapture = vi.fn();
    host.releasePointerCapture = vi.fn();
    host.hasPointerCapture = vi.fn(() => true);
    onPoint = vi.fn();
    viewport.attach(host, onPoint);
    stage = Konva.stages[Konva.stages.length - 1];
    image = document.createElement('img');
    image.width = 1000;
    image.height = 500;
    masks = [{ objectId: 1, color: '#00aaff', source: [[true, false]] }];
    render();
  });

  afterEach(() => {
    viewport.detach();
    host.remove();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  function render(): void {
    viewport.render(image, masks, [{ x: 100, y: 50, label: 1 }], [], [], 'point', 0);
  }

  it('composes individual mask nodes with cached native-resolution textures', () => {
    const first = stage.findOne<Konva.Image>('.mask')!;
    const texture = first.image() as HTMLCanvasElement;
    expect(texture.width).toBe(2);
    expect(texture.height).toBe(1);
    expect(first.width()).toBe(1000);
    expect(first.height()).toBe(500);
    masks.push({ objectId: 2, color: '#ffaa00', source: [[false, true]] });
    render();
    expect(stage.find('.mask')).toHaveLength(2);
    expect(stage.findOne<Konva.Image>('.mask')!.image()).toBe(texture);
    masks = [masks[1]];
    render();
    expect(stage.find('.mask')).toHaveLength(1);
    expect(stage.findOne('.mask')!.getAttr('objectId')).toBe(2);
  });

  function pointer(type: string, x: number, y: number, button = 0): void {
    const event = new MouseEvent(type, { clientX: x, clientY: y, button, bubbles: true });
    Object.defineProperty(event, 'pointerId', { value: 1 });
    host.dispatchEvent(event);
  }

  it('uses separate Konva images and vector markers in source-image coordinates', () => {
    expect(stage.find('Image')).toHaveLength(2);
    const circle = stage.findOne<Konva.Circle>('Circle')!;
    expect(circle.position()).toEqual({ x: 100, y: 50 });
    expect(circle.fill()).toBe('#00ff00');
    expect(viewport.clientToImage(132, 92)).toEqual({ x: 100, y: 50 });
    expect(viewport.clientToImage(20, 30)).toBeNull();
  });

  it('zooms around the pointer while keeping markers a constant screen size', () => {
    viewport.zoomBy(2, { x: 112, y: 62 });
    expect(viewport.zoom()).toBe(2);
    expect(viewport.clientToImage(132, 92)).toEqual({ x: 100, y: 50 });
    expect(stage.findOne<Konva.Circle>('Circle')!.radius()).toBe(2.5);
    viewport.zoomBy(0.5, { x: 112, y: 62 });
    expect(viewport.zoom()).toBe(1);
    expect(viewport.clientToImage(132, 92)).toEqual({ x: 100, y: 50 });
  });

  it('handles wheel zoom without scrolling the page or rebuilding scene nodes', () => {
    const circle = stage.findOne('Circle');
    const event = new WheelEvent('wheel', {
      clientX: 132,
      clientY: 92,
      deltaY: -100,
      cancelable: true,
    });
    host.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(viewport.zoom()).toBeGreaterThan(1);
    expect(stage.findOne('Circle')).toBe(circle);
    const point = viewport.clientToImage(132, 92)!;
    expect(point.x).toBeCloseTo(100);
    expect(point.y).toBeCloseTo(50);
  });

  it('pans with the middle button and does not add a point on release', () => {
    pointer('pointerdown', 132, 92, 1);
    pointer('pointermove', 172, 102, 1);
    pointer('pointerup', 172, 102, 1);
    expect(viewport.clientToImage(172, 102)).toEqual({ x: 100, y: 50 });
    expect(onPoint).not.toHaveBeenCalled();
    expect(viewport.panning()).toBe(false);
  });

  it('supports Space-drag and Pan mode without placing prompts', () => {
    host.dispatchEvent(new KeyboardEvent('keydown', { code: 'Space' }));
    pointer('pointerdown', 132, 92);
    pointer('pointermove', 172, 102);
    pointer('pointerup', 172, 102);
    host.dispatchEvent(new KeyboardEvent('keyup', { code: 'Space' }));
    expect(viewport.spacePressed()).toBe(false);
    viewport.panMode.set(true);
    pointer('pointerdown', 172, 102);
    pointer('pointerup', 172, 102);
    expect(onPoint).not.toHaveBeenCalled();
  });

  it('submits image coordinates after zoom and rejects background clicks and drags', () => {
    viewport.zoomBy(2, { x: 112, y: 62 });
    pointer('pointerdown', 132, 92);
    pointer('pointerup', 132, 92);
    expect(onPoint).toHaveBeenCalledWith({ x: 100, y: 50 });
    viewport.fit();
    pointer('pointerdown', 20, 30);
    pointer('pointerup', 20, 30);
    pointer('pointerdown', 132, 92);
    pointer('pointermove', 172, 102);
    pointer('pointerup', 172, 102);
    expect(onPoint).toHaveBeenCalledTimes(1);
  });

  it('preserves the viewport across frames and manual resizing, and refits on demand', () => {
    viewport.zoomBy(2);
    const center = viewport.clientToImage(532, 292);
    render();
    expect(viewport.zoom()).toBe(2);
    width = 500;
    height = 300;
    resizeCallback();
    expect(viewport.clientToImage(270, 180)).toEqual(center);
    viewport.fit();
    expect(viewport.zoom()).toBeCloseTo(0.476);
    width = 1024;
    height = 524;
    resizeCallback();
    expect(viewport.zoom()).toBe(1);
  });

  it('bounds zoom and resets the viewport for a new session', () => {
    viewport.zoomBy(1e9);
    expect(viewport.zoom()).toBe(32);
    viewport.zoomBy(1e-9);
    expect(viewport.zoom()).toBe(0.05);
    viewport.reset();
    expect(viewport.ready()).toBe(false);
    render();
    expect(viewport.zoom()).toBe(1);
  });

  it('removes Konva nodes, observers, and input handlers on detach', () => {
    viewport.detach();
    expect(disconnect).toHaveBeenCalled();
    expect(host.querySelector('canvas')).toBeNull();
    expect(viewport.ready()).toBe(false);
    pointer('pointerdown', 132, 92);
    pointer('pointerup', 132, 92);
    expect(onPoint).not.toHaveBeenCalled();
  });
});
