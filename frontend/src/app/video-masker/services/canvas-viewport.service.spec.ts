import { TestBed } from '@angular/core/testing';
import Konva from 'konva';
import { vi } from 'vitest';
import { CanvasViewportService, PointEdit } from './canvas-viewport.service';
import { MaskOverlay } from './mask-texture';

describe('CanvasViewportService', () => {
  let viewport: CanvasViewportService;
  let host: HTMLDivElement;
  let stage: Konva.Stage;
  let onPoint = vi.fn<(point: { x: number; y: number }) => void>();
  let onPointEdit = vi.fn<(edit: PointEdit) => void>();
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
    onPointEdit = vi.fn();
    viewport.attach(host, onPoint, onPointEdit);
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

  function pointer(type: string, x: number, y: number, button = 0, shiftKey = false): void {
    const event = new MouseEvent(type, { clientX: x, clientY: y, button, shiftKey, bubbles: true });
    Object.defineProperty(event, 'pointerId', { value: 1 });
    host.dispatchEvent(event);
  }

  describe('editing a point that is already placed', () => {
    // The point of object 7 sits at image (100, 50), which is client (132, 92) in this layout.
    beforeEach(() => {
      viewport.render(
        image,
        masks,
        [{ x: 100, y: 50, label: 1, objId: 7, index: 0 }],
        [],
        [],
        'point',
        0,
      );
    });

    it('moves it by dragging: it follows the pointer and is reported once, on release', () => {
      pointer('pointerdown', 132, 92);
      pointer('pointermove', 172, 102);
      expect(stage.findOne<Konva.Circle>('.prompt')!.position()).toEqual({ x: 140, y: 60 });
      expect(onPointEdit).not.toHaveBeenCalled();
      pointer('pointerup', 172, 102);
      expect(onPointEdit).toHaveBeenCalledExactlyOnceWith({
        action: 'move',
        objId: 7,
        index: 0,
        x: 140,
        y: 60,
      });
      expect(onPoint).not.toHaveBeenCalled();
    });

    it('keeps a point dragged past the edge inside the image', () => {
      pointer('pointerdown', 132, 92);
      pointer('pointermove', 5, 5);
      pointer('pointerup', 5, 5);
      expect(onPointEdit).toHaveBeenCalledWith(
        expect.objectContaining({ action: 'move', x: 0, y: 0 }),
      );
    });

    it('removes it with the right button and keeps the browser menu closed over it', () => {
      pointer('pointerdown', 132, 92, 2);
      pointer('pointerup', 132, 92, 2);
      expect(onPointEdit).toHaveBeenCalledExactlyOnceWith({ action: 'delete', objId: 7, index: 0 });
      const overPoint = new MouseEvent('contextmenu', {
        clientX: 132,
        clientY: 92,
        cancelable: true,
      });
      host.dispatchEvent(overPoint);
      expect(overPoint.defaultPrevented).toBe(true);
      const elsewhere = new MouseEvent('contextmenu', {
        clientX: 400,
        clientY: 300,
        cancelable: true,
      });
      host.dispatchEvent(elsewhere);
      expect(elsewhere.defaultPrevented).toBe(false);
    });

    it('flips it between positive and negative with a shift-click', () => {
      pointer('pointerdown', 132, 92, 0, true);
      pointer('pointerup', 132, 92, 0, true);
      expect(onPointEdit).toHaveBeenCalledExactlyOnceWith({ action: 'flip', objId: 7, index: 0 });
    });

    it('does not add a second point on top of it, and still adds one elsewhere', () => {
      pointer('pointerdown', 134, 93);
      pointer('pointerup', 134, 93);
      expect(onPoint).not.toHaveBeenCalled();
      expect(onPointEdit).not.toHaveBeenCalled();
      pointer('pointerdown', 400, 300);
      pointer('pointerup', 400, 300);
      expect(onPoint).toHaveBeenCalledTimes(1);
    });

    it('leaves points alone while panning and when only masks are shown', () => {
      viewport.panMode.set(true);
      pointer('pointerdown', 132, 92);
      pointer('pointermove', 172, 102);
      pointer('pointerup', 172, 102);
      viewport.panMode.set(false);
      viewport.fit();
      viewport.toggleViewMode('masks');
      pointer('pointerdown', 132, 92, 2);
      pointer('pointerup', 132, 92, 2);
      expect(onPointEdit).not.toHaveBeenCalled();
    });

    it('shows a grab cursor over a point', () => {
      pointer('pointermove', 132, 92);
      expect(host.style.cursor).toBe('grab');
      pointer('pointermove', 400, 300);
      expect(host.style.cursor).toBe('');
    });
  });

  it('draws a tracked point only from the frame it was placed on', () => {
    const track = {
      point_id: 'p0',
      obj_id: 1,
      source_frame_idx: 2,
      source_x: 10,
      source_y: 10,
      tracks: [
        [10, 10],
        [10, 10],
        [10, 10],
        [12, 11],
      ],
      visibility: [true, true, true, true],
    };
    const trackedPointCircles = (frame: number) => {
      viewport.render(image, masks, [], [track], [], 'point', frame);
      return stage.find<Konva.Circle>('Circle').length;
    };

    expect(trackedPointCircles(1)).toBe(0);
    expect(trackedPointCircles(2)).toBe(1);
    expect(trackedPointCircles(3)).toBe(1);
  });

  it('uses separate Konva images and vector markers in source-image coordinates', () => {
    expect(stage.find('Image')).toHaveLength(2);
    const circle = stage.findOne<Konva.Circle>('Circle')!;
    expect(circle.position()).toEqual({ x: 100, y: 50 });
    expect(circle.fill()).toBe('#00ff00');
    expect(viewport.clientToImage(132, 92)).toEqual({ x: 100, y: 50 });
    expect(viewport.clientToImage(20, 30)).toBeNull();
  });

  it('hides the video image for masks + points, and the points too for masks only', () => {
    const frame = stage.find<Konva.Image>('Image').find((node) => node.name() !== 'mask')!;
    const mask = stage.findOne<Konva.Image>('.mask')!;
    const point = stage.findOne<Konva.Circle>('Circle')!;
    const shown = () => [frame, mask, point].map((node) => node.isVisible());

    expect(viewport.viewMode()).toBe('full');
    expect(shown()).toEqual([true, true, true]);

    viewport.toggleViewMode('masks-points');
    expect(viewport.viewMode()).toBe('masks-points');
    expect(shown()).toEqual([false, true, true]);

    viewport.toggleViewMode('masks');
    expect(viewport.viewMode()).toBe('masks');
    expect(shown()).toEqual([false, true, false]);

    // Choosing the mode that is already shown goes back to the full view.
    viewport.toggleViewMode('masks');
    expect(viewport.viewMode()).toBe('full');
    expect(shown()).toEqual([true, true, true]);
  });

  it('keeps the chosen view mode when frames change and when the canvas is attached again', () => {
    viewport.toggleViewMode('masks-points');
    render();
    const frame = () => stage.find<Konva.Image>('Image').find((node) => node.name() !== 'mask')!;
    expect(frame().isVisible()).toBe(false);
    expect(stage.findOne<Konva.Circle>('Circle')!.isVisible()).toBe(true);

    viewport.attach(host, onPoint);
    stage = Konva.stages[Konva.stages.length - 1];
    render();
    expect(viewport.viewMode()).toBe('masks-points');
    expect(frame().isVisible()).toBe(false);
    expect(stage.findOne<Konva.Circle>('Circle')!.isVisible()).toBe(true);
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
