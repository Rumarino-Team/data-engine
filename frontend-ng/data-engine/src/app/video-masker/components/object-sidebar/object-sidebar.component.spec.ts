import { describe, beforeEach, expect, it, vi } from 'vitest';
import { createVideoMaskerFixture } from '../../testing/component-test-helpers';
import { By } from '@angular/platform-browser';
import { ObjectSidebarComponent } from './object-sidebar.component';

describe('ObjectSidebarComponent', () => {
  let component: Awaited<ReturnType<typeof createVideoMaskerFixture>>['component'];
  let fixture: Awaited<ReturnType<typeof createVideoMaskerFixture>>['fixture'];
  let backendMock: Awaited<ReturnType<typeof createVideoMaskerFixture>>['backendMock'];

  beforeEach(async () => {
    ({ component, fixture, backendMock } = await createVideoMaskerFixture());
  });

  const sidebar = (): ObjectSidebarComponent => {
    fixture.detectChanges();
    return fixture.debugElement.query(By.directive(ObjectSidebarComponent)).componentInstance;
  };

  it('creates and removes objects through the sidebar controls', () => {
    vi.spyOn(component.framePipeline, 'scheduleFrameLoad').mockImplementation(() => {});
    fixture.detectChanges();
    const buttons = fixture.nativeElement.querySelectorAll(
      'app-object-sidebar .button-group button',
    );
    buttons[0].click();
    expect(component.store.objects().map((object) => object.id)).toEqual([1, 2]);
    expect(component.store.selectedObjectId()).toBe(2);
    buttons[1].click();
    expect(backendMock.removeObject).toHaveBeenCalledWith(2);
    expect(component.store.objects().map((object) => object.id)).toEqual([1]);
    expect(component.store.selectedObjectId()).toBe(1);
  });

  it('edits a sidebar label and restores the accepted value after an empty edit', () => {
    fixture.detectChanges();
    const input = fixture.nativeElement.querySelector(
      'app-object-sidebar .object-name-input',
    ) as HTMLInputElement;
    input.value = '  Car  ';
    input.dispatchEvent(new Event('change'));
    expect(component.store.objects()[0].name).toBe('Car');
    expect(input.value).toBe('Car');
    input.value = '   ';
    input.dispatchEvent(new Event('change'));
    expect(component.store.objects()[0].name).toBe('Car');
    expect(input.value).toBe('Car');
  });

  it('navigates to a child frame and sends the correct index when removing its point', () => {
    component.store.numFrames.set(20);
    component.store.points.set(
      new Map([
        [
          2,
          new Map([
            [
              1,
              [
                { x: 1, y: 2, label: 1 },
                { x: 3, y: 4, label: 0 },
              ],
            ],
          ]),
        ],
      ]),
    );
    const removePoint = vi.spyOn(component.prompts, 'removePoint').mockResolvedValue();
    fixture.detectChanges();
    const buttons = fixture.nativeElement.querySelectorAll('app-object-sidebar .point-remove');
    buttons[1].click();
    expect(component.store.targetFrameIdx()).toBe(2);
    expect(removePoint).toHaveBeenCalledWith(1, 2, 1);
    component.store.isLoading.set(true);
    fixture.detectChanges();
    buttons[0].click();
    expect(removePoint).toHaveBeenCalledTimes(1);
  });

  it('shows separately placed points under their owning object and navigates to their frame', () => {
    component.store.objects.set([
      { id: 1, name: 'Car', color: '#ff0000' },
      { id: 2, name: 'Person', color: '#00ff00' },
    ]);
    component.store.numFrames.set(20);
    component.store.points.set(
      new Map([
        [10, new Map([[1, [{ x: 4, y: 5, label: 0 }]]])],
        [
          2,
          new Map([
            [2, [{ x: 8, y: 9, label: 1 }]],
            [1, [{ x: 1, y: 3, label: 1 }]],
          ]),
        ],
      ]),
    );
    const groups = component.store.objectPointGroups();
    expect(groups[0].children.map((point) => point.frameIdx)).toEqual([2, 10]);
    expect(groups[1].children).toHaveLength(1);
    sidebar().showObjectPoint(2, 2);
    expect(component.store.selectedObjectId()).toBe(2);
    expect(component.store.targetFrameIdx()).toBe(2);
    component.store.isPointRequestInFlight.set(true);
    sidebar().showObjectPoint(1, 10);
    expect(component.store.targetFrameIdx()).toBe(2);
    expect(component.store.selectedObjectId()).toBe(2);
  });

  it('edits object labels without changing point ownership and rejects blank labels', () => {
    const input = document.createElement('input');
    input.value = '  Left hand  ';
    sidebar().renameObject(1, input);
    expect(component.store.objects()[0]).toEqual({ id: 1, name: 'Left hand', color: '#ff0000' });
    input.value = '   ';
    sidebar().renameObject(1, input);
    expect(input.value).toBe('Left hand');
  });
});
