import { setAcceptedPoints } from '../../testing/workflow-test-helpers';
import { describe, beforeEach, expect, it, vi } from 'vitest';
import { createVideoMaskerFixture } from '../../testing/component-test-helpers';

describe('ObjectSidebarComponent', () => {
  let component: Awaited<ReturnType<typeof createVideoMaskerFixture>>['component'];
  let fixture: Awaited<ReturnType<typeof createVideoMaskerFixture>>['fixture'];
  let backendMock: Awaited<ReturnType<typeof createVideoMaskerFixture>>['backendMock'];

  beforeEach(async () => {
    ({ component, fixture, backendMock } = await createVideoMaskerFixture());
  });

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
    expect(backendMock.removeObject).toHaveBeenCalledWith(2, {
      session_id: 'session-1',
      expected_revision: 1,
    });
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
    expect(component.store.objects()[0].label).toBe('Car');
    expect(input.value).toBe('Car');
    input.value = '   ';
    input.dispatchEvent(new Event('change'));
    expect(component.store.objects()[0].label).toBe('Car');
    expect(input.value).toBe('Car');
  });

  it('navigates to a child frame and sends the correct index when removing its point', () => {
    component.store.numFrames.set(20);
    setAcceptedPoints(
      component.store,
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
    fixture.nativeElement.querySelector('app-object-sidebar .point-list-toggle').click();
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

  it('collapses and opens an object point list, whether or not the object is selected', () => {
    component.store.annotations.set({
      version: 1,
      revision: 0,
      next_object_id: 3,
      objects: {
        '1': { id: 1, label: 'Car', color: '#ff0000' },
        '2': { id: 2, label: 'Person', color: '#00ff00' },
      },
      prompts: [],
    });
    setAcceptedPoints(
      component.store,
      new Map([
        [
          0,
          new Map([
            [1, [{ x: 1, y: 2, label: 1 }]],
            [2, [{ x: 3, y: 4, label: 1 }]],
          ]),
        ],
      ]),
    );
    component.store.selectedObjectId.set(1);
    fixture.detectChanges();
    const element: HTMLElement = fixture.nativeElement;
    const toggles = () =>
      element.querySelectorAll<HTMLButtonElement>('app-object-sidebar .point-list-toggle');
    const lists = () => element.querySelectorAll('app-object-sidebar .child-point-list').length;

    // Every list starts collapsed, selected or not.
    expect(lists()).toBe(0);
    expect(toggles()[0].getAttribute('aria-expanded')).toBe('false');
    expect(toggles()[1].getAttribute('aria-expanded')).toBe('false');

    toggles()[1].click();
    fixture.detectChanges();
    expect(lists()).toBe(1);
    expect(toggles()[1].getAttribute('aria-expanded')).toBe('true');
    expect(component.store.selectedObjectId()).toBe(1);

    toggles()[1].click();
    fixture.detectChanges();
    expect(lists()).toBe(0);
  });
});
