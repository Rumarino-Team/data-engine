import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { VideoMaskerComponent } from '../video-masker.component';
import { ObjectOperationsService } from '../services/object-operations.service';
import { createBackendMocks, mockProviders, initializeEditorStore } from './workflow-test-helpers';

/** Creates the page only for tests that exercise view bindings and UI handlers. */
export async function createVideoMaskerFixture() {
  const mocks = createBackendMocks();
  await TestBed.configureTestingModule({
    imports: [VideoMaskerComponent],
    providers: mockProviders(mocks),
  }).compileComponents();
  const fixture = TestBed.createComponent(VideoMaskerComponent);
  const component = fixture.componentInstance;
  vi.spyOn(component.viewport, 'attach').mockImplementation(() => {});
  initializeEditorStore(component.store);
  const objectOperations = fixture.debugElement.injector.get(ObjectOperationsService);
  return { ...mocks, component, fixture, objectOperations };
}
