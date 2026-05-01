import { ComponentFixture, TestBed } from '@angular/core/testing';
import { BackendJob } from '../../../services/backend.service';
import { JobStatusPanelComponent } from './job-status-panel.component';

describe('JobStatusPanelComponent', () => {
  let fixture: ComponentFixture<JobStatusPanelComponent>;
  let component: JobStatusPanelComponent;

  const makeJob = (overrides: Partial<BackendJob> = {}): BackendJob => ({
    job_id: 'job-1',
    operation: 'mask_propagation',
    status: 'running',
    stage: 'propagating_window',
    stage_label: 'Propagating masks',
    progress: 0.5,
    current: 64,
    total: 128,
    window_index: 1,
    window_count: 2,
    frame_idx: 64,
    batch_current: 32,
    batch_total: 32,
    batch_index: 2,
    batch_count: 4,
    stage_history: [],
    message: 'Processed 64 of 128 frames',
    result: null,
    error: null,
    started_at: 'now',
    updated_at: 'now',
    completed_at: null,
    ...overrides,
  });

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [JobStatusPanelComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(JobStatusPanelComponent);
    component = fixture.componentInstance;
    component.isLoading = true;
  });

  it('renders batch and overall progress for propagation jobs without frame label', () => {
    component.activeJobTitle = 'Propagating masks';
    component.activeJob = makeJob();

    fixture.detectChanges();

    const element: HTMLElement = fixture.nativeElement;
    expect(element.textContent).toContain('Batch');
    expect(element.textContent).toContain('32 / 32');
    expect(element.textContent).toContain('Overall');
    expect(element.textContent).toContain('64 / 128');
    expect(element.textContent).toContain('50%');
    expect(element.textContent).toContain('Window 1 of 2');
    expect(element.textContent).not.toContain('Frame 64');
    expect(element.querySelectorAll('.progress-track').length).toBe(2);
  });

  it('renders a single progress bar for non-propagation jobs', () => {
    component.activeJob = makeJob({
      operation: 'video_init',
      stage: 'indexing_frames',
      stage_label: 'Indexing frames',
      current: 8,
      total: 10,
      progress: 0.8,
      window_index: null,
      window_count: null,
      frame_idx: null,
      batch_current: null,
      batch_total: null,
      batch_index: null,
      batch_count: null,
    });

    fixture.detectChanges();

    const element: HTMLElement = fixture.nativeElement;
    expect(element.textContent).not.toContain('Batch');
    expect(element.textContent).toContain('80%');
    expect(element.textContent).toContain('8 / 10');
    expect(element.querySelectorAll('.progress-track').length).toBe(1);
  });
});
