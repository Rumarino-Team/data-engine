import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { BackendJob, JobStageHistoryEntry } from '../../../services/backend.service';

@Component({
  selector: 'app-job-status-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './job-status-panel.component.html',
})
export class JobStatusPanelComponent {
  @Input({ required: true }) isLoading = false;
  @Input() activeJob: BackendJob | null = null;
  @Input() activeJobTitle = '';
  @Input() recentHistory: JobStageHistoryEntry[] = [];
}
