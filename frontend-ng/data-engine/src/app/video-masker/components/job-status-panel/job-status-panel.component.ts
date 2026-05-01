import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { BackendJob, JobStageHistoryEntry } from '../../../services/backend.service';

@Component({
  selector: 'app-job-status-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './job-status-panel.component.html',
  styleUrl: './job-status-panel.component.css',
})
export class JobStatusPanelComponent {
  @Input({ required: true }) isLoading = false;
  @Input() activeJob: BackendJob | null = null;
  @Input() activeJobTitle = '';
  @Input() recentHistory: JobStageHistoryEntry[] = [];

  isPropagationJob(): boolean {
    return this.activeJob?.operation === 'mask_propagation';
  }

  hasBatchProgress(): boolean {
    return (
      this.activeJob?.batch_current !== null &&
      this.activeJob?.batch_current !== undefined &&
      this.activeJob?.batch_total !== null &&
      this.activeJob?.batch_total !== undefined &&
      this.activeJob.batch_total > 0
    );
  }

  batchProgressPercent(): number {
    if (!this.hasBatchProgress()) {
      return 35;
    }
    return Math.min(
      100,
      Math.max(0, ((this.activeJob?.batch_current || 0) / (this.activeJob?.batch_total || 1)) * 100),
    );
  }

  overallProgressPercent(): number {
    if (this.activeJob?.progress === null || this.activeJob?.progress === undefined) {
      return 35;
    }
    return Math.min(100, Math.max(0, (this.activeJob.progress || 0) * 100));
  }

  hasOverallCount(): boolean {
    return (
      this.activeJob?.current !== null &&
      this.activeJob?.current !== undefined &&
      this.activeJob?.total !== null &&
      this.activeJob?.total !== undefined &&
      this.activeJob.total > 0
    );
  }

  windowLabel(): string | null {
    if (!this.activeJob?.window_count) {
      return null;
    }
    return `Window ${this.activeJob.window_index || 1} of ${this.activeJob.window_count}`;
  }
}
