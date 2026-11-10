import { CommonModule } from '@angular/common';
import {
  Component,
  ElementRef,
  EventEmitter,
  HostListener,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
  inject,
} from '@angular/core';
import { ConfirmRequest } from '../../services/confirm.service';

@Component({
  selector: 'app-confirm-dialog',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './confirm-dialog.component.html',
  styleUrl: './confirm-dialog.component.css',
})
export class ConfirmDialogComponent implements OnChanges {
  @Input() request: ConfirmRequest | null = null;
  /** Emits the chosen action's value, or null when dismissed. */
  @Output() answer = new EventEmitter<string | null>();

  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['request'] && this.request) {
      // Start on the first (safest) action so Enter never discards work by accident.
      queueMicrotask(() => this.host.nativeElement.querySelector<HTMLElement>('button')?.focus());
    }
  }

  @HostListener('document:keydown.escape')
  dismiss(): void {
    if (this.request) this.answer.emit(null);
  }
}
