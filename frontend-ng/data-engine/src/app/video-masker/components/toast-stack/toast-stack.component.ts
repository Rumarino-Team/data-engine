import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { AppToast } from '../../state/video-masker-ui.types';

@Component({
  selector: 'app-toast-stack',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './toast-stack.component.html',
  styleUrls: ['./toast-stack.component.css'],
})
export class ToastStackComponent {
  @Input() toasts: AppToast[] = [];
  @Output() dismiss = new EventEmitter<number>();
}
