import type { BackendJob } from '../../services/backend.service';

export type ToastSeverity = 'error' | 'warning' | 'info' | 'success';
export type LoadSourceMode = 'frames_dir' | 'video_file' | 'saved_session_dir';
export type TrackingOverlayStyle = 'point' | 'short' | 'full';
export type DebugMaskSource = 'live' | 'manifest' | 'none';

export interface AppToast {
  id: number;
  severity: ToastSeverity;
  title: string;
  message: string;
  createdAt: number;
}

export interface JobUiState {
  isLoading: boolean;
  activeJob: BackendJob | null;
  activeJobTitle: string;
}

export interface TrackingUiState {
  overlayStyle: TrackingOverlayStyle;
  useSupportGrid: boolean;
}

export interface MaskEditState {
  epoch: number;
  lastDiscardReason: string | null;
}

export interface VideoMaskerViewState {
  job: JobUiState;
  tracking: TrackingUiState;
  mask: MaskEditState;
  toasts: AppToast[];
}
