export type ToastSeverity = 'error' | 'warning' | 'info' | 'success';
export type LoadSourceMode = 'frames_dir' | 'video_file' | 'saved_session_dir';
export type TrackingOverlayStyle = 'point' | 'short' | 'full';

export interface AppToast {
  id: number;
  severity: ToastSeverity;
  title: string;
  message: string;
  createdAt: number;
}
