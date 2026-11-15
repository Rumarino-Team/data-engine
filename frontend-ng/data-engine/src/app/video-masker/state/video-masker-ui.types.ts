export type ToastSeverity = 'error' | 'warning' | 'info' | 'success';
export type LoadSourceMode = 'frames_dir' | 'video_file' | 'saved_session_dir';
export type TrackingOverlayStyle = 'point' | 'short' | 'full';
/** What the canvas shows: everything, masks and points without the video image, or only masks. */
export type CanvasViewMode = 'full' | 'masks-points' | 'masks';

export interface AppToast {
  id: number;
  severity: ToastSeverity;
  title: string;
  message: string;
  createdAt: number;
}
