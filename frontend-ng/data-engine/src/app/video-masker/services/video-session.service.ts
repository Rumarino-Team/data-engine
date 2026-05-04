import { Injectable } from '@angular/core';
import { DesktopBridgeService } from '../../services/desktop-bridge.service';
import { LoadSourceMode } from '../state/video-masker-ui.types';

@Injectable({ providedIn: 'root' })
export class VideoSessionService {
  constructor(private desktopBridge: DesktopBridgeService) {}

  getLoadPathPlaceholder(mode: LoadSourceMode): string {
    switch (mode) {
      case 'video_file':
        return 'Enter video file path (.mp4, .mov, .avi, .mkv, .webm, .m4v)';
      case 'saved_session_dir':
        return 'Enter saved session directory path (contains session.json, frames/, and masks/)';
      default:
        return 'Enter frames directory path';
    }
  }

  getBrowseLabel(mode: LoadSourceMode): string {
    switch (mode) {
      case 'video_file':
        return 'Browse Video';
      case 'saved_session_dir':
        return 'Browse Saved Session';
      default:
        return 'Browse Frames';
    }
  }

  getLoadModeHint(mode: LoadSourceMode): string {
    if (mode === 'saved_session_dir') {
      return 'Choose a saved session directory containing session.json, frames/, and masks/.';
    }
    if (mode === 'video_file') {
      return 'Choose a video file; backend will extract frames and create a new session.';
    }
    return 'Choose a directory that already contains extracted video frames.';
  }

  async browse(mode: LoadSourceMode): Promise<string | null> {
    if (!this.desktopBridge.isTauri()) {
      return null;
    }
    if (mode === 'video_file') {
      return this.desktopBridge.pickVideoFile();
    }
    return this.desktopBridge.pickFramesDirectory();
  }
}
