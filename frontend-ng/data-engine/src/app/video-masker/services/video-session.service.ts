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
