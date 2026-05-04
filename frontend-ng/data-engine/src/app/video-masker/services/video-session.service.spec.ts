import { describe, expect, it, vi } from 'vitest';
import { DesktopBridgeService } from '../../services/desktop-bridge.service';
import { VideoSessionService } from './video-session.service';

describe('VideoSessionService', () => {
  it('returns mode-specific labels and placeholders', () => {
    const desktop = { isTauri: vi.fn(() => false) } as unknown as DesktopBridgeService;
    const service = new VideoSessionService(desktop);

    expect(service.getBrowseLabel('video_file')).toBe('Browse Video');
    expect(service.getLoadPathPlaceholder('saved_session_dir')).toContain('session.json');
  });

  it('uses native video picker in tauri video mode', async () => {
    const desktop = {
      isTauri: vi.fn(() => true),
      pickVideoFile: vi.fn(() => Promise.resolve('/tmp/a.mp4')),
      pickFramesDirectory: vi.fn(() => Promise.resolve('/tmp/frames')),
    } as unknown as DesktopBridgeService;
    const service = new VideoSessionService(desktop);

    const out = await service.browse('video_file');

    expect(out).toBe('/tmp/a.mp4');
    expect((desktop as any).pickVideoFile).toHaveBeenCalled();
  });
});
