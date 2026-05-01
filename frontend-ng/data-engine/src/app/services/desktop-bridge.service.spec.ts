import { describe, expect, it, vi } from 'vitest';
import { DesktopBridgeService } from './desktop-bridge.service';

vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: vi.fn(),
}));

describe('DesktopBridgeService', () => {
  it('returns false for browser environments without Tauri globals', () => {
    const service = new DesktopBridgeService();

    expect(service.isTauri()).toBe(false);
  });

  it('returns null without throwing when native pickers are unavailable', async () => {
    const service = new DesktopBridgeService();

    await expect(service.pickVideoFile()).resolves.toBeNull();
    await expect(service.pickFramesDirectory()).resolves.toBeNull();
  });
});
