import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { describe, expect, it, vi } from 'vitest';
import { BackendService } from '../services/backend.service';
import { DesktopBridgeService } from '../services/desktop-bridge.service';
import { FrameRendererService } from './services/frame-renderer.service';
import { MaskStateService } from './services/mask-state.service';
import { ToastService } from './services/toast.service';
import { VideoJobsService } from './services/video-jobs.service';
import { VideoSessionService } from './services/video-session.service';
import { VideoMaskerFacade } from './video-masker.facade';

describe('VideoMaskerFacade', () => {
  it('delegates source labels/hints to session service', () => {
    const sessionMock = {
      getLoadPathPlaceholder: vi.fn(() => 'placeholder'),
      getBrowseLabel: vi.fn(() => 'browse'),
      getLoadModeHint: vi.fn(() => 'hint'),
      browse: vi.fn(async () => '/tmp/frames'),
    } as unknown as VideoSessionService;

    TestBed.configureTestingModule({
      providers: [
        {
          provide: BackendService,
          useValue: {
            getApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
            health: vi.fn(() => of({ status: 'ok' })),
          },
        },
        { provide: DesktopBridgeService, useValue: {} },
        { provide: VideoJobsService, useValue: { run: vi.fn() } },
        { provide: VideoSessionService, useValue: sessionMock },
        FrameRendererService,
        MaskStateService,
        ToastService,
        VideoMaskerFacade,
      ],
    });

    const facade = TestBed.inject(VideoMaskerFacade);

    expect(facade.getLoadPathPlaceholder()).toBe('placeholder');
    expect(facade.getBrowseLabel()).toBe('browse');
    expect(facade.getLoadModeHint()).toBe('hint');
  });

  it('sets videoDir from browseSelectedSource', async () => {
    const sessionMock = {
      getLoadPathPlaceholder: vi.fn(() => 'placeholder'),
      getBrowseLabel: vi.fn(() => 'browse'),
      getLoadModeHint: vi.fn(() => 'hint'),
      browse: vi.fn(async () => '/tmp/frames'),
    } as unknown as VideoSessionService;

    TestBed.configureTestingModule({
      providers: [
        {
          provide: BackendService,
          useValue: {
            getApiUrl: vi.fn(() => 'http://127.0.0.1:8000'),
            health: vi.fn(() => of({ status: 'ok' })),
          },
        },
        { provide: DesktopBridgeService, useValue: {} },
        { provide: VideoJobsService, useValue: { run: vi.fn() } },
        { provide: VideoSessionService, useValue: sessionMock },
        FrameRendererService,
        MaskStateService,
        ToastService,
        VideoMaskerFacade,
      ],
    });

    const facade = TestBed.inject(VideoMaskerFacade);
    await facade.browseSelectedSource();

    expect(facade.videoDir()).toBe('/tmp/frames');
  });
});
