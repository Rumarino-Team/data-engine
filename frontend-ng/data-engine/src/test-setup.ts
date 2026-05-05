import '@angular/compiler';
import * as AngularCore from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { BrowserTestingModule, platformBrowserTesting } from '@angular/platform-browser/testing';
import { JSDOM } from 'jsdom';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { afterEach } from 'vitest';

const dom = new JSDOM('<!doctype html><html><body></body></html>', {
  url: 'http://localhost/',
});

if (!globalThis.window || !globalThis.document) {
  Object.assign(globalThis, {
    window: dom.window,
    document: dom.window.document,
    Node: dom.window.Node,
    HTMLElement: dom.window.HTMLElement,
    HTMLCanvasElement: dom.window.HTMLCanvasElement,
    HTMLImageElement: dom.window.HTMLImageElement,
    MouseEvent: dom.window.MouseEvent,
  });
}

Object.assign(globalThis, {
  requestAnimationFrame:
    globalThis.requestAnimationFrame ??
    ((callback: FrameRequestCallback) => setTimeout(() => callback(Date.now()), 0)),
  cancelAnimationFrame: globalThis.cancelAnimationFrame ?? ((handle: number) => clearTimeout(handle)),
});

try {
  TestBed.initTestEnvironment(BrowserTestingModule, platformBrowserTesting());
} catch (error) {
  if (!String(error).includes('Cannot set base providers')) {
    throw error;
  }
}

const testRootUrl = pathToFileURL(`${process.cwd()}/src/`);
const resourceFiles: Record<string, URL> = {
  'app.html': new URL('app/app.html', testRootUrl),
  'app.css': new URL('app/app.css', testRootUrl),
  'video-masker.component.html': new URL('app/video-masker/video-masker.component.html', testRootUrl),
  'video-masker.component.css': new URL('app/video-masker/video-masker.component.css', testRootUrl),
  'job-status-panel.component.html': new URL(
    'app/video-masker/components/job-status-panel/job-status-panel.component.html',
    testRootUrl,
  ),
  'job-status-panel.component.css': new URL(
    'app/video-masker/components/job-status-panel/job-status-panel.component.css',
    testRootUrl,
  ),
  'toast-stack.component.html': new URL(
    'app/video-masker/components/toast-stack/toast-stack.component.html',
    testRootUrl,
  ),
  'toast-stack.component.css': new URL(
    'app/video-masker/components/toast-stack/toast-stack.component.css',
    testRootUrl,
  ),
};

const resolveFixtureUrl = (fixtureUrl: URL | string) => {
  const url = fixtureUrl.toString();
  const resourcePath = Object.entries(resourceFiles).find(([fileName]) =>
    url.includes(fileName),
  )?.[1];
  if (resourcePath) {
    return resourcePath;
  }
  if (typeof fixtureUrl === 'string' && /^[a-z]+:\/\//i.test(fixtureUrl)) {
    throw new Error(`Unknown text fixture URL: ${url}`);
  }
  if (
    typeof fixtureUrl !== 'string' &&
    'protocol' in fixtureUrl &&
    fixtureUrl.protocol !== 'file:'
  ) {
    throw new Error(`Unknown text fixture URL: ${url}`);
  }
  if (typeof fixtureUrl !== 'string' && !url.includes('file:')) {
    throw new Error(
      `Unknown text fixture URL: ${url}; keys=${Object.keys(fixtureUrl).join(',')}`,
    );
  }
  return fixtureUrl;
};

(globalThis as any).readTextFixture = async (fixtureUrl: URL | string) =>
  readFile(resolveFixtureUrl(fixtureUrl), 'utf-8');

(globalThis as any).resolveAngularTestResources = async () =>
  (AngularCore as any)['\u0275resolveComponentResources'](async (url: string) => {
    const resourcePath = Object.entries(resourceFiles).find(([fileName]) =>
      url.includes(fileName),
    )?.[1];
    if (!resourcePath) {
      throw new Error(`Unknown Angular test resource: ${url}`);
    }
    return await readFile(resourcePath, 'utf-8');
  });

afterEach(() => {
  TestBed.resetTestingModule();
});
