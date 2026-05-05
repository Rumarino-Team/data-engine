declare module 'jsdom' {
  export class JSDOM {
    constructor(html?: string, options?: { url?: string });
    window: Window & typeof globalThis;
  }
}

interface ImportMeta {
  readonly dir: string;
}

declare const Bun: {
  file(path: string): {
    text(): Promise<string>;
  };
};

declare namespace NodeJS {
  interface ProcessEnv {
    [key: string]: string | undefined;
  }
}
