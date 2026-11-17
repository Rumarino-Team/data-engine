import { describe, expect, it } from 'vitest';
import { ConfirmService } from './confirm.service';

const question = {
  title: 'Unsaved changes',
  message: 'Save first?',
  actions: [{ value: 'save' as const, label: 'Save', tone: 'primary' as const }],
};

describe('ConfirmService', () => {
  it('shows one question and resolves with the chosen action', async () => {
    const service = new ConfirmService();
    const answer = service.ask(question);
    expect(service.request()).toEqual(question);
    service.answer('save');
    await expect(answer).resolves.toBe('save');
    expect(service.request()).toBeNull();
  });

  it('resolves null when dismissed', async () => {
    const service = new ConfirmService();
    const answer = service.ask(question);
    service.answer(null);
    await expect(answer).resolves.toBeNull();
  });

  it('dismisses an older question when a newer one is asked', async () => {
    const service = new ConfirmService();
    const first = service.ask(question);
    const second = service.ask({ ...question, title: 'Another' });
    await expect(first).resolves.toBeNull();
    expect(service.request()?.title).toBe('Another');
    service.answer('save');
    await expect(second).resolves.toBe('save');
  });

  it('ignores an answer when nothing is asked', () => {
    const service = new ConfirmService();
    expect(() => service.answer('save')).not.toThrow();
  });
});
