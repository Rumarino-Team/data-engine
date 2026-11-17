import { describe, expect, it } from 'vitest';
import { ToastService } from './toast.service';

describe('ToastService', () => {
  it('adds toast to the stack', () => {
    const service = new ToastService();
    service.show('error', 'Oops', 'Failed');

    expect(service.toasts().length).toBe(1);
    expect(service.toasts()[0].title).toBe('Oops');
  });

  it('dismisses toast by id', () => {
    const service = new ToastService();
    service.show('warning', 'Warn', 'Heads up');
    const id = service.toasts()[0].id;

    service.dismiss(id);

    expect(service.toasts()).toEqual([]);
  });
});
