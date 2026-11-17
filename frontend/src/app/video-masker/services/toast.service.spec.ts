import { describe, expect, it } from 'vitest';
import { ToastService } from './toast.service';

describe('ToastService', () => {
  it('adds toast to the stack', () => {
    const service = new ToastService();
    service.show('error', 'Oops', 'Failed');

    expect(service.toasts().length).toBe(1);
    expect(service.toasts()[0].title).toBe('Oops');
  });
});
