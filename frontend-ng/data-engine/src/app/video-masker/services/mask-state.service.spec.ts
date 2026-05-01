import { describe, expect, it } from 'vitest';
import { MaskStateService } from './mask-state.service';

describe('MaskStateService', () => {
  const service = new MaskStateService();

  it('flags epoch mismatch for live state clear', () => {
    const res = service.updateStateEpoch(3, 4);
    expect(res).toEqual({ normalizedEpoch: 4, shouldClearLiveState: true });
  });

  it('does not clear when epoch is unchanged', () => {
    const res = service.updateStateEpoch(3, 3);
    expect(res).toEqual({ normalizedEpoch: 3, shouldClearLiveState: false });
  });

  it('tracks mark/unmark correctly', () => {
    const marked = service.markObjectAsLiveEdited(new Map(), 2, 7);
    expect(marked.get(2)?.has(7)).toBe(true);

    const unmarked = service.unmarkObjectAsLiveEdited(marked, 2, 7);
    expect(unmarked.get(2)).toBeUndefined();
  });
});
