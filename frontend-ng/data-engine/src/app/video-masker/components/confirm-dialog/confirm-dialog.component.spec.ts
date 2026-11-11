import { ComponentFixture, TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it } from 'vitest';
import { ConfirmDialogComponent } from './confirm-dialog.component';

const request = {
  title: 'Unsaved changes',
  message: 'Save them first?',
  actions: [
    { value: 'save', label: 'Save', tone: 'primary' as const },
    { value: 'discard', label: 'Discard changes', tone: 'danger' as const },
  ],
};

describe('ConfirmDialogComponent', () => {
  let fixture: ComponentFixture<ConfirmDialogComponent>;
  let answers: (string | null)[];

  const buttons = () =>
    Array.from(fixture.nativeElement.querySelectorAll('button')) as HTMLButtonElement[];

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [ConfirmDialogComponent] }).compileComponents();
    fixture = TestBed.createComponent(ConfirmDialogComponent);
    answers = [];
    fixture.componentInstance.answer.subscribe((value) => answers.push(value));
  });

  it('renders nothing without a request', () => {
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[role="alertdialog"]')).toBeNull();
  });

  it('renders the request as an accessible dialog with its actions and Cancel', () => {
    fixture.componentRef.setInput('request', request);
    fixture.detectChanges();
    const dialog = fixture.nativeElement.querySelector('[role="alertdialog"]') as HTMLElement;
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(dialog.textContent).toContain('Unsaved changes');
    expect(dialog.textContent).toContain('Save them first?');
    expect(buttons().map((button) => button.textContent?.trim())).toEqual([
      'Save',
      'Discard changes',
      'Cancel',
    ]);
  });

  it('emits the chosen action value, and null for Cancel', () => {
    fixture.componentRef.setInput('request', request);
    fixture.detectChanges();
    buttons()[1].click();
    buttons()[2].click();
    expect(answers).toEqual(['discard', null]);
  });

  it('dismisses on Escape and on a backdrop click, but not on a dialog click', () => {
    fixture.componentRef.setInput('request', request);
    fixture.detectChanges();
    (fixture.nativeElement.querySelector('.dialog') as HTMLElement).click();
    expect(answers).toEqual([]);
    (fixture.nativeElement.querySelector('.backdrop') as HTMLElement).click();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(answers).toEqual([null, null]);
  });

  it('ignores Escape while no question is open', () => {
    fixture.detectChanges();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(answers).toEqual([]);
  });

  it('focuses the first action when a question opens', async () => {
    document.body.appendChild(fixture.nativeElement);
    fixture.componentRef.setInput('request', request);
    fixture.detectChanges();
    await Promise.resolve();
    expect(document.activeElement).toBe(buttons()[0]);
    fixture.nativeElement.remove();
  });
});
