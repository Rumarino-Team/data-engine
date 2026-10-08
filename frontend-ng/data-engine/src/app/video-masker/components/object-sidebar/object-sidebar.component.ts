import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ObjectOperationsService } from '../../services/object-operations.service';
import { PointEditingService } from '../../services/point-editing.service';
import { VideoMaskerStateStore } from '../../services/video-masker-state.store';
import { clampFrameIndex } from '../../video-masker.util';

@Component({
  selector: 'app-object-sidebar',
  imports: [CommonModule],
  templateUrl: './object-sidebar.component.html',
  styleUrl: './object-sidebar.component.css',
})
export class ObjectSidebarComponent {
  readonly store = inject(VideoMaskerStateStore);
  readonly objects = inject(ObjectOperationsService);
  private readonly prompts = inject(PointEditingService);

  renameObject(id: number, input: HTMLInputElement): void {
    this.objects.renameObject(id, input.value);
    input.value = this.store.objects().find((object) => object.id === id)?.name ?? '';
  }

  showObjectPoint(objectId: number, frameIdx: number): void {
    if (this.store.isInteractionBusy()) return;
    this.store.selectedObjectId.set(objectId);
    this.store.targetFrameIdx.set(clampFrameIndex(frameIdx, this.store.numFrames() - 1));
  }

  removePoint(objectId: number, frameIdx: number, pointNumber: number): void {
    if (this.store.isInteractionBusy()) return;
    this.showObjectPoint(objectId, frameIdx);
    void this.prompts.removePoint(objectId, frameIdx, pointNumber - 1);
  }
}
