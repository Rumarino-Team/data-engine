"""Accepted annotation inputs, independent of predictor lifetime and UI caches."""

from typing import Literal
from secrets import randbelow

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.video import VideoObject, VideoPrompt, InteractiveMaskRLE


class SessionAnnotations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    revision: int = Field(default=0, ge=0)
    next_object_id: int = Field(default=1, ge=1)
    objects: dict[int, VideoObject] = Field(default_factory=dict)
    prompts: list[VideoPrompt] = Field(default_factory=list)

    mask_overrides: list[InteractiveMaskRLE] = Field(default_factory=list)
    input_revisions: dict[int, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self):
        if any(key != obj.id for key, obj in self.objects.items()):
            raise ValueError("Object keys must match object IDs.")
        if any(prompt.obj_id not in self.objects for prompt in self.prompts):
            raise ValueError("Prompts must reference an existing object.")
        if any(mask.obj_id not in self.objects for mask in self.mask_overrides):
            raise ValueError("Masks must reference an existing object.")
        mask_keys = [(mask.frame_idx, mask.obj_id) for mask in self.mask_overrides]
        if len(set(mask_keys)) != len(mask_keys) or any(
                (prompt.frame_idx, prompt.obj_id) in mask_keys for prompt in self.prompts):
            raise ValueError("Each frame/object accepts prompts or one explicit mask.")
        if any(obj_id not in self.objects or not 0 <= revision <= self.revision
               for obj_id, revision in self.input_revisions.items()):
            raise ValueError("Invalid mask input revisions.")
        self.next_object_id = max(self.next_object_id, max(self.objects, default=0) + 1)
        return self

    def create_object(self) -> VideoObject:
        obj = VideoObject(id=self.next_object_id, label=f"Object {self.next_object_id}",
                          color=f"#{randbelow(0x1000000):06x}")
        self.upsert_object(obj.model_dump())
        return obj

    def upsert_object(self, obj: dict) -> None:
        accepted = VideoObject.model_validate(obj)
        if self.objects.get(accepted.id) != accepted:
            self.objects[accepted.id] = accepted
            self.next_object_id = max(self.next_object_id, accepted.id + 1)
            self.revision += 1

    def set_prompt(self, prompt: VideoPrompt) -> None:
        if prompt.obj_id not in self.objects:
            raise ValueError("Prompt object does not exist.")
        if prompt.clear_old_points:
            self.prompts = [event for event in self.prompts
                            if (event.frame_idx, event.obj_id) != (prompt.frame_idx, prompt.obj_id)]
        self.mask_overrides = [mask for mask in self.mask_overrides
                               if (mask.frame_idx, mask.obj_id) != (prompt.frame_idx, prompt.obj_id)]
        self.prompts.append(prompt.model_copy(deep=True))
        self.revision += 1
        self.input_revisions[prompt.obj_id] = self.revision

    def clear_prompts(self, frame_idx: int | None = None, obj_id: int | None = None) -> None:
        remaining = [event for event in self.prompts
                     if not ((frame_idx is None or event.frame_idx == frame_idx)
                             and (obj_id is None or event.obj_id == obj_id))]
        remaining_masks = [mask for mask in self.mask_overrides
                           if not ((frame_idx is None or mask.frame_idx == frame_idx)
                                   and (obj_id is None or mask.obj_id == obj_id))]
        affected = {event.obj_id for event in self.prompts if event not in remaining}
        affected.update(mask.obj_id for mask in self.mask_overrides if mask not in remaining_masks)
        if affected:
            self.prompts = remaining
            self.mask_overrides = remaining_masks
            self.revision += 1
            for object_id in affected:
                self.input_revisions[object_id] = self.revision

    def set_mask(self, mask: InteractiveMaskRLE) -> None:
        if mask.obj_id not in self.objects:
            raise ValueError("Mask object does not exist.")
        key = (mask.frame_idx, mask.obj_id)
        self.prompts = [event for event in self.prompts if (event.frame_idx, event.obj_id) != key]
        self.mask_overrides = [event for event in self.mask_overrides if (event.frame_idx, event.obj_id) != key]
        self.mask_overrides.append(mask.model_copy(deep=True))
        self.revision += 1
        self.input_revisions[mask.obj_id] = self.revision

    def remove_object(self, obj_id: int) -> None:
        if obj_id in self.objects:
            del self.objects[obj_id]
            self.mask_overrides = [mask for mask in self.mask_overrides if mask.obj_id != obj_id]
            self.input_revisions.pop(obj_id, None)
            self.prompts = [event for event in self.prompts if event.obj_id != obj_id]
            self.revision += 1

    def prompt_events(self) -> list[dict]:
        """Detached predictor input; runtime consumers cannot mutate the annotations."""
        return [event.model_dump() for event in self.prompts]

    def snapshot(self) -> dict:
        return self.model_dump(mode="json")

    def interactive_points(self) -> list[dict]:
        """Editor projection of accepted prompts, never an input source."""
        grouped: dict[tuple[int, int], list[dict]] = {}
        for event in self.prompts:
            key = (event.frame_idx, event.obj_id)
            if event.clear_old_points:
                grouped[key] = []
            grouped.setdefault(key, []).extend(
                {"frame_idx": event.frame_idx, "obj_id": event.obj_id,
                 "x": point[0], "y": point[1], "label": label}
                for point, label in zip(event.points, event.labels)
            )
        return [point for points in grouped.values() for point in points]
