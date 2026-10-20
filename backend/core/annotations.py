"""Accepted annotation inputs, independent of predictor lifetime and UI caches."""

from typing import Literal
from secrets import randbelow

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from schemas.video import VideoObject, VideoPrompt, MaskOverride


class SessionAnnotations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    revision: int = Field(default=0, ge=0)
    next_object_id: int = Field(default=1, ge=1)
    objects: dict[int, VideoObject] = Field(default_factory=dict)
    # At most one merged prompt per frame/object; size tracks annotated pairs, not clicks.
    prompts: list[VideoPrompt] = Field(default_factory=list)

    mask_overrides: list[MaskOverride] = Field(default_factory=list)
    input_revisions: dict[int, int] = Field(default_factory=dict)

    # Keys touched since the last delta; a superset is harmless because deltas carry current values.
    _changed: set[tuple[str, object]] = PrivateAttr(default_factory=set)

    @model_validator(mode="after")
    def validate_references(self):
        if any(key != obj.id for key, obj in self.objects.items()):
            raise ValueError("Object keys must match object IDs.")
        if any(prompt.obj_id not in self.objects for prompt in self.prompts):
            raise ValueError("Prompts must reference an existing object.")
        if any(mask.obj_id not in self.objects for mask in self.mask_overrides):
            raise ValueError("Masks must reference an existing object.")
        prompt_keys = [(prompt.frame_idx, prompt.obj_id) for prompt in self.prompts]
        if len(set(prompt_keys)) != len(prompt_keys):
            raise ValueError("Each frame/object accepts one prompt.")
        mask_keys = {(mask.frame_idx, mask.obj_id) for mask in self.mask_overrides}
        if len(mask_keys) != len(self.mask_overrides) or mask_keys.intersection(prompt_keys):
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
            self._changed.add(("object", accepted.id))

    def set_prompt(self, prompt: VideoPrompt, clear_old_points: bool = True) -> None:
        if prompt.obj_id not in self.objects:
            raise ValueError("Prompt object does not exist.")
        key = (prompt.frame_idx, prompt.obj_id)
        existing = next((event for event in self.prompts if (event.frame_idx, event.obj_id) == key), None)
        if existing is not None and not clear_old_points:
            # Appending clicks merges into the accepted entry; replaying it is equivalent for SAM2.
            prompt = VideoPrompt(frame_idx=prompt.frame_idx, obj_id=prompt.obj_id,
                                 points=[*existing.points, *prompt.points],
                                 labels=[*existing.labels, *prompt.labels],
                                 box=prompt.box if prompt.box is not None else existing.box)
        self.prompts = [event for event in self.prompts if (event.frame_idx, event.obj_id) != key]
        self._drop_mask(key)
        self.prompts.append(prompt.model_copy(deep=True))
        self._changed.add(("prompt", key))
        self._bump_inputs({prompt.obj_id})

    def clear_prompts(self, frame_idx: int | None = None, obj_id: int | None = None) -> None:
        def matches(event) -> bool:
            return ((frame_idx is None or event.frame_idx == frame_idx)
                    and (obj_id is None or event.obj_id == obj_id))

        # Single pass: membership tests against `remaining` were quadratic in prompt count.
        remaining, remaining_masks, affected = [], [], set()
        for event in self.prompts:
            if matches(event):
                affected.add(event.obj_id)
                self._changed.add(("prompt", (event.frame_idx, event.obj_id)))
            else:
                remaining.append(event)
        for mask in self.mask_overrides:
            if matches(mask):
                affected.add(mask.obj_id)
                self._changed.add(("mask", (mask.frame_idx, mask.obj_id)))
            else:
                remaining_masks.append(mask)
        if affected:
            self.prompts = remaining
            self.mask_overrides = remaining_masks
            self._bump_inputs(affected)

    def invalidate_inputs(self) -> None:
        """Expire every generated result without changing accepted inputs (e.g. reset)."""
        self._bump_inputs(set(self.objects))

    def set_mask(self, mask: MaskOverride) -> None:
        if mask.obj_id not in self.objects:
            raise ValueError("Mask object does not exist.")
        key = (mask.frame_idx, mask.obj_id)
        if any((event.frame_idx, event.obj_id) == key for event in self.prompts):
            self.prompts = [event for event in self.prompts if (event.frame_idx, event.obj_id) != key]
            self._changed.add(("prompt", key))
        self._drop_mask(key)
        self.mask_overrides.append(mask.model_copy(deep=True))
        self._changed.add(("mask", key))
        self._bump_inputs({mask.obj_id})

    def remove_object(self, obj_id: int) -> None:
        if obj_id in self.objects:
            del self.objects[obj_id]
            self._changed.update(("prompt", (event.frame_idx, obj_id)) for event in self.prompts if event.obj_id == obj_id)
            self._changed.update(("mask", (mask.frame_idx, obj_id)) for mask in self.mask_overrides if mask.obj_id == obj_id)
            self._changed.update({("object", obj_id), ("input", obj_id)})
            self.mask_overrides = [mask for mask in self.mask_overrides if mask.obj_id != obj_id]
            self.input_revisions.pop(obj_id, None)
            self.prompts = [event for event in self.prompts if event.obj_id != obj_id]
            self.revision += 1

    def _drop_mask(self, key: tuple[int, int]) -> None:
        if any((mask.frame_idx, mask.obj_id) == key for mask in self.mask_overrides):
            self.mask_overrides = [mask for mask in self.mask_overrides if (mask.frame_idx, mask.obj_id) != key]
            self._changed.add(("mask", key))

    def _bump_inputs(self, obj_ids: set[int]) -> None:
        self.revision += 1
        for object_id in obj_ids:
            self.input_revisions[object_id] = self.revision
            self._changed.add(("input", object_id))

    def prompt_events(self) -> list[dict]:
        """Detached predictor input; runtime consumers cannot mutate the annotations."""
        return [event.model_dump() for event in self.prompts]

    def snapshot(self) -> dict:
        return self.model_dump(mode="json")

    def take_delta(self, base_revision: int) -> dict:
        """Current values for keys changed since the previous delta, applied to `base_revision`."""
        changed, self._changed = self._changed, set()
        prompts = {(event.frame_idx, event.obj_id): event for event in self.prompts}
        masks = {(mask.frame_idx, mask.obj_id): mask for mask in self.mask_overrides}
        delta = {"base_revision": base_revision, "revision": self.revision, "next_object_id": self.next_object_id,
                 "objects": {}, "removed_objects": [], "prompts": [], "removed_prompts": [],
                 "mask_overrides": [], "removed_mask_overrides": [],
                 "input_revisions": {}, "removed_input_revisions": []}
        for kind, key in sorted(changed, key=repr):
            if kind == "object":
                if key in self.objects:
                    delta["objects"][str(key)] = self.objects[key].model_dump(mode="json")
                else:
                    delta["removed_objects"].append(key)
            elif kind == "prompt":
                if key in prompts:
                    delta["prompts"].append(prompts[key].model_dump(mode="json"))
                else:
                    delta["removed_prompts"].append(list(key))
            elif kind == "mask":
                if key in masks:
                    delta["mask_overrides"].append(masks[key].model_dump(mode="json"))
                else:
                    delta["removed_mask_overrides"].append(list(key))
            elif key in self.input_revisions:
                delta["input_revisions"][str(key)] = self.input_revisions[key]
            else:
                delta["removed_input_revisions"].append(key)
        return delta
