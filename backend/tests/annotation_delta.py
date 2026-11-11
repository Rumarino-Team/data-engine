"""Reference client for mutation deltas; mirrors the frontend store's patch application."""
import copy


def apply_delta(snapshot: dict, delta: dict) -> dict:
    assert snapshot["revision"] == delta["base_revision"], "delta does not apply to this snapshot"
    result = copy.deepcopy(snapshot)
    result["revision"] = delta["revision"]
    result["next_object_id"] = delta["next_object_id"]
    for obj_id in delta["removed_objects"]:
        result["objects"].pop(str(obj_id), None)
    result["objects"].update(copy.deepcopy(delta["objects"]))
    for field, removed in (("prompts", "removed_prompts"), ("mask_overrides", "removed_mask_overrides")):
        replaced = {tuple(key) for key in delta[removed]}
        replaced.update((entry["frame_idx"], entry["obj_id"]) for entry in delta[field])
        result[field] = [entry for entry in result[field] if (entry["frame_idx"], entry["obj_id"]) not in replaced]
        result[field].extend(copy.deepcopy(delta[field]))
    for obj_id in delta["removed_input_revisions"]:
        result["input_revisions"].pop(str(obj_id), None)
    result["input_revisions"].update(delta["input_revisions"])
    return result


def same_annotations(left: dict, right: dict) -> bool:
    """Equality ignoring list order, which carries no meaning for keyed entries."""
    def keyed(snapshot):
        return {**snapshot,
                "prompts": sorted(snapshot["prompts"], key=lambda entry: (entry["frame_idx"], entry["obj_id"])),
                "mask_overrides": sorted(snapshot["mask_overrides"], key=lambda entry: (entry["frame_idx"], entry["obj_id"]))}
    return keyed(left) == keyed(right)
