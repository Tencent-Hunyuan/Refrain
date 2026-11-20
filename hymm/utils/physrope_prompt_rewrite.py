import copy
import re
from collections import defaultdict


REFERENCE_VIDEO_SEGMENT_HINT = (
    "Use the provided reference video segment for this time range and preserve its "
    "subject identity, motion, camera movement, and temporal dynamics."
)
REFERENCE_IMAGE_ANCHOR_HINT_TEMPLATE = (
    "Use the provided reference image anchor at latent frame {frames} and preserve "
    "the subject identity and appearance around that moment."
)


def join_prompt_prefix_body(prefix, body):
    prefix = str(prefix or "").strip()
    body = str(body or "").strip()
    if not prefix:
        return body
    if not body:
        return prefix
    return f"{prefix} {body}"


def append_prompt_hint_once(text, hint):
    text = str(text or "")
    hint = str(hint or "").strip()
    if not hint or hint in text:
        return text
    stripped = text.rstrip()
    trailing = text[len(stripped):]
    if stripped and stripped[-1] not in ".!?;:" and not re.search(r"\[\d+-\d+\]$", stripped):
        stripped = f"{stripped}."
    return f"{stripped} {hint}{trailing}"


def format_anchor_frames_for_prompt(frames):
    unique = []
    for frame in frames or []:
        try:
            frame = int(frame)
        except (TypeError, ValueError):
            continue
        if frame not in unique:
            unique.append(frame)
    if not unique:
        return ""
    if len(unique) == 1:
        return str(unique[0])
    return ", ".join(str(x) for x in unique)


def _as_int_pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _as_float_pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if end < start:
        start, end = end, start
    return start, end


def resolve_reference_prompt_segment_id(
    segment_infos,
    prepared_segment_infos,
    *,
    declared_seg_id,
    src_time_range_sec=None,
    dst_range=None,
    time_tolerance_sec=0.05,
):
    """Resolve a reference condition to its original prompt shot.

    ``src_time_range_sec`` is the strongest identity because stage2 references
    are cut from one original visual event. ``dst_range`` is only a latent-grid
    fallback and can differ by one at a causal-VAE boundary. The declared id is
    retained only when neither stronger signal resolves the shot.
    """

    try:
        declared_seg_id = int(declared_seg_id)
    except (TypeError, ValueError):
        declared_seg_id = -1

    raw_by_id = {}
    for seg in segment_infos or []:
        try:
            seg_id = int(seg.get("segment_id", -1))
        except (TypeError, ValueError):
            continue
        if seg_id > 0:
            raw_by_id[seg_id] = _as_float_pair(seg.get("time_range"))

    src_pair = _as_float_pair(src_time_range_sec)
    if src_pair is not None:
        matches = []
        for seg_id, prompt_pair in raw_by_id.items():
            if prompt_pair is None:
                continue
            error = abs(src_pair[0] - prompt_pair[0]) + abs(
                src_pair[1] - prompt_pair[1]
            )
            if (
                abs(src_pair[0] - prompt_pair[0]) <= time_tolerance_sec
                and abs(src_pair[1] - prompt_pair[1]) <= time_tolerance_sec
            ):
                matches.append((error, seg_id, prompt_pair))
        if matches:
            matches.sort()
            _, seg_id, prompt_pair = matches[0]
            return {
                "seg_id": int(seg_id),
                "reason": "src_time_range_sec_exact_prompt_shot",
                "declared_seg_id": declared_seg_id,
                "src_time_range_sec": src_pair,
                "prompt_time_range_sec": prompt_pair,
            }

    dst_pair = _as_int_pair(dst_range)
    prepared_by_id = {}
    for seg in prepared_segment_infos or []:
        try:
            seg_id = int(seg.get("segment_id", -1))
        except (TypeError, ValueError):
            continue
        pair = _as_float_pair(seg.get("time_range"))
        if seg_id > 0 and pair is not None:
            prepared_by_id[seg_id] = pair
    if dst_pair is not None and prepared_by_id:
        overlaps = []
        for seg_id, prompt_pair in prepared_by_id.items():
            overlap = max(
                0.0,
                min(float(dst_pair[1]), prompt_pair[1])
                - max(float(dst_pair[0]), prompt_pair[0]),
            )
            if overlap > 0:
                overlaps.append((overlap, seg_id, prompt_pair))
        if overlaps:
            overlaps.sort(key=lambda item: (-item[0], item[1]))
            overlap, seg_id, prompt_pair = overlaps[0]
            return {
                "seg_id": int(seg_id),
                "reason": "maximum_latent_overlap_prompt_shot",
                "declared_seg_id": declared_seg_id,
                "dst_range": dst_pair,
                "prompt_latent_range": prompt_pair,
                "overlap": float(overlap),
            }

    if declared_seg_id > 0 and declared_seg_id in raw_by_id:
        return {
            "seg_id": declared_seg_id,
            "reason": "declared_seg_id_fallback",
            "declared_seg_id": declared_seg_id,
        }
    return {
        "seg_id": -1,
        "reason": "unresolved_prompt_shot",
        "declared_seg_id": declared_seg_id,
        "src_time_range_sec": src_pair,
        "dst_range": dst_pair,
    }


def validate_reference_prompt_segment_identity(
    segment_infos,
    prepared_segment_infos,
    condition,
    *,
    require_explicit_ranges=True,
):
    """Validate dataset-owned prompt-shot identity without repairing it."""

    if not isinstance(condition, dict):
        return {
            "valid": False,
            "reason": "condition_not_dict",
        }
    try:
        declared_seg_id = int(condition.get("seg_id", -1))
    except (TypeError, ValueError):
        declared_seg_id = -1
    if declared_seg_id <= 0:
        return {
            "valid": False,
            "reason": "invalid_declared_seg_id",
            "declared_seg_id": declared_seg_id,
        }

    dst_range = _as_int_pair(condition.get("dst_range"))
    prompt_dst_range = _as_int_pair(condition.get("prompt_dst_range"))
    target_dst_range = _as_int_pair(condition.get("target_dst_range"))
    if dst_range is None:
        return {
            "valid": False,
            "reason": "invalid_dst_range",
            "declared_seg_id": declared_seg_id,
        }
    if require_explicit_ranges and (
        prompt_dst_range is None or prompt_dst_range != dst_range
    ):
        return {
            "valid": False,
            "reason": "prompt_dst_range_mismatch",
            "declared_seg_id": declared_seg_id,
            "dst_range": dst_range,
            "prompt_dst_range": prompt_dst_range,
        }
    if require_explicit_ranges and (
        target_dst_range is None or target_dst_range != dst_range
    ):
        return {
            "valid": False,
            "reason": "target_dst_range_mismatch",
            "declared_seg_id": declared_seg_id,
            "dst_range": dst_range,
            "target_dst_range": target_dst_range,
        }
    if prompt_dst_range is not None and prompt_dst_range != dst_range:
        return {
            "valid": False,
            "reason": "prompt_dst_range_mismatch",
            "declared_seg_id": declared_seg_id,
            "dst_range": dst_range,
            "prompt_dst_range": prompt_dst_range,
        }
    if target_dst_range is not None and target_dst_range != dst_range:
        return {
            "valid": False,
            "reason": "target_dst_range_mismatch",
            "declared_seg_id": declared_seg_id,
            "dst_range": dst_range,
            "target_dst_range": target_dst_range,
        }

    diagnostic = resolve_reference_prompt_segment_id(
        segment_infos,
        prepared_segment_infos,
        declared_seg_id=declared_seg_id,
        src_time_range_sec=condition.get("src_time_range_sec"),
        dst_range=dst_range,
    )
    resolved_seg_id = int(diagnostic.get("seg_id", -1))
    if resolved_seg_id != declared_seg_id:
        return {
            "valid": False,
            "reason": "declared_seg_id_not_original_prompt_shot",
            "declared_seg_id": declared_seg_id,
            "diagnostic_resolved_seg_id": resolved_seg_id,
            "diagnostic": diagnostic,
        }
    return {
        "valid": True,
        "reason": (
            "dataset_prompt_shot_identity_verified"
            if prompt_dst_range is not None and target_dst_range is not None
            else "legacy_dataset_prompt_shot_identity_verified"
        ),
        "declared_seg_id": declared_seg_id,
        "dst_range": dst_range,
        "diagnostic": diagnostic,
    }


def _anchor_frames_by_segment_infos(segment_infos, anchor_frames):
    out = defaultdict(list)
    for frame in anchor_frames or []:
        try:
            frame = int(frame)
        except (TypeError, ValueError):
            continue
        for idx, seg in enumerate(segment_infos or [], start=1):
            if int(seg.get("segment_id", -1)) <= 0:
                continue
            pair = _as_int_pair(seg.get("time_range"))
            if pair is None:
                continue
            start, end = pair
            if start <= frame < end:
                out[idx].append(frame)
                break
    return out


def rewrite_segment_infos_for_reference_conditions(
    segment_infos,
    *,
    video_seg_ids=None,
    video_ranges=None,
    anchor_frames=None,
):
    """Rewrite training segment_infos to explicit reference prompts.

    This mirrors the post-train prompt semantics without importing post_train.py:
    video reference segments become a reference-video instruction, while image
    anchors append a reference-image instruction to the segment that contains
    the anchor frame.
    """
    if not segment_infos:
        return segment_infos, False

    video_id_set = set()
    for seg_id in video_seg_ids or []:
        try:
            seg_id = int(seg_id)
        except (TypeError, ValueError):
            continue
        if seg_id > 0:
            video_id_set.add(seg_id)

    video_range_set = set()
    for item in video_ranges or []:
        pair = _as_int_pair(item)
        if pair is not None:
            video_range_set.add(pair)

    anchors_by_order = _anchor_frames_by_segment_infos(segment_infos, anchor_frames)
    if not video_id_set and not video_range_set and not anchors_by_order:
        return segment_infos, False

    rewritten = []
    changed = False
    for order, seg in enumerate(segment_infos, start=1):
        new_seg = copy.deepcopy(seg)
        seg_id = int(new_seg.get("segment_id", -1))
        pair = _as_int_pair(new_seg.get("time_range"))
        is_video_ref = seg_id in video_id_set or (pair is not None and pair in video_range_set)

        if is_video_ref:
            prefix = ""
            if seg_id == 1 and "global_prefix" in new_seg:
                prefix = str(new_seg.get("global_prefix", "")).strip()
            body = join_prompt_prefix_body(prefix, REFERENCE_VIDEO_SEGMENT_HINT)
            new_seg["body_text"] = body
            new_seg["text"] = body
            changed = True

        frames_text = format_anchor_frames_for_prompt(anchors_by_order.get(order))
        if frames_text:
            hint = REFERENCE_IMAGE_ANCHOR_HINT_TEMPLATE.format(frames=frames_text)
            body = append_prompt_hint_once(new_seg.get("body_text", new_seg.get("text", "")), hint)
            new_seg["body_text"] = body
            new_seg["text"] = body
            changed = True

        rewritten.append(new_seg)

    return rewritten, changed


def rewrite_bracket_prompt_for_reference_conditions(
    prompt,
    *,
    condition_prompt_seg_ids=None,
    condition_prompt_ranges=None,
    conditions=None,
    global_prompt_prefix="",
):
    if not isinstance(prompt, str):
        return prompt, False

    seg_pattern = re.compile(r"\[(\d+)-(\d+)\]\s*")
    matches = list(seg_pattern.finditer(prompt))
    if not matches:
        return prompt, False

    cond_id_set = set()
    for seg_id in condition_prompt_seg_ids or []:
        try:
            seg_id = int(seg_id)
        except (TypeError, ValueError):
            continue
        if seg_id > 0:
            cond_id_set.add(seg_id)

    cond_range_set = set()
    for item in condition_prompt_ranges or []:
        pair = _as_int_pair(item)
        if pair is not None:
            cond_range_set.add(pair)

    anchors_by_segment = defaultdict(list)
    if isinstance(conditions, dict):
        for anchor in conditions.get("anchors", []) or []:
            if not isinstance(anchor, dict):
                continue
            try:
                frame_idx = int(anchor.get("frame_idx", anchor.get("anchor_frame")))
            except (TypeError, ValueError):
                continue
            for seg_order, match in enumerate(matches, start=1):
                seg_start = int(match.group(1))
                seg_end = int(match.group(2))
                if seg_start <= frame_idx < seg_end:
                    anchors_by_segment[seg_order].append(frame_idx)
                    break

    if not cond_id_set and not cond_range_set and not anchors_by_segment:
        return prompt, False

    rewritten = []
    last_end = 0
    changed = False
    for seg_order, match in enumerate(matches, start=1):
        seg_start = int(match.group(1))
        seg_end = int(match.group(2))
        body_end = matches[seg_order].start() if seg_order < len(matches) else len(prompt)
        segment_text = prompt[last_end:body_end]
        is_video_ref = seg_order in cond_id_set or (seg_start, seg_end) in cond_range_set

        if is_video_ref:
            prefix = segment_text[:match.end() - last_end]
            body = str(global_prompt_prefix or "").strip() if seg_order == 1 else ""
            body = join_prompt_prefix_body(body, REFERENCE_VIDEO_SEGMENT_HINT)
            segment_text = prefix if prefix.endswith(" ") else f"{prefix} "
            segment_text = f"{segment_text}{body} "
            changed = True

        frames_text = format_anchor_frames_for_prompt(anchors_by_segment.get(seg_order))
        if frames_text:
            segment_text = append_prompt_hint_once(
                segment_text,
                REFERENCE_IMAGE_ANCHOR_HINT_TEMPLATE.format(frames=frames_text),
            )
            changed = True

        rewritten.append(segment_text)
        last_end = body_end

    rewritten.append(prompt[last_end:])
    if not changed:
        return prompt, False
    return "".join(rewritten), True
