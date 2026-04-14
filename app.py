"""
Product Recolor Studio
======================
Streamlit app that recolors a product photo to a specific Pantone code
using Freepik's Nano Banana Pro (Gemini 3 Pro Image) endpoint.

Reference images are uploaded to a private Cloudflare R2 bucket,
served to Freepik via a short-lived presigned URL, then deleted.

The user can add free-text "Additional instructions" (e.g. "keep the white
hood lining white") that get appended to the recolor prompt to fix
multi-color edge cases.

Secrets / env vars:
    FREEPIK_API_KEY
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET
"""
from __future__ import annotations

import base64
import io
import os
import time
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
from PIL import Image

from pantone_data import PANTONE_TCX, PANTONE_PMS
from r2_client import from_secrets as build_r2_from_secrets

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NANO_BANANA_PRO_ENDPOINT = "https://api.freepik.com/v1/ai/text-to-image/nano-banana-pro"
RESOLUTION = "4K"
POLL_INTERVAL_S = 3
POLL_TIMEOUT_S = 360

PREVIEW_WIDTH = 520  # display width in px for both gallery images

ASPECT_RATIO_PRESETS = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
}


def detect_aspect_ratio(w: int, h: int) -> str:
    ratio = w / h
    return min(ASPECT_RATIO_PRESETS.items(), key=lambda kv: abs(kv[1] - ratio))[0]


def _secret(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name, None)
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.environ.get(name, default).strip()


def _secrets_map() -> dict:
    keys = [
        "FREEPIK_API_KEY",
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
    ]
    return {k: _secret(k) for k in keys}


SECRETS = _secrets_map()
FREEPIK_API_KEY = SECRETS["FREEPIK_API_KEY"]
R2 = build_r2_from_secrets(SECRETS)

st.set_page_config(
    page_title="Product Recolor Studio",
    page_icon="🎨",
    layout="wide",
)

# Inject CSS — larger, friendlier file uploader drop zone
st.markdown(
    """
    <style>
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] > section,
    [data-testid="stFileUploaderDropzone"] {
        min-height: 180px !important;
        padding: 2rem 1.5rem !important;
        border-width: 2px !important;
        border-style: dashed !important;
    }
    [data-testid="stFileUploader"] section > div:first-child,
    [data-testid="stFileUploaderDropzone"] > div:first-child {
        font-size: 1.05rem !important;
    }
    [data-testid="stFileUploader"] button {
        padding: 0.55rem 1.25rem !important;
        font-size: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts
if "active_idx" not in st.session_state:
    st.session_state.active_idx = 0
if "extra_instructions" not in st.session_state:
    st.session_state.extra_instructions = ""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def hex_to_rgb(hex_str: str) -> tuple:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def image_to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
def build_recolor_prompt(
    name: str,
    hex_code: str,
    system: str,
    code: str,
    extra: str = "",
) -> str:
    r, g, b = hex_to_rgb(hex_code)

    extra_block = ""
    if extra.strip():
        extra_block = (
            "\n\nADDITIONAL USER-SPECIFIED PRESERVATION RULES (these override any "
            "ambiguity — follow them exactly):\n"
            f"{extra.strip()}"
        )

    return (
        f"Recolor the main product in the reference image to Pantone {code} {name} "
        f"(hex {hex_code}, sRGB {r},{g},{b})."
        f"{extra_block}"
        "\n\nABSOLUTE PRESERVATION RULES — the output must remain pixel-level identical "
        "to the reference EXCEPT for the primary fabric/material base color: "
        "exact same product shape, cut, silhouette, proportions, and scale; "
        "every wrinkle, fold, crease, and drape in the exact same position; "
        "all stitching, seams, hems, topstitching, zippers, buttons, drawstrings, eyelets "
        "and any hardware — unchanged in color, material, and position; "
        "all prints, logos, graphics, labels, tags, patches, embroidery — DO NOT recolor, "
        "leave exactly as in the reference; "
        "all contrast panels, side stripes, hood linings, pocket linings, and any "
        "secondary-color areas — DO NOT recolor, leave their original colors untouched; "
        "exact same fabric texture, knit/weave, pile, nap, and material finish; "
        "exact same lighting, shadows, highlights, specular reflections, ambient occlusion; "
        "exact same camera angle, focal length, perspective, framing, composition; "
        "exact same background, floor, props, mannequin, model, skin, hair, accessories."
        "\n\nThe ONLY change is the base color of the main product surface. "
        "Preserve the full product — do not crop, zoom, re-frame, or cut off any part "
        "of the product. Keep the same framing and whitespace as the reference. "
        "Photorealistic studio product photography, 4K sharpness, crisp micro-detail, "
        "zero artistic reinterpretation, zero re-styling, zero re-lighting, zero re-cropping."
    )


# ---------------------------------------------------------------------------
# Freepik API
# ---------------------------------------------------------------------------
def submit_nano_banana_pro(
    prompt: str,
    reference_url: str,
    api_key: str,
    aspect_ratio: str,
) -> dict:
    headers = {
        "x-freepik-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "reference_images": [
            {"image": reference_url, "mime_type": "image/jpeg"},
        ],
        "aspect_ratio": aspect_ratio,
        "resolution": RESOLUTION,
    }
    resp = requests.post(NANO_BANANA_PRO_ENDPOINT, json=payload, headers=headers, timeout=60)
    return {"status_code": resp.status_code, "body": _safe_json(resp)}


def poll_task(task_id: str, api_key: str) -> dict:
    url = f"{NANO_BANANA_PRO_ENDPOINT}/{task_id}"
    headers = {"x-freepik-api-key": api_key}
    t0 = time.time()
    last_body = None
    while time.time() - t0 < POLL_TIMEOUT_S:
        resp = requests.get(url, headers=headers, timeout=30)
        body = _safe_json(resp)
        last_body = body
        data = (body or {}).get("data") or {}
        status = (data.get("status") or "").upper()
        if status in ("COMPLETED", "SUCCESS", "SUCCEEDED"):
            return {"ok": True, "body": body}
        if status in ("FAILED", "ERROR"):
            return {"ok": False, "body": body, "reason": "task failed"}
        time.sleep(POLL_INTERVAL_S)
    return {"ok": False, "body": last_body, "reason": "timeout"}


def _safe_json(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except ValueError:
        return {"raw_text": resp.text}


def extract_image_url(body: dict) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    data = body.get("data") or {}
    generated = data.get("generated") or []
    if isinstance(generated, list) and generated:
        first = generated[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return (
                first.get("url")
                or first.get("image")
                or first.get("base64")
                or first.get("b64_json")
            )
    return None


def fetch_bytes(url_or_b64: str) -> bytes:
    if url_or_b64.startswith("http"):
        r = requests.get(url_or_b64, timeout=90)
        r.raise_for_status()
        return r.content
    raw = url_or_b64.split(",", 1)[-1]
    return base64.b64decode(raw)


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.title("Product Recolor Studio")
st.caption(
    "E-commerce product recoloring · Freepik Nano Banana Pro · "
    f"original aspect · {RESOLUTION}"
)

# ---------------------------------------------------------------------------
# UI — Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Pantone System")
    system_label = st.selectbox(
        "Select system",
        ["TCX (Textile)", "PMS (Solid Coated)"],
        help="TCX = textile/fabric references. PMS = print/graphics references.",
    )
    system_key = "TCX" if system_label.startswith("TCX") else "PMS"
    lookup = PANTONE_TCX if system_key == "TCX" else PANTONE_PMS

    st.header("2 · Pantone Code")

    search_raw = st.text_input(
        "Search (code or name)",
        placeholder="e.g. 18-1663  or  Tomato",
        key=f"search_{system_key}",
    )
    search = (search_raw or "").strip().lower()

    all_codes = sorted(lookup.keys())
    if search:
        filtered = [
            c for c in all_codes
            if search in c.lower() or search in lookup[c]["name"].lower()
        ]
    else:
        filtered = all_codes

    if not filtered:
        st.warning("No matches — showing full list.")
        filtered = all_codes

    def _label(c: str) -> str:
        return f"{c} · {lookup[c]['name']}"

    code = st.selectbox(
        f"{len(filtered)} / {len(all_codes)} {system_key} codes",
        filtered,
        index=0,
        format_func=_label,
        key=f"pantone_select_{system_key}_{len(filtered)}",
    )
    entry = lookup[code]
    name = entry["name"]
    hex_code = entry["hex"]

    st.markdown("**Preview**")
    st.markdown(
        f"""
        <div style="
            width:100%;
            height:80px;
            background:{hex_code};
            border-radius:10px;
            border:1px solid rgba(0,0,0,0.15);
            box-shadow:0 1px 3px rgba(0,0,0,0.08);
        "></div>
        <div style="margin-top:10px;font-size:13px;line-height:1.5;">
            <strong>{code}</strong><br/>
            {name}<br/>
            <code>{hex_code}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Backend status**")
    if FREEPIK_API_KEY:
        st.success("Freepik API key loaded")
    else:
        st.error("FREEPIK_API_KEY missing")
    if R2 is not None:
        st.success("R2 bucket configured")
    else:
        st.error("R2 credentials missing")

    st.divider()
    if st.session_state.history:
        if st.button(f"🗑 Clear history ({len(st.session_state.history)})"):
            st.session_state.history = []
            st.session_state.active_idx = 0
            st.rerun()

# ---------------------------------------------------------------------------
# UI — Upload (full width)
# ---------------------------------------------------------------------------
st.subheader("Upload")
upload = st.file_uploader(
    "Product image (PNG / JPG / WEBP, max ~10 MB)",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
)

# Process upload: load original
original: Optional[Image.Image] = None
original_bytes: Optional[bytes] = None
if upload:
    original = Image.open(upload).convert("RGB")
    original_bytes = image_to_jpeg_bytes(original)

# ---------------------------------------------------------------------------
# UI — Additional instructions (manual prompt append)
# ---------------------------------------------------------------------------
st.markdown(
    "**Additional instructions** (optional) · appended to the recolor prompt"
)
st.session_state.extra_instructions = st.text_area(
    "extra",
    value=st.session_state.extra_instructions,
    height=90,
    placeholder=(
        "e.g. The white hood lining must stay white. The three white "
        "side stripes must stay white. The drawstrings must stay white."
    ),
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# UI — Gallery (prev/next + side-by-side pair)
# ---------------------------------------------------------------------------
st.divider()

history = st.session_state.history
has_history = len(history) > 0

# Clamp active_idx
if has_history:
    st.session_state.active_idx = max(
        0, min(st.session_state.active_idx, len(history) - 1)
    )
active_idx = st.session_state.active_idx

# Navigation row (only when there is at least one generation)
if has_history:
    nav_l, nav_c, nav_r = st.columns([1, 3, 1])
    with nav_l:
        if st.button(
            "← Prev",
            disabled=(active_idx == 0),
            use_container_width=True,
            key="nav_prev",
        ):
            st.session_state.active_idx = max(0, active_idx - 1)
            st.rerun()
    with nav_c:
        cur = history[active_idx]
        ts = datetime.fromtimestamp(cur["timestamp"]).strftime("%H:%M:%S")
        st.markdown(
            f"""
            <div style="text-align:center;padding:6px 0;">
              <strong style="font-size:15px;">{active_idx + 1} / {len(history)}</strong>
              &nbsp;·&nbsp;
              <span style="display:inline-block;width:14px;height:14px;
                background:{cur['hex']};border-radius:3px;
                border:1px solid rgba(255,255,255,0.25);
                vertical-align:middle;margin-right:6px;"></span>
              <strong>{cur['code']}</strong> · {cur['name']}
              <span style="opacity:0.55;"> · {ts}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav_r:
        if st.button(
            "Next →",
            disabled=(active_idx >= len(history) - 1),
            use_container_width=True,
            key="nav_next",
        ):
            st.session_state.active_idx = min(len(history) - 1, active_idx + 1)
            st.rerun()

col_left, col_right = st.columns(2, gap="medium")

with col_left:
    st.markdown("**Original**")
    # Priority: show the CURRENT upload if one is present (fresh uploads always
    # take precedence). Otherwise fall back to the active history entry's
    # original so navigation still makes sense when browsing past generations.
    if original is not None:
        st.image(original, width=PREVIEW_WIDTH)
    elif has_history:
        st.image(history[active_idx]["original_bytes"], width=PREVIEW_WIDTH)
    else:
        st.info("Upload a product image to start.")

with col_right:
    st.markdown("**Recolored (4K)**")
    if has_history:
        cur = history[active_idx]
        st.image(cur["result_bytes"], width=PREVIEW_WIDTH)
        fname = (
            f"recolor_{cur['system']}_"
            f"{cur['code'].replace(' ', '_').replace('/', '-')}_4K.png"
        )
        dl_col, rm_col = st.columns([3, 1])
        with dl_col:
            st.download_button(
                label="⬇ Download 4K PNG",
                data=cur["result_bytes"],
                file_name=fname,
                mime="image/png",
                use_container_width=True,
                key=f"dl_active_{active_idx}_{cur['timestamp']}",
            )
        with rm_col:
            if st.button(
                "✕ Remove",
                use_container_width=True,
                key=f"rm_active_{active_idx}_{cur['timestamp']}",
            ):
                st.session_state.history.pop(active_idx)
                st.session_state.active_idx = max(0, active_idx - 1)
                st.rerun()
    else:
        st.info("Pick a Pantone and click Generate.")

st.divider()

# ---------------------------------------------------------------------------
# Generate button
# ---------------------------------------------------------------------------
ready = bool(upload and FREEPIK_API_KEY and R2 is not None)
aspect_label = detect_aspect_ratio(*original.size) if original is not None else "—"
generate = st.button(
    f"Generate → {code} {name}  ·  {aspect_label}  ·  4K",
    type="primary",
    use_container_width=True,
    disabled=not ready,
)

# ---------------------------------------------------------------------------
# Generate flow
# ---------------------------------------------------------------------------
if generate and original is not None:
    extra = st.session_state.extra_instructions or ""
    prompt = build_recolor_prompt(name, hex_code, system_key, code, extra)
    aspect_ratio = detect_aspect_ratio(*original.size)

    with st.expander("Prompt sent to Nano Banana Pro", expanded=False):
        st.code(prompt, language=None)

    jpg_bytes = original_bytes  # native size, no crop/pad/resize

    r2_key = None
    try:
        with st.spinner("Uploading reference to R2…"):
            r2_key = R2.upload(jpg_bytes, mime="image/jpeg", suffix="jpg")
            ref_url = R2.presigned_get(r2_key, ttl_seconds=600)

        with st.spinner("Submitting Nano Banana Pro task…"):
            submit = submit_nano_banana_pro(
                prompt, ref_url, FREEPIK_API_KEY, aspect_ratio
            )

        if submit["status_code"] not in (200, 201):
            st.error(f"Submit failed (HTTP {submit['status_code']})")
            with st.expander("Raw response", expanded=True):
                st.json(submit["body"])
        else:
            task_id = ((submit["body"] or {}).get("data") or {}).get("task_id")
            if not task_id:
                st.error("No task_id in submit response")
                with st.expander("Raw response", expanded=True):
                    st.json(submit["body"])
            else:
                t0 = time.time()
                with st.spinner(
                    f"Generating {code} {name} at 4K… (task {task_id[:8]}…)"
                ):
                    final = poll_task(task_id, FREEPIK_API_KEY)
                elapsed = time.time() - t0

                if not final["ok"]:
                    st.error(f"Task did not complete ({final.get('reason')})")
                    with st.expander("Raw response", expanded=True):
                        st.json(final["body"])
                else:
                    img_ref = extract_image_url(final["body"])
                    if not img_ref:
                        st.warning("Completed but no image URL in response.")
                        with st.expander("Raw response", expanded=True):
                            st.json(final["body"])
                    else:
                        try:
                            img_bytes = fetch_bytes(img_ref)
                        except Exception as e:
                            st.error(f"Failed to fetch image: {e}")
                        else:
                            st.session_state.history.append({
                                "timestamp": time.time(),
                                "system": system_key,
                                "code": code,
                                "name": name,
                                "hex": hex_code,
                                "original_bytes": jpg_bytes,
                                "result_bytes": img_bytes,
                                "extra_instructions": extra,
                                "elapsed_s": elapsed,
                            })
                            # Jump to the newest entry
                            st.session_state.active_idx = len(st.session_state.history) - 1
                            st.success(
                                f"Done in {elapsed:.0f}s — "
                                f"{code} {name} · {aspect_ratio} · 4K · "
                                f"{len(img_bytes) // 1024} KB"
                            )
                            st.rerun()
    finally:
        if r2_key and R2 is not None:
            R2.delete(r2_key)
