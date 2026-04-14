"""
Product Recolor Studio
======================
Streamlit app that recolors a product photo to a specific Pantone code
using Freepik's Nano Banana Pro (Gemini 3 Pro Image) endpoint.

Hardcoded output specs: aspect_ratio = 1:1, resolution = 4K.
Reference images are uploaded to a private Cloudflare R2 bucket,
served to Freepik via a short-lived presigned URL, then deleted.

Secrets (via st.secrets on Streamlit Community Cloud, or env vars locally):
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
RESOLUTION = "4K"         # hardcoded — 4096 long edge
POLL_INTERVAL_S = 3
POLL_TIMEOUT_S = 360      # Pro can take 60–180s

# Aspect ratio is auto-detected from the source so the original is never cropped.
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
    """Return the closest supported Freepik aspect ratio preset."""
    ratio = w / h
    return min(ASPECT_RATIO_PRESETS.items(), key=lambda kv: abs(kv[1] - ratio))[0]

# Display sizes (pixels) — keeps the UI tight, not full-width
PREVIEW_WIDTH = 460
HISTORY_THUMB_WIDTH = 260


def _secret(name: str, default: str = "") -> str:
    """Read from st.secrets first, fall back to os.environ."""
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

# ---------------------------------------------------------------------------
# Session state — persistent history
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list[dict]: {timestamp, code, name, hex, system, original_bytes, result_bytes}


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
def build_recolor_prompt(name: str, hex_code: str, system: str, code: str) -> str:
    r, g, b = hex_to_rgb(hex_code)
    return (
        f"Recolor the main product in the reference image to Pantone {code} {name} "
        f"(hex {hex_code}, sRGB {r},{g},{b}). "
        "\n\nABSOLUTE PRESERVATION RULES — the output must remain pixel-level identical "
        "to the reference EXCEPT for the fabric/material base color: "
        "exact same product shape, cut, silhouette, proportions, and scale; "
        "every wrinkle, fold, crease, and drape in the exact same position; "
        "all stitching, seams, hems, topstitching, zippers, buttons, drawstrings, eyelets "
        "and any hardware — unchanged in color, material, and position; "
        "all prints, logos, graphics, labels, tags, patches, embroidery — DO NOT recolor, "
        "leave exactly as in the reference; "
        "exact same fabric texture, knit/weave, pile, nap, and material finish; "
        "exact same lighting, shadows, highlights, specular reflections, ambient occlusion; "
        "exact same camera angle, focal length, perspective, framing, composition; "
        "exact same background, floor, props, mannequin, model, skin, hair, accessories. "
        "\n\nThe ONLY change is the base color of the main product surface. "
        "Preserve the full product — do not crop, zoom, re-frame, or cut off any part "
        "of the product. Keep the same framing and whitespace as the reference. "
        "Photorealistic studio product photography, 4K sharpness, crisp micro-detail, "
        "zero artistic reinterpretation, zero re-styling, zero re-lighting, zero re-cropping."
    )


# ---------------------------------------------------------------------------
# Freepik API — submit + poll
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

    # --- Search / filter text input -----------------------------------------
    search = st.text_input(
        "Search (code or name)",
        value="",
        placeholder="e.g. 18-1663  or  Tomato",
        key=f"search_{system_key}",
    ).strip().lower()

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

    # Show code + name in the dropdown so the native typeahead finds both
    def _label(c: str) -> str:
        return f"{c} · {lookup[c]['name']}"

    code = st.selectbox(
        f"{len(filtered)} / {len(all_codes)} {system_key} codes",
        filtered,
        index=0,
        format_func=_label,
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
            st.rerun()

# ---------------------------------------------------------------------------
# UI — Main (upload + latest result)
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("Original")
    upload = st.file_uploader(
        "Upload product image (PNG / JPG, max ~10 MB)",
        type=["png", "jpg", "jpeg", "webp"],
    )
    original = None
    if upload:
        original = Image.open(upload).convert("RGB")
        ow, oh = original.size
        st.image(
            original,
            caption=f"{ow}×{oh}  ·  aspect {detect_aspect_ratio(ow, oh)} (auto)",
            width=PREVIEW_WIDTH,
        )

with col_right:
    st.subheader("Recolored (4K)")
    result_slot = st.empty()
    download_slot = st.empty()

st.divider()

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
    prompt = build_recolor_prompt(name, hex_code, system_key, code)
    aspect_ratio = detect_aspect_ratio(*original.size)

    with st.expander("Prompt sent to Nano Banana Pro", expanded=False):
        st.code(prompt, language=None)

    # Send the ORIGINAL at its native size — no crop, no pad, no resize.
    jpg_bytes = image_to_jpeg_bytes(original)

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
                st.caption(f"Task completed in {elapsed:.0f}s")

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
                            # Persist into history
                            st.session_state.history.append({
                                "timestamp": time.time(),
                                "system": system_key,
                                "code": code,
                                "name": name,
                                "hex": hex_code,
                                "original_bytes": jpg_bytes,
                                "result_bytes": img_bytes,
                            })
                            result_slot.image(img_bytes, width=PREVIEW_WIDTH)
                            filename = (
                                f"recolor_{system_key}_"
                                f"{code.replace(' ', '_').replace('/', '-')}_4K.png"
                            )
                            download_slot.download_button(
                                label="⬇ Download 4K PNG",
                                data=img_bytes,
                                file_name=filename,
                                mime="image/png",
                                use_container_width=True,
                                key=f"dl_latest_{time.time()}",
                            )
                            st.success(
                                f"Done — {code} {name} · {aspect_ratio} · 4K · "
                                f"{len(img_bytes) // 1024} KB"
                            )
    finally:
        if r2_key and R2 is not None:
            R2.delete(r2_key)

# ---------------------------------------------------------------------------
# History timeline (persistent — downloads don't remove entries)
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    st.subheader(f"History · {len(st.session_state.history)} generation(s)")

    for idx, item in enumerate(reversed(st.session_state.history)):
        real_idx = len(st.session_state.history) - 1 - idx
        ts = datetime.fromtimestamp(item["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 1, 1.2], gap="medium")
            with c1:
                st.caption("Original")
                st.image(item["original_bytes"], width=HISTORY_THUMB_WIDTH)
            with c2:
                st.caption("Recolored (4K)")
                st.image(item["result_bytes"], width=HISTORY_THUMB_WIDTH)
            with c3:
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                      <div style="
                        width:28px;height:28px;border-radius:6px;
                        background:{item['hex']};
                        border:1px solid rgba(0,0,0,0.2);
                      "></div>
                      <div style="font-size:15px;">
                        <strong>{item['code']}</strong> · {item['name']}<br/>
                        <code style="font-size:12px;">{item['hex']}</code>
                        <span style="opacity:0.6;font-size:12px;"> · {item['system']}</span>
                      </div>
                    </div>
                    <div style="opacity:0.6;font-size:12px;margin-bottom:10px;">{ts}</div>
                    """,
                    unsafe_allow_html=True,
                )
                filename = (
                    f"recolor_{item['system']}_"
                    f"{item['code'].replace(' ', '_').replace('/', '-')}_4K.png"
                )
                st.download_button(
                    label="⬇ Download 4K PNG",
                    data=item["result_bytes"],
                    file_name=filename,
                    mime="image/png",
                    use_container_width=True,
                    key=f"dl_hist_{real_idx}",
                )
                if st.button(
                    "✕ Remove",
                    key=f"rm_hist_{real_idx}",
                    use_container_width=True,
                ):
                    st.session_state.history.pop(real_idx)
                    st.rerun()
