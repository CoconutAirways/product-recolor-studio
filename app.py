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
from typing import Optional

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from pantone_data import PANTONE_TCX, PANTONE_PMS
from r2_client import from_secrets as build_r2_from_secrets

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NANO_BANANA_PRO_ENDPOINT = "https://api.freepik.com/v1/ai/text-to-image/nano-banana-pro"
ASPECT_RATIO = "1:1"      # hardcoded — square only
RESOLUTION = "4K"         # hardcoded — 4096×4096 native
POLL_INTERVAL_S = 3
POLL_TIMEOUT_S = 360      # Pro can take 60–180s


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
    """Build a plain dict of the secrets our helpers expect."""
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
# Image helpers
# ---------------------------------------------------------------------------
def hex_to_rgb(hex_str: str) -> tuple:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def center_crop_square(img: Image.Image) -> Image.Image:
    """Crop to a centered square — Nano Banana Pro 1:1 expects square input."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


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
        "Photorealistic studio product photography, 4K sharpness, crisp micro-detail, "
        "zero artistic reinterpretation, zero re-styling, zero re-lighting, zero re-cropping."
    )


# ---------------------------------------------------------------------------
# Freepik API — submit + poll
# ---------------------------------------------------------------------------
def submit_nano_banana_pro(prompt: str, reference_url: str, api_key: str) -> dict:
    headers = {
        "x-freepik-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "reference_images": [
            {"image": reference_url, "mime_type": "image/jpeg"},
        ],
        "aspect_ratio": ASPECT_RATIO,
        "resolution": RESOLUTION,
    }
    resp = requests.post(NANO_BANANA_PRO_ENDPOINT, json=payload, headers=headers, timeout=60)
    return {
        "status_code": resp.status_code,
        "body": _safe_json(resp),
    }


def poll_task(task_id: str, api_key: str) -> dict:
    """Poll until COMPLETED/FAILED or timeout."""
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
    """Pull the first generated image URL/base64 from a completed response."""
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
# Auto-download (triggered once per generation)
# ---------------------------------------------------------------------------
def trigger_auto_download(img_bytes: bytes, filename: str):
    b64 = base64.b64encode(img_bytes).decode("ascii")
    components.html(
        f"""
        <a id="auto-dl"
           href="data:image/png;base64,{b64}"
           download="{filename}"
           style="display:none;">dl</a>
        <script>
          setTimeout(function() {{
            document.getElementById('auto-dl').click();
          }}, 200);
        </script>
        """,
        height=0,
    )


# ---------------------------------------------------------------------------
# UI — Sidebar
# ---------------------------------------------------------------------------
st.title("Product Recolor Studio")
st.caption(
    "E-commerce product recoloring · Freepik Nano Banana Pro · "
    f"{ASPECT_RATIO} · {RESOLUTION}"
)

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
    codes = sorted(lookup.keys())
    code = st.selectbox(
        f"Choose from {len(codes)} {system_key} codes",
        codes,
        index=0,
    )
    entry = lookup[code]
    name = entry["name"]
    hex_code = entry["hex"]

    st.markdown("**Preview**")
    st.markdown(
        f"""
        <div style="
            width:100%;
            height:90px;
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

# ---------------------------------------------------------------------------
# UI — Main
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Original")
    upload = st.file_uploader(
        "Upload product image (PNG / JPG, max ~10 MB)",
        type=["png", "jpg", "jpeg", "webp"],
    )
    original = None
    cropped = None
    if upload:
        original = Image.open(upload).convert("RGB")
        cropped = center_crop_square(original)
        st.image(cropped, caption="Center-cropped to 1:1", use_container_width=True)

with col_right:
    st.subheader("Recolored (4K)")
    result_slot = st.empty()
    download_slot = st.empty()

st.divider()

ready = bool(upload and FREEPIK_API_KEY and R2 is not None)
generate = st.button(
    f"Generate → {code} {name}  ·  1:1  ·  4K",
    type="primary",
    use_container_width=True,
    disabled=not ready,
)

# ---------------------------------------------------------------------------
# Generate flow
# ---------------------------------------------------------------------------
if generate and cropped is not None:
    prompt = build_recolor_prompt(name, hex_code, system_key, code)

    with st.expander("Prompt sent to Nano Banana Pro", expanded=False):
        st.code(prompt, language=None)

    jpg_bytes = image_to_jpeg_bytes(cropped)

    r2_key = None
    try:
        # 1) Upload to R2
        with st.spinner("Uploading reference to R2…"):
            r2_key = R2.upload(jpg_bytes, mime="image/jpeg", suffix="jpg")
            ref_url = R2.presigned_get(r2_key, ttl_seconds=600)

        # 2) Submit Nano Banana Pro task
        with st.spinner("Submitting Nano Banana Pro task…"):
            submit = submit_nano_banana_pro(prompt, ref_url, FREEPIK_API_KEY)

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
                # 3) Poll
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
                            result_slot.image(img_bytes, use_container_width=True)
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
                            )
                            # Fire the browser download automatically
                            trigger_auto_download(img_bytes, filename)
                            st.success(
                                f"Done — {code} {name} · 1:1 · 4K · "
                                f"{len(img_bytes) // 1024} KB"
                            )
    finally:
        # 4) Cleanup R2
        if r2_key and R2 is not None:
            R2.delete(r2_key)
