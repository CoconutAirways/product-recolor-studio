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
import html as html_mod
import io
import os
import time
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
from PIL import Image
from streamlit.components.v1 import html as components_html

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

    /* Primary Generate button — petrol */
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #1A535C !important;
        border-color: #1A535C !important;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #144148 !important;
        border-color: #144148 !important;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:disabled,
    button[data-testid="stBaseButton-primary"]:disabled {
        background-color: #1A535C !important;
        border-color: #1A535C !important;
        color: rgba(255,255,255,0.55) !important;
        opacity: 0.6 !important;
    }

    /* Constrain gallery columns to image width so X and Download
       sit flush with / under the 520px image. */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        max-width: 520px;
    }

    /* Auto-width download button, horizontally centered under image */
    [data-testid="stDownloadButton"] {
        text-align: center !important;
        margin-top: 10px !important;
        width: 100% !important;
        display: block !important;
    }
    [data-testid="stDownloadButton"] button {
        display: inline-block !important;
        width: auto !important;
        min-width: 0 !important;
        padding: 0.4rem 1.1rem !important;
        font-size: 13px !important;
    }

    /* Square nav buttons (← → ✕) — sit inside the 4-column nav row.
       The :has() selector uniquely targets that row because it's the
       only horizontal block on the page with a 4th column child. */
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4)) button[kind="secondary"],
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(4)) button[kind="secondary"] {
        height: 40px !important;
        min-height: 40px !important;
        width: 40px !important;
        min-width: 40px !important;
        aspect-ratio: 1 / 1 !important;
        padding: 0 !important;
        font-size: 16px !important;
        line-height: 1 !important;
    }

    /* Gallery header alignment — both columns get an equal-height
       header area so the image top edges line up exactly. */
    .gallery-header {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        display: flex !important;
        align-items: center !important;
        padding: 0 4px !important;
        margin: 0 0 8px 0 !important;
        box-sizing: border-box !important;
        font-size: 13px !important;
    }
    /* Force the nav row horizontal block to exactly match the left
       column's gallery-header height. */
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4)),
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(4)) {
        min-height: 48px !important;
        max-height: 48px !important;
        margin-bottom: 8px !important;
        align-items: center !important;
    }
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4)) [data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(4)) [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4)) [data-testid="stElementContainer"],
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(4)) [data-testid="stElementContainer"] {
        margin: 0 !important;
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
if "pending_task" not in st.session_state:
    # Non-blocking generation state. While populated, every rerun polls
    # once and shows a banner. The user can click around freely — the
    # Freepik task keeps running on Freepik's servers regardless.
    st.session_state.pending_task = None
if "upload_cache" not in st.session_state:
    # Cached upload so it survives Streamlit reruns independently of
    # the file_uploader widget's internal state. Shape:
    # { "id": str, "bytes": bytes }
    st.session_state.upload_cache = None


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
            "\n\nADDITIONAL USER-SPECIFIED INSTRUCTIONS (these override any "
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

# Persistent "generating" banner — survives prev/next clicks
if st.session_state.pending_task:
    _pt = st.session_state.pending_task
    _elapsed = int(time.time() - _pt["t0"])
    st.markdown(
        f"""
        <div style="
            background:#1A535C;
            color:#FFFFFF;
            padding:12px 18px;
            border-radius:6px;
            margin:8px 0 12px 0;
            font-size:14px;
            display:flex;
            align-items:center;
            gap:10px;
        ">
          <span style="font-size:16px;">⏳</span>
          <span>Generating <strong>{_pt['code']} {_pt['name']}</strong>
                · {_elapsed}s elapsed · you can browse older results while
                this runs</span>
        </div>
        """,
        unsafe_allow_html=True,
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

    all_codes = sorted(lookup.keys())
    state_key = f"pantone_input_{system_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = all_codes[0] if all_codes else ""

    # Plain text_input backing store — an injected datalist turns it
    # into a native combobox (Chrome shows the full list on click and
    # filters live as the user types digits).
    pantone_input = st.text_input(
        "Pantone code",
        key=state_key,
        label_visibility="collapsed",
        help="Click once → existing code is auto-selected. Start typing "
             "to replace it; the list filters live.",
    )
    q = (pantone_input or "").strip()

    # ------------------------------------------------------------------
    # Datalist + select-all-on-focus + auto-commit-on-exact-match.
    #
    # Injected into the parent document because Streamlit's text_input
    # lives outside this component iframe. A MutationObserver re-wires
    # the input on every rerun. Strategy:
    #   • focus / click  → input.select()     (overwrite-on-type UX)
    #   • 'input'        → if value is an exact valid code, blur()
    #                      which makes Streamlit commit and rerun. The
    #                      observer then refocuses the new input and
    #                      puts the cursor at the end so typing can
    #                      continue seamlessly.
    #   • 'change'       → datalist pick → blur() to commit.
    # Partial-typing values do NOT commit — the preview only updates
    # once a full code is matched or the user clicks elsewhere.
    # ------------------------------------------------------------------
    options_html = "".join(
        f'<option value="{html_mod.escape(c)}">'
        f'{html_mod.escape(lookup[c]["name"])}</option>'
        for c in all_codes
    )
    valid_codes_js = "[" + ",".join(f'"{c}"' for c in all_codes) + "]"
    list_id = f"pantone-options-{system_key}"
    input_label = "Pantone code"
    components_html(
        f"""
        <script>
        (function() {{
          const parentDoc = window.parent.document;
          const parentWin = window.parent;
          const LIST_ID = "{list_id}";
          const LABEL   = {repr(input_label)};
          const VALID   = new Set({valid_codes_js});

          // 1. Build / refresh the datalist in the parent document.
          let dl = parentDoc.getElementById(LIST_ID);
          if (!dl) {{
            dl = parentDoc.createElement('datalist');
            dl.id = LIST_ID;
            parentDoc.body.appendChild(dl);
          }}
          dl.innerHTML = `{options_html}`;

          // Remove stale datalists from the other Pantone system.
          parentDoc.querySelectorAll('datalist[id^="pantone-options-"]')
            .forEach(el => {{ if (el.id !== LIST_ID) el.remove(); }});

          function attach() {{
            const inp = parentDoc.querySelector(
              'input[aria-label="' + LABEL + '"]'
            );
            if (!inp) return;

            if (inp.getAttribute('list') !== LIST_ID) {{
              inp.setAttribute('list', LIST_ID);
              inp.removeAttribute('autocomplete');
            }}

            if (!inp.__pantoneWired) {{
              inp.__pantoneWired = true;

              // Single-click → whole value selected → overwrite-on-type.
              // Strategy: on mouseup, preventDefault (so the browser
              // does NOT place the cursor where clicked) and then
              // programmatically select(). Works for both first-click
              // (fires after focus) and repeat clicks on an already
              // focused input.
              const selectAll = () => {{
                try {{ inp.select(); }} catch (e) {{}}
              }};
              inp.addEventListener('focus', () => {{
                requestAnimationFrame(selectAll);
              }});
              inp.addEventListener('mouseup', (e) => {{
                e.preventDefault();
                setTimeout(selectAll, 0);
              }});

              // React/Streamlit tracks the input value via its own
              // "value tracker" hack. When the browser sets input.value
              // programmatically (datalist click), React may or may
              // not notice — we force a clean notification by calling
              // React's native setter and dispatching a synthetic
              // input event.
              //
              // Recursion guard: `dispatching` is a closure flag set
              // true BEFORE dispatchEvent and cleared in the finally
              // block. JS event dispatch is synchronous, so when our
              // dispatched 'input' event re-fires this listener, the
              // flag is still true and we skip. The finally clears it
              // the instant dispatch returns, so the flag can NEVER
              // get stuck — no timing races, no stale state across
              // reruns, no value-comparison traps.
              const nativeSetter = Object.getOwnPropertyDescriptor(
                parentWin.HTMLInputElement.prototype, 'value'
              ).set;
              let dispatching = false;

              const commitValue = (v, refocus) => {{
                if (dispatching) return;
                if (!VALID.has(v)) return;
                dispatching = true;
                try {{
                  nativeSetter.call(inp, v);
                  inp.dispatchEvent(
                    new Event('input', {{ bubbles: true }})
                  );
                }} catch (e) {{}}
                finally {{
                  dispatching = false;
                }}
                parentWin.__pantoneRefocus = !!refocus;
                // Let React finish its onChange cycle, then blur so
                // Streamlit commits the new value to Python.
                setTimeout(() => {{
                  try {{ inp.blur(); }} catch (e) {{}}
                }}, 0);
              }};

              // Exact-match auto-commit while typing — user never has
              // to press Enter.
              inp.addEventListener('input', () => {{
                if (dispatching) return;
                const v = (inp.value || "").trim().toUpperCase();
                if (VALID.has(v)) commitValue(v, true);
              }});

              // Datalist pick fires 'change'. Commit, keep focus lost
              // (user just clicked, they don't need the cursor back).
              inp.addEventListener('change', () => {{
                if (dispatching) return;
                const v = (inp.value || "").trim().toUpperCase();
                if (VALID.has(v)) commitValue(v, false);
              }});
            }}

            // After a self-triggered blur-commit, Streamlit reruns and
            // the input is recreated — put the cursor back at the end
            // so the user can keep typing.
            if (parentWin.__pantoneRefocus) {{
              parentWin.__pantoneRefocus = false;
              setTimeout(() => {{
                try {{
                  inp.focus();
                  const end = (inp.value || "").length;
                  inp.setSelectionRange(end, end);
                }} catch (e) {{}}
              }}, 0);
            }}
          }}
          attach();

          if (!window.__pantoneObserver) {{
            window.__pantoneObserver = new MutationObserver(attach);
            window.__pantoneObserver.observe(parentDoc.body, {{
              childList: true, subtree: true
            }});
          }}
        }})();
        </script>
        """,
        height=0,
    )

    q_up = q.upper()
    if q_up in lookup:
        code = q_up
        entry = lookup[code]
        name = entry["name"]
        hex_code = entry["hex"]
    else:
        code = None
        name = None
        hex_code = None
        if q:
            st.caption(f"No exact {system_key} match for '{q}' yet")

    if hex_code:
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
# Small Clear link rendered under the uploader — only when we have a
# cached image. Kept outside any columns wrapper so it can't mess with
# the file_uploader widget's identity across reruns.
if st.session_state.upload_cache is not None:
    if st.button(
        "✕ Clear uploaded image",
        key="clear_upload",
        type="secondary",
        help="Remove the uploaded image",
    ):
        st.session_state.upload_cache = None
        st.rerun()

# ---------------------------------------------------------------------------
# Upload handling — sticky cache that survives every rerun
# ---------------------------------------------------------------------------
# Streamlit's UploadedFile object is unreliable across reruns triggered
# by other widgets: its internal read pointer can end up at EOF (which
# causes Image.open() to fail silently), and under some conditions the
# widget itself can briefly return None after an internal remount. We
# sidestep both by:
#   1. Reading the raw bytes once into session_state on first sight.
#   2. Keeping that cache "sticky" — NEVER auto-dropping it just because
#      `upload is None` on a given rerun. The cache is only replaced
#      when a genuinely different file_id is uploaded, or explicitly
#      cleared via the Clear button above.
if upload is not None:
    upload_id = getattr(upload, "file_id", None) or f"{upload.name}:{upload.size}"
    cached = st.session_state.upload_cache
    if cached is None or cached.get("id") != upload_id:
        try:
            raw = upload.getvalue()  # does not move the pointer
        except Exception:
            try:
                upload.seek(0)
                raw = upload.read()
            except Exception:
                raw = b""
        # Only overwrite the cache if we actually read real bytes.
        # Streamlit can occasionally hand us an UploadedFile with an
        # already-drained buffer after a rerun; an empty read would
        # clobber a perfectly good cached image and blank the gallery.
        if raw:
            st.session_state.upload_cache = {
                "id": upload_id,
                "bytes": raw,
                "name": upload.name,
            }
# NOTE: the `else` branch (upload is None) is intentionally absent.
# Keep the cache alive across transient None states.

original: Optional[Image.Image] = None
original_bytes: Optional[bytes] = None
original_name: str = "image"
if st.session_state.upload_cache and st.session_state.upload_cache.get("bytes"):
    try:
        raw = st.session_state.upload_cache["bytes"]
        original = Image.open(io.BytesIO(raw)).convert("RGB")
        original_bytes = image_to_jpeg_bytes(original)
        original_name = st.session_state.upload_cache.get("name") or "image"
    except Exception as e:
        st.error(f"Could not read uploaded image: {e}")
        original = None
        original_bytes = None

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

# Live prompt preview — shows what will be sent to Nano Banana Pro
# given the current Pantone selection + extra instructions.
if code is not None and name is not None and hex_code is not None:
    with st.expander("Prompt preview (click to expand)", expanded=False):
        _preview = build_recolor_prompt(
            name, hex_code, system_key, code,
            st.session_state.extra_instructions or "",
        )
        st.code(_preview, language=None)

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

PETROL_BOX = (
    "<div style=\""
    "background:#1A535C;"
    "color:#FFFFFF;"
    "padding:14px 18px;"
    "border-radius:6px;"
    "font-size:14px;"
    "height:48px;"
    "display:flex;"
    "align-items:center;"
    "box-sizing:border-box;"
    "\">{text}</div>"
)

GALLERY_HEADER = (
    "<div class='gallery-header'><strong>{text}</strong></div>"
)

col_left, col_right = st.columns(2, gap="medium")

with col_left:
    st.markdown(GALLERY_HEADER.format(text="Original"), unsafe_allow_html=True)
    # Priority: show the CURRENT upload if one is present (fresh uploads always
    # take precedence). Otherwise fall back to the active history entry's
    # original so navigation still makes sense when browsing past generations.
    if original is not None:
        st.image(original, width=PREVIEW_WIDTH)
    elif has_history:
        st.image(history[active_idx]["original_bytes"], width=PREVIEW_WIDTH)
    else:
        st.markdown(
            PETROL_BOX.format(text="Upload a product image to start."),
            unsafe_allow_html=True,
        )

with col_right:
    if not has_history:
        # Matching gallery header so the petrol placeholder sits at the
        # same y-coordinate as the left column's image.
        st.markdown(GALLERY_HEADER.format(text="Result"), unsafe_allow_html=True)
        st.markdown(
            PETROL_BOX.format(text="Pick a Pantone and click Generate."),
            unsafe_allow_html=True,
        )
    else:
        cur = history[active_idx]
        ts = datetime.fromtimestamp(cur["timestamp"]).strftime("%H:%M:%S")

        # Compact nav row: ← → · caption · X (flush right with image)
        nav_prev, nav_next, nav_caption, nav_remove = st.columns([1, 1, 9, 1])
        with nav_prev:
            if st.button(
                "←",
                disabled=(active_idx == 0),
                key="nav_prev",
                use_container_width=True,
            ):
                st.session_state.active_idx = max(0, active_idx - 1)
                st.rerun()
        with nav_next:
            if st.button(
                "→",
                disabled=(active_idx >= len(history) - 1),
                key="nav_next",
                use_container_width=True,
            ):
                st.session_state.active_idx = min(len(history) - 1, active_idx + 1)
                st.rerun()
        with nav_caption:
            st.markdown(
                f"""
                <div style="padding:8px 0 0 4px;text-align:left;font-size:13px;line-height:1.3;">
                  <strong>{active_idx + 1} / {len(history)}</strong>
                  &nbsp;·&nbsp;
                  <span style="display:inline-block;width:12px;height:12px;
                    background:{cur['hex']};border-radius:2px;
                    border:1px solid rgba(255,255,255,0.25);
                    vertical-align:middle;margin-right:5px;"></span>
                  <strong>{cur['code']}</strong> · {cur['name']}
                  <span style="opacity:0.55;"> · {ts}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with nav_remove:
            if st.button(
                "✕",
                use_container_width=True,
                key=f"rm_active_{active_idx}_{cur['timestamp']}",
                help="Remove this generation",
            ):
                st.session_state.history.pop(active_idx)
                st.session_state.active_idx = max(0, active_idx - 1)
                st.rerun()

        st.image(cur["result_bytes"], width=PREVIEW_WIDTH)

        # Build filename: <original_basename>_<pantone_code>.png
        base = os.path.splitext(cur.get("original_name") or "image")[0]
        safe_code = cur["code"].replace(" ", "_").replace("/", "-")
        fname = f"{base}_{safe_code}.png"

        # Custom save button — uses the File System Access API
        # (showSaveFilePicker) to prompt for a save location. Falls back
        # to a plain <a download> for browsers without FSA support.
        b64_data = base64.b64encode(cur["result_bytes"]).decode("ascii")
        btn_uid = f"dl_{active_idx}_{int(cur['timestamp'] * 1000)}"
        components_html(
            f"""
            <style>
              #{btn_uid}_wrap {{
                display: flex;
                justify-content: center;
                margin-top: 6px;
                font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
              }}
              #{btn_uid} {{
                display: inline-block;
                padding: 0.4rem 1.1rem;
                font-size: 13px;
                background: #ffffff;
                color: #31333f;
                border: 1px solid rgba(49,51,63,0.2);
                border-radius: 0.5rem;
                cursor: pointer;
                transition: all 0.15s;
              }}
              #{btn_uid}:hover {{
                border-color: #1A535C;
                color: #1A535C;
              }}
              #{btn_uid}:active {{
                background: #f5f5f5;
              }}
            </style>
            <div id="{btn_uid}_wrap">
              <button id="{btn_uid}">⬇ Download PNG</button>
            </div>
            <script>
            (function() {{
              const B64 = "{b64_data}";
              const FNAME = {repr(fname)};
              const btn = document.getElementById("{btn_uid}");
              if (!btn) return;

              function b64ToBlob(b64) {{
                const bin = atob(b64);
                const len = bin.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
                return new Blob([bytes], {{ type: "image/png" }});
              }}

              btn.addEventListener("click", async () => {{
                const blob = b64ToBlob(B64);
                // Prefer File System Access API for a real save dialog.
                const fsa = window.showSaveFilePicker
                  || (window.parent && window.parent.showSaveFilePicker);
                const scope = window.showSaveFilePicker ? window : window.parent;
                if (typeof fsa === "function") {{
                  try {{
                    const handle = await fsa.call(scope, {{
                      suggestedName: FNAME,
                      types: [{{
                        description: "PNG image",
                        accept: {{ "image/png": [".png"] }}
                      }}],
                    }});
                    const writable = await handle.createWritable();
                    await writable.write(blob);
                    await writable.close();
                    return;
                  }} catch (err) {{
                    if (err && err.name === "AbortError") return;
                    console.warn("showSaveFilePicker failed, falling back:", err);
                  }}
                }}
                // Fallback: classic <a download> trigger
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = FNAME;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                setTimeout(() => URL.revokeObjectURL(url), 1000);
              }});
            }})();
            </script>
            """,
            height=56,
        )

st.divider()

# ---------------------------------------------------------------------------
# Generate button
# ---------------------------------------------------------------------------
is_generating = st.session_state.pending_task is not None
ready = bool(
    upload
    and FREEPIK_API_KEY
    and R2 is not None
    and code is not None
    and not is_generating
)
aspect_label = detect_aspect_ratio(*original.size) if original is not None else "—"
if is_generating:
    button_label = "Generating… please wait"
elif code is None:
    button_label = "Generate → (enter a valid Pantone)"
else:
    button_label = f"Generate → {code} {name}  ·  {aspect_label}  ·  4K"
generate = st.button(
    button_label,
    type="primary",
    use_container_width=True,
    disabled=not ready,
)

# ---------------------------------------------------------------------------
# Generate flow — submit only, then hand off to the background poller
# ---------------------------------------------------------------------------
if generate and original is not None and not is_generating:
    extra = st.session_state.extra_instructions or ""
    prompt = build_recolor_prompt(name, hex_code, system_key, code, extra)
    aspect_ratio = detect_aspect_ratio(*original.size)
    jpg_bytes = original_bytes  # native size, no crop/pad/resize

    r2_key = None
    try:
        with st.spinner("Submitting task…"):
            r2_key = R2.upload(jpg_bytes, mime="image/jpeg", suffix="jpg")
            ref_url = R2.presigned_get(r2_key, ttl_seconds=600)
            submit = submit_nano_banana_pro(
                prompt, ref_url, FREEPIK_API_KEY, aspect_ratio
            )

        if submit["status_code"] not in (200, 201):
            st.error(f"Submit failed (HTTP {submit['status_code']})")
            with st.expander("Raw response", expanded=True):
                st.json(submit["body"])
            if r2_key and R2 is not None:
                R2.delete(r2_key)
        else:
            task_id = ((submit["body"] or {}).get("data") or {}).get("task_id")
            if not task_id:
                st.error("No task_id in submit response")
                with st.expander("Raw response", expanded=True):
                    st.json(submit["body"])
                if r2_key and R2 is not None:
                    R2.delete(r2_key)
            else:
                # Store everything we need so the background poller can
                # finish the job independently of any subsequent reruns.
                st.session_state.pending_task = {
                    "task_id": task_id,
                    "r2_key": r2_key,
                    "t0": time.time(),
                    "system": system_key,
                    "code": code,
                    "name": name,
                    "hex_code": hex_code,
                    "original_bytes": jpg_bytes,
                    "original_name": original_name,
                    "extra": extra,
                    "aspect_ratio": aspect_ratio,
                    "prompt": prompt,
                }
                st.rerun()
    except Exception as e:
        st.error(f"Submit error: {e}")
        if r2_key and R2 is not None:
            try:
                R2.delete(r2_key)
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Background poller — runs once per rerun while a task is pending
# ---------------------------------------------------------------------------
if st.session_state.pending_task:
    pt = st.session_state.pending_task
    elapsed = time.time() - pt["t0"]

    try:
        resp = requests.get(
            f"{NANO_BANANA_PRO_ENDPOINT}/{pt['task_id']}",
            headers={"x-freepik-api-key": FREEPIK_API_KEY},
            timeout=15,
        )
        body = _safe_json(resp)
        data = (body or {}).get("data") or {}
        status = (data.get("status") or "").upper()
    except Exception:
        status = "POLLING"
        body = {}

    if status in ("COMPLETED", "SUCCESS", "SUCCEEDED"):
        img_ref = extract_image_url(body)
        if img_ref:
            try:
                img_bytes = fetch_bytes(img_ref)
                st.session_state.history.append({
                    "timestamp": time.time(),
                    "system": pt["system"],
                    "code": pt["code"],
                    "name": pt["name"],
                    "hex": pt["hex_code"],
                    "original_bytes": pt["original_bytes"],
                    "original_name": pt.get("original_name") or "image",
                    "result_bytes": img_bytes,
                    "extra_instructions": pt["extra"],
                    "elapsed_s": elapsed,
                })
                st.session_state.active_idx = len(st.session_state.history) - 1
            except Exception as e:
                st.error(f"Failed to fetch image: {e}")
        else:
            st.warning("Completed but no image URL in response.")
        if pt.get("r2_key") and R2 is not None:
            try:
                R2.delete(pt["r2_key"])
            except Exception:
                pass
        st.session_state.pending_task = None
        st.rerun()
    elif status in ("FAILED", "ERROR"):
        st.error("Generation failed.")
        if pt.get("r2_key") and R2 is not None:
            try:
                R2.delete(pt["r2_key"])
            except Exception:
                pass
        st.session_state.pending_task = None
    elif elapsed > POLL_TIMEOUT_S:
        st.error(f"Generation timed out after {int(elapsed)}s.")
        if pt.get("r2_key") and R2 is not None:
            try:
                R2.delete(pt["r2_key"])
            except Exception:
                pass
        st.session_state.pending_task = None
    else:
        # Still running — wait a bit and rerun to poll again.
        # If the user clicks anything during this sleep, Streamlit will
        # interrupt and start a fresh rerun, which picks up the same
        # pending_task from session_state and keeps polling. The Freepik
        # task itself is completely unaffected by UI interactions.
        time.sleep(POLL_INTERVAL_S)
        st.rerun()
