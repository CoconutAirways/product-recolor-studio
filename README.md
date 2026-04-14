# Product Recolor Studio

Streamlit app that recolors a product photograph to a specific Pantone code
using Freepik's **Nano Banana Pro** (Gemini 3 Pro Image) endpoint.

Output is hardcoded to **1:1** and **4K** for clean e-commerce assets.
Reference uploads are routed through a private **Cloudflare R2** bucket
(presigned GET URL, 10-minute TTL, auto-deleted after generation).

## Secrets

Set these in `.streamlit/secrets.toml` locally or in the Streamlit Community
Cloud dashboard under **Settings → Secrets**:

```toml
FREEPIK_API_KEY      = "fpk_..."
R2_ACCOUNT_ID        = "..."
R2_ACCESS_KEY_ID     = "..."
R2_SECRET_ACCESS_KEY = "..."
R2_BUCKET            = "product-recolor-uploads"
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Streamlit Community Cloud → New app → point at this repo → `app.py` → add
the secrets above → deploy. No build step needed.
