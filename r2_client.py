"""
Cloudflare R2 client for private, short-lived image uploads.

R2 is S3-compatible, so we use boto3. Uploaded objects get a random key,
a presigned GET URL (default 10 min TTL) is returned for the Freepik API
to fetch, and the object is deleted after generation.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import boto3
from botocore.client import Config


@dataclass
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


class R2Client:
    def __init__(self, cfg: R2Config):
        self.cfg = cfg
        self._s3 = boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            aws_access_key_id=cfg.access_key_id,
            aws_secret_access_key=cfg.secret_access_key,
            config=Config(
                signature_version="s3v4",
                region_name="auto",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def upload(self, data: bytes, mime: str, suffix: str = "jpg") -> str:
        """
        Upload bytes to R2 under a random UUID key.
        Returns the object key.
        """
        key = f"uploads/{uuid.uuid4().hex}.{suffix}"
        self._s3.put_object(
            Bucket=self.cfg.bucket,
            Key=key,
            Body=data,
            ContentType=mime,
            # 1-hour automatic expiry hint (bucket lifecycle rule should also delete)
            Metadata={"purpose": "nano-banana-pro-reference"},
        )
        return key

    def presigned_get(self, key: str, ttl_seconds: int = 600) -> str:
        """Return a short-lived HTTPS URL that Freepik can GET."""
        return self._s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.cfg.bucket, "Key": key},
            ExpiresIn=ttl_seconds,
        )

    def delete(self, key: str) -> None:
        """Remove the reference image after generation is done."""
        try:
            self._s3.delete_object(Bucket=self.cfg.bucket, Key=key)
        except Exception:
            # swallow — cleanup is best-effort, don't break the UI
            pass


def from_secrets(secrets_like) -> R2Client | None:
    """
    Build an R2Client from a dict-like object (st.secrets, os.environ, dict).
    Returns None if any required field is missing.
    """
    def get(k: str) -> str:
        v = secrets_like.get(k, "") if hasattr(secrets_like, "get") else ""
        return str(v or "").strip()

    cfg = R2Config(
        account_id=get("R2_ACCOUNT_ID"),
        access_key_id=get("R2_ACCESS_KEY_ID"),
        secret_access_key=get("R2_SECRET_ACCESS_KEY"),
        bucket=get("R2_BUCKET"),
    )
    if not all([cfg.account_id, cfg.access_key_id, cfg.secret_access_key, cfg.bucket]):
        return None
    return R2Client(cfg)
