"""Files by key (spec-117 R7, R14): a local folder when ``R2_*`` is unset, the private R2 bucket otherwise.

A key is job-relative: ``<job_id>/display/<photo>.jpg``, ``<job_id>/faces/<face_id>.jpg``,
``<job_id>/results/embeddings.npy`` ... The two stores place it differently:

    LocalObjectStore(jobs_root)   <jobs_root>/<key>          the job folders, exactly as before spec-117
    R2ObjectStore(bucket)         s3://<bucket>/jobs/<key>   everything of one job under jobs/<id>/

so a key names the same file whichever store holds it, and a job's files are one prefix (``delete_prefix``).
spec-120: the public site's uploads go straight from the browser to ``s3://<bucket>/uploads/<key>`` through signed
URLs (``R2ObjectStore.under("uploads/")``, ``presigned_put``); R2 refuses a body whose size or type differs from the
signed one. Keys are checked (no ``..``, no absolute paths, no backslashes) before either store touches anything.
"""
from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["ObjectStore", "LocalObjectStore", "R2ObjectStore", "make_object_store", "job_key", "R2_ENV"]

R2_ENV = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")


def check_key(key: str, *, prefix: bool = False) -> str:
    """``key`` if it is a safe relative key (a prefix may end in ``/``), else ``ValueError``."""
    body = key[:-1] if prefix and (key or "").endswith("/") else (key or "")
    parts = body.split("/")
    if not body or "\\" in body or ":" in body or "\x00" in body \
            or any(p in ("", ".", "..") or p != p.strip() for p in parts):
        raise ValueError(f"invalid object key {key!r}")
    return key


def job_key(job_id: str, rel: str) -> str:
    """``<job_id>/<rel>``, checked."""
    return check_key(f"{job_id}/{rel}")


# Parallel uploads (display images and face crops): one PUT to R2 takes 0.5-1 s, so a job's ~1,300 files one
# after another took 10-20 min (spec-117 T8.2). The R2 client keeps as many connections open.
UPLOAD_WORKERS = 16

class ObjectStore(ABC):
    """put / get / exists / list / delete_prefix by key. ``get`` of a missing key raises ``KeyError``."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str) -> List[str]: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int: ...

    def sizes(self, prefix: str) -> Dict[str, int]:
        """Size in bytes of every object under ``prefix`` (one listing, no per-object request)."""
        raise NotImplementedError

    def presigned_put(self, key: str, size: int, content_type: str, expires_s: int = 3600) -> str:
        """A URL the browser PUTs exactly ``size`` bytes of ``content_type`` to (spec-120 R5). R2 only."""
        raise NotImplementedError("signed upload URLs need the R2 store")

    def presigned_get(self, key: str, expires_s: int = 3600) -> str:
        """A short-lived URL to read one object (spec-120 R11). R2 only."""
        raise NotImplementedError("signed read URLs need the R2 store")

    def put_file(self, key: str, path: Path) -> None:
        self.put(key, Path(path).read_bytes())

    def get_or_none(self, key: str) -> Optional[bytes]:
        try:
            return self.get(key)
        except (KeyError, ValueError):
            return None

    @property
    def is_local(self) -> bool:
        return False


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / check_key(key)

    def put(self, key: str, data: bytes) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)                 # a reader never sees half a file

    def put_file(self, key: str, path: Path) -> None:
        dest = self._path(key)
        if dest.exists() and dest.resolve() == Path(path).resolve():
            return                            # already where the key says (a local run's own job folder)
        super().put_file(key, path)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str) -> List[str]:
        base = self.root / check_key(prefix, prefix=True)
        if prefix.endswith("/"):
            if not base.is_dir():
                return []
            files = base.rglob("*")
        else:
            files = base.parent.glob(base.name + "*") if base.parent.is_dir() else []
        out = []
        for p in files:
            if p.is_file() and not p.name.endswith(".part"):
                out.append(p.relative_to(self.root).as_posix())
        return sorted(out)

    def sizes(self, prefix: str) -> Dict[str, int]:
        return {k: (self.root / k).stat().st_size for k in self.list(prefix)}

    def delete_prefix(self, prefix: str) -> int:
        """Delete every file under ``prefix`` (a folder prefix, ending in ``/``); returns how many."""
        if not prefix.endswith("/"):
            raise ValueError("delete_prefix takes a folder prefix ending in '/'")
        keys = self.list(prefix)
        base = self.root / check_key(prefix, prefix=True)
        shutil.rmtree(base, ignore_errors=True)
        return len(keys)

    @property
    def is_local(self) -> bool:
        return True


class R2ObjectStore(ObjectStore):
    """Cloudflare R2 through its S3 API (boto3, imported lazily). Objects live under ``jobs/``."""

    def __init__(self, bucket: str, client, *, base: str = "jobs/"):
        self.bucket, self._s3, self._base = bucket, client, base

    @classmethod
    def from_env(cls) -> "R2ObjectStore":
        import boto3
        from botocore.config import Config

        missing = [v for v in R2_ENV if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"R2 is not configured: {', '.join(missing)} unset")
        client = boto3.client(
            "s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto",
            config=Config(retries={"max_attempts": 5, "mode": "standard"}, max_pool_connections=UPLOAD_WORKERS,
                          signature_version="s3v4"))   # signed URLs carry size + type (spec-120)
        return cls(os.environ["R2_BUCKET"], client)

    def under(self, base: str) -> "R2ObjectStore":
        """The same bucket and connection, objects under another top-level prefix (e.g. ``uploads/``)."""
        return R2ObjectStore(self.bucket, self._s3, base=check_key(base, prefix=True))

    def _name(self, key: str) -> str:
        return self._base + check_key(key)

    def presigned_put(self, key: str, size: int, content_type: str, expires_s: int = 3600) -> str:
        # Content-Length and Content-Type are signed headers: R2 answers 403 to any other size or type
        # (checked on the real bucket, 2026-11-28). Only SigV4 signs them; an older-style URL would take any size.
        url = self._s3.generate_presigned_url(
            "put_object", ExpiresIn=int(expires_s),
            Params={"Bucket": self.bucket, "Key": self._name(key), "ContentType": content_type,
                    "ContentLength": int(size)})
        if "X-Amz-SignedHeaders=content-length%3Bcontent-type" not in url:
            raise RuntimeError("the R2 client does not sign size and type (needs signature_version='s3v4')")
        return url

    def presigned_get(self, key: str, expires_s: int = 3600) -> str:
        return self._s3.generate_presigned_url(
            "get_object", ExpiresIn=int(expires_s), Params={"Bucket": self.bucket, "Key": self._name(key)})

    def sizes(self, prefix: str) -> Dict[str, int]:
        full = self._base + check_key(prefix, prefix=True)
        out: Dict[str, int] = {}
        for page in self._s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=full):
            out.update({o["Key"][len(self._base):]: int(o["Size"]) for o in page.get("Contents", [])})
        return out

    def put(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._name(key), Body=data)

    def get(self, key: str) -> bytes:
        try:
            return self._s3.get_object(Bucket=self.bucket, Key=self._name(key))["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            raise KeyError(key) from None

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._name(key))
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str) -> List[str]:
        full = self._base + check_key(prefix, prefix=True)
        out: List[str] = []
        for page in self._s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=full):
            out += [o["Key"][len(self._base):] for o in page.get("Contents", [])]
        return sorted(out)

    def delete_prefix(self, prefix: str) -> int:
        if not prefix.endswith("/"):
            raise ValueError("delete_prefix takes a folder prefix ending in '/'")
        keys = self.list(prefix)
        for i in range(0, len(keys), 1000):          # S3 deletes at most 1000 per call
            batch = [{"Key": self._base + k} for k in keys[i:i + 1000]]
            self._s3.delete_objects(Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True})
        return len(keys)


def make_object_store(jobs_root: Path | str) -> ObjectStore:
    """R2 when every ``R2_*`` variable is set, else the local job folders (R15)."""
    if all(os.environ.get(v) for v in R2_ENV):
        store = R2ObjectStore.from_env()
        logger.info("object store: R2 bucket %s", store.bucket)
        return store
    return LocalObjectStore(jobs_root)
