import os


BUCKET_ENV_VAR = "EDGESAT_GCS_BUCKET"


def require_gcs_bucket():
    bucket = os.getenv(BUCKET_ENV_VAR, "").strip()
    if not bucket:
        raise RuntimeError(
            f"Set {BUCKET_ENV_VAR} to your Google Cloud Storage bucket name or pass an explicit gs:// path."
        )
    return bucket.removeprefix("gs://").strip("/")


def bucket_path(*parts):
    bucket = require_gcs_bucket()
    clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
    return "gs://" + "/".join([bucket] + clean_parts)
