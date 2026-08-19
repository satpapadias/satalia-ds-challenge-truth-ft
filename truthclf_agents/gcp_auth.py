"""Per-hop OIDC ID token minting for GCP service-to-service calls.

For a Cloud Run URL, it obtains a Google-signed OIDC ID token for the target
service's base URL (the audience) and attaches it as a bearer token. For any
other URL (e.g. localhost or a compose service name), it does nothing, so local
runs continue to work without credentials.
"""

from __future__ import annotations

import logging
import os

import httpx
from google.auth.exceptions import DefaultCredentialsError

logger = logging.getLogger(__name__)

# Set by the Cloud Run environment automatically.
_GCP_SA = os.environ.get("GOOGLE_CLOUD_SERVICE_ACCOUNT", "")


def is_cloud_run_url(url: str) -> bool:
    """True if the URL is a GCP Cloud Run service address."""
    return url.endswith(".run.app")


class GcpAuth(httpx.Auth):
    """An httpx auth-flow class that mints a per-hop OIDC ID token for GCP calls."""

    def __init__(self):
        self._creds = None
        self._project = None
        if _GCP_SA:
            try:
                # Lazy import: only needed in the GCP environment.
                from google.auth import default, transport
                self._creds, self._project = default()
                self._request = transport.requests.Request()
            except (ImportError, DefaultCredentialsError) as e:
                logger.warning("gcp auth unavailable", error=type(e).__name__,
                               detail=str(e), service_account=_GCP_SA)

    def auth_flow(self, request: httpx.Request):
        if self._creds and is_cloud_run_url(str(request.url)):
            from google.oauth2 import id_token
            audience = f"{request.url.scheme}://{request.url.host}"
            token = id_token.fetch_id_token(self._request, audience)
            request.headers["Authorization"] = f"Bearer {token}"
        yield request