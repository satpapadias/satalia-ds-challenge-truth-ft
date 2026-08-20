"""GCP Identity Token Auth for httpx client calls."""

import httpx
import logging

logger = logging.getLogger("truthclf.auth")

class GcpAuth(httpx.Auth):
    """httpx-compatible Auth flow for fetching GCP Identity Tokens."""

    def __init__(self, target_audience: str = ""):
        self.target_audience = target_audience

    def auth_flow(self, request: httpx.Request):
        aud = (self.target_audience or f"{request.url.scheme}://{request.url.netloc}").rstrip("/")
        
        if "127.0.0.1" not in aud and "localhost" not in aud:
            token = None
            try:
                meta_url = f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={aud}"
                res = httpx.get(meta_url, headers={"Metadata-Flavor": "Google"}, timeout=5.0)
                if res.status_code == 200:
                    token = res.text.strip()
                else:
                    logger.error(f"Metadata identity fetch returned {res.status_code} for aud={aud}")
            except Exception as meta_err:
                logger.error(f"Metadata identity fetch failed for aud={aud}: {meta_err}")

            if token:
                request.headers["Authorization"] = f"Bearer {token}"
            else:
                logger.error(f"Failed to acquire OIDC token for target audience: {aud}")

        yield request
