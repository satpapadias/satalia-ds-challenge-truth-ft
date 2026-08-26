"""GCP Identity Token Auth for httpx client calls."""

import httpx
import logging
import google.auth.transport.requests
import google.oauth2.id_token

logger = logging.getLogger("truthclf.auth")

class GcpAuth(httpx.Auth):
    """httpx-compatible Auth flow for fetching GCP Identity Tokens."""

    def __init__(self, target_audience: str = ""):
        self.target_audience = target_audience

    def auth_flow(self, request: httpx.Request):
        aud = (self.target_audience or f"{request.url.scheme}://{request.url.host}").rstrip("/")
        
        if "127.0.0.1" not in aud and "localhost" not in aud:
            try:
                auth_req = google.auth.transport.requests.Request()
                token = google.oauth2.id_token.fetch_id_token(auth_req, aud)
                request.headers["Authorization"] = f"Bearer {token}"
            except Exception as e:
                logger.error(f"Explicit token fetch failed for {aud}: {e}")

        yield request

    async def async_auth_flow(self, request: httpx.Request):
        # httpx prefers async_auth_flow for AsyncClients. We explicitly delegate 
        # to the sync flow. google.auth caches tokens, so this is safe and reliable.
        for req in self.auth_flow(request):
            yield req
