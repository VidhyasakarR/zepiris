from __future__ import annotations

import asyncio
import ipaddress
import time
from collections import OrderedDict

import httpx

from zepiris.exceptions import ReferenceImageFetchError


class _ReferenceBytesCache:
    """Byte-budget LRU of fetched reference images, keyed by URL.

    The enrolled selfie is fetched from S3 on *every* verification of that
    person, and it is the same bytes every time: production reference URLs are
    stable, unsigned object URLs whose keys are content-unique (uuid /
    user+timestamp), so a URL cannot quietly start serving different pixels.
    Caching the bytes here removes one of the two S3 round trips per request —
    and with it S3's latency tail, which otherwise lands directly in the
    verification path — while the probe (a fresh capture behind a fresh URL on
    every request) rightly never hits.

    Bounded by **bytes**, not entries, because the values are whole images:
    entry-count budgets look small until 4096 x 300 KB turns out to be 1.2 GB.
    The TTL is a safety valve for the one assumption above — an overwritten key
    — bounding how long a stale body could be served if a deployment breaks the
    content-unique naming.

    No lock: the fetcher is awaited only on the event loop, and neither ``get``
    nor ``put`` yields between check and mutate, so single-threaded execution
    is the synchronization.
    """

    def __init__(self, max_bytes: int, ttl_seconds: float) -> None:
        self._max_bytes = max(0, int(max_bytes))
        self._ttl = float(ttl_seconds)
        self._entries: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
        self._total_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0

    @property
    def enabled(self) -> bool:
        return self._max_bytes > 0

    def get(self, url: str) -> bytes | None:
        if not self._max_bytes:
            return None
        entry = self._entries.get(url)
        if entry is None:
            self._misses += 1
            return None
        data, deadline = entry
        if time.monotonic() >= deadline:
            del self._entries[url]
            self._total_bytes -= len(data)
            self._expired += 1
            self._misses += 1
            return None
        # Refresh recency: eviction takes the references nobody is verifying
        # against any more.
        self._entries.move_to_end(url)
        self._hits += 1
        return data

    def put(self, url: str, data: bytes) -> None:
        if not self._max_bytes or len(data) > self._max_bytes:
            return
        old = self._entries.pop(url, None)
        if old is not None:
            self._total_bytes -= len(old[0])
        self._entries[url] = (data, time.monotonic() + self._ttl)
        self._total_bytes += len(data)
        while self._total_bytes > self._max_bytes:
            _, (evicted, _) = self._entries.popitem(last=False)
            self._total_bytes -= len(evicted)
            self._evictions += 1

    def snapshot(self) -> dict[str, float | int | bool]:
        """Hit rate and occupancy, for /metrics.

        The hit rate is the number worth watching: near zero means this traffic
        is first-time verifications and the memory is buying nothing — near one
        means half the S3 traffic (and its tail) is gone.
        """
        looked_up = self._hits + self._misses
        return {
            "enabled": self._max_bytes > 0,
            "entries": len(self._entries),
            "bytes": self._total_bytes,
            "max_bytes": self._max_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "expired": self._expired,
            "hit_rate": round(self._hits / looked_up, 3) if looked_up else 0.0,
        }


def _blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified)


def make_url_guard(*, allowed_hosts: tuple[str, ...] = (), block_private: bool = True):
    """An httpx ``request`` event hook that refuses internal / unlisted targets.

    Image URLs arrive from rider devices (``face_check_s3``, ``source_selfie_s3``,
    ``image_s3``), so without this the server can be made to GET its own ML
    service, the cloud metadata endpoint (169.254.169.254) or anything else on
    the private network. The hook runs on every hop, so a redirect cannot
    bounce past it.

    ``allowed_hosts``: exact hosts or ``.suffix`` entries (``.amazonaws.com``);
    empty = any public host. ``block_private``: refuse hosts that resolve to
    loopback / private / link-local / reserved addresses.
    """

    async def guard(request: httpx.Request) -> None:
        host = (request.url.host or "").lower().rstrip(".")
        if request.url.scheme not in ("http", "https") or not host:
            raise ReferenceImageFetchError(reason="url_not_allowed", detail_msg="scheme")
        if allowed_hosts and not any(
            host == h or (h.startswith(".") and host.endswith(h)) for h in allowed_hosts
        ):
            raise ReferenceImageFetchError(reason="url_not_allowed", detail_msg=f"host {host}")
        if not block_private:
            return
        try:
            ips = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                infos = await asyncio.get_running_loop().getaddrinfo(host, request.url.port or 443)
            except OSError as exc:
                raise ReferenceImageFetchError(reason="transport_error", detail_msg="dns") from exc
            ips = [ipaddress.ip_address(i[4][0].split("%")[0]) for i in infos]
        if any(_blocked_ip(ip) for ip in ips):
            raise ReferenceImageFetchError(reason="url_not_allowed", detail_msg="private address")

    return guard


class S3ImageFetcher:
    """Fetch a reference image from a presigned/public URL via a guarded HTTP GET.

    No AWS credentials: the URL must be directly retrievable. Guards against
    slow responses (timeout on the client) and oversized payloads (max_bytes).

    Async because both images are fetched on every request: at high concurrency
    a blocking fetch would hold a worker thread per in-flight request purely to
    wait on a socket, and the thread pool — not S3 — would become the limit.

    ``cache_max_bytes`` > 0 enables an in-process bytes cache for URLs fetched
    with ``cacheable=True`` — see :class:`_ReferenceBytesCache` for why only the
    reference side qualifies. 0 (the default) keeps every fetch a real GET.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        max_bytes: int,
        *,
        cache_max_bytes: int = 0,
        cache_ttl_seconds: float = 900.0,
    ) -> None:
        self._client = client
        self._max_bytes = max_bytes
        self._cache = _ReferenceBytesCache(cache_max_bytes, cache_ttl_seconds)

    async def fetch(self, url: str, *, cacheable: bool = False) -> bytes:
        if cacheable:
            cached = self._cache.get(url)
            if cached is not None:
                return cached

        # Streamed, so an oversized (or endless) body is cut off at max_bytes
        # instead of being read into memory first.
        try:
            async with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise ReferenceImageFetchError(
                        reason="bad_status", detail_msg=f"status_{response.status_code}"
                    )
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self._max_bytes:
                        raise ReferenceImageFetchError(
                            reason="too_large", detail_msg=f"over_{self._max_bytes}_bytes"
                        )
                    chunks.append(chunk)
                data = b"".join(chunks)
        except httpx.TimeoutException as exc:
            raise ReferenceImageFetchError(reason="timeout", detail_msg="timeout") from exc
        except httpx.HTTPError as exc:
            raise ReferenceImageFetchError(reason="transport_error", detail_msg=type(exc).__name__) from exc
        if not data:
            raise ReferenceImageFetchError(reason="empty", detail_msg="empty_body")

        if cacheable:
            self._cache.put(url, data)
        return data

    def cache_snapshot(self) -> dict[str, float | int | bool]:
        """Cache saturation for the API's /metrics endpoint."""
        return self._cache.snapshot()

    async def aclose(self) -> None:
        await self._client.aclose()
