"""
Reproduce the proxy failure modes that make glm-5.1 and gpt-5.4 impractical
for the BioGuider pipeline, WITHOUT running the full ~17-min pipeline.

Two distinct failures are demonstrated:

* gpt-5.4  -> HTTP 429 RateLimitError (tiny TPM quota on the proxy). Fire a
             burst of concurrent requests and watch them throttle.
* glm-5.1  -> APITimeoutError / APIConnectionError / RemoteProtocolError
             ("server disconnected"). Send a large-output request under a
             tight timeout and/or a few sequential calls; the proxy drops the
             connection or blows the deadline.

Usage:
    conda run -n bioguider python benchmark/repro_slow_models.py gpt-5.4
    conda run -n bioguider python benchmark/repro_slow_models.py glm-5.1
    conda run -n bioguider python benchmark/repro_slow_models.py both

Env knobs:
    REPRO_BURST     concurrent requests for the 429 test   (default 8)
    REPRO_TIMEOUT   per-call timeout seconds (both tests)   (default 300)
    REPRO_BIGTOKENS max_tokens for the glm large-output test (default 4000)
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env from the repo root so the proxy key/base_url are available when run
# standalone (the pytest path loads these via conftest; this script does not).
load_dotenv()

from benchmark.shared import MODELS, resolve_proxy_credentials


def build_client(model_name: str, *, timeout: int, max_tokens: int | None = None) -> ChatOpenAI:
    """Construct a proxy ChatOpenAI for `model_name` with NO client retries,
    so the raw failure surfaces immediately instead of being hidden."""
    cfg = MODELS.get(model_name, {"type": "litellm", "model": model_name})
    model_id = cfg.get("model", model_name)
    key, base_url = resolve_proxy_credentials()
    kwargs = dict(
        model=model_id,
        api_key=key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,  # surface the first error, don't silently retry
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def _one_call(client: ChatOpenAI, prompt: str) -> tuple[bool, str, float]:
    t0 = time.time()
    try:
        resp = client.invoke(prompt)
        dt = time.time() - t0
        n = len(resp.content or "")
        return True, f"ok ({n} chars)", dt
    except Exception as e:  # noqa: BLE001
        dt = time.time() - t0
        return False, f"{type(e).__name__}: {str(e)[:140]}", dt


def repro_gpt54() -> None:
    """Burst of LARGE-token concurrent requests -> HTTP 429 throttling.

    gpt-5.4's proxy limit is tokens-per-minute, not requests-per-minute, so
    small calls slip through. The pipeline trips it because every call is big.
    We emulate that: a burst of large-output requests that exhausts the TPM
    quota within a minute, exactly like the pipeline does.
    """
    burst = int(os.environ.get("REPRO_BURST", "8"))
    big = int(os.environ.get("REPRO_BIGTOKENS", "8000"))
    timeout = int(os.environ.get("REPRO_TIMEOUT", "300"))
    print(f"\n=== gpt-5.4: {burst} concurrent LARGE requests "
          f"(max_tokens={big}, timeout={timeout}s) to exhaust the TPM quota "
          "(expect 429s) ===")
    client = build_client("gpt-5.4", timeout=timeout, max_tokens=big)
    prompt = (
        "Write an extremely long, exhaustive technical manual (use the entire "
        "token budget) covering bacteriophage genomics end to end: biology, "
        "sequencing, assembly, annotation with pharokka, every CLI flag, "
        "plotting, benchmarking, and troubleshooting. Be maximally verbose."
    )
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=burst) as pool:
        futs = [pool.submit(_one_call, client, prompt) for _ in range(burst)]
        for i, fut in enumerate(as_completed(futs), 1):
            success, msg, dt = fut.result()
            ok += success
            fail += not success
            print(f"  req {i:>2}: {'OK ' if success else 'ERR'}  {dt:6.1f}s  {msg}")
    print(f"  -> {ok} ok, {fail} failed. Under the TPM cap the proxy serialises "
          "large calls, so the rest either 429 (RateLimitError) or block until "
          "they hit the client deadline (APITimeoutError) — the pipeline saw both.")


def repro_glm51() -> None:
    """Large-output request under a tight timeout -> timeout / disconnect."""
    timeout = int(os.environ.get("REPRO_TIMEOUT", "300"))
    big = int(os.environ.get("REPRO_BIGTOKENS", "4000"))
    n_seq = 4
    print(f"\n=== glm-5.1: {n_seq} sequential large-output calls, "
          f"timeout={timeout}s, max_tokens={big} (expect timeouts/disconnects) ===")
    client = build_client("glm-5.1", timeout=timeout, max_tokens=big)
    prompt = (
        "Write an exhaustive, very long technical tutorial (use the full token "
        "budget) on assembling and annotating bacteriophage genomes with "
        "pharokka: installation, every CLI flag, plotting, and troubleshooting."
    )
    ok = fail = 0
    for i in range(1, n_seq + 1):
        success, msg, dt = _one_call(client, prompt)
        ok += success
        fail += not success
        print(f"  call {i}: {'OK ' if success else 'ERR'}  {dt:6.1f}s  {msg}")
    print(f"  -> {ok} ok, {fail} failed "
          f"(APITimeoutError / APIConnectionError / RemoteProtocolError = the slow-model wall)")


def main() -> None:
    target = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    if target in ("gpt-5.4", "gpt5.4", "both"):
        repro_gpt54()
    if target in ("glm-5.1", "glm5.1", "both"):
        repro_glm51()
    print("\nNote: these are proxy-side limits (tiny gpt-5.4 TPM quota; glm-5.1 "
          "latency/instability), not bugs in BioGuider. The full pipeline hits "
          "the same walls but takes ~17 min/run to get there.")


if __name__ == "__main__":
    main()
