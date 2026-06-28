# RALI Searcher — Deployment

The searcher is a **pure local web service** (`uvicorn` on `127.0.0.1:<port>`). How it is exposed to
users is a **separate, swappable layer** — the app never needs a public IP or inbound firewall hole.
Pick one of the three publish options below; none of them change the application code.

## 1. Run the service (local)

The app is built by `apcir.interactive.search_server.create_app(config, eager_load=False, store=...)`:
- `eager_load=False` → start EMPTY; load/evict indexes on demand via `POST /activate` (the RALI
  Searcher mode). `eager_load=True` is the legacy single-index Sim.API server.
- `store=Store("/path/rali_searcher.db")` → a persistent SQLite file (users / tokens / sessions /
  turns / per-user PTKB). Omit for an in-memory store (dev only).

Minimal launcher (e.g. `run_searcher.py`):

```python
import uvicorn
from apcir.interactive.pipeline import PipelineConfig
from apcir.interactive.search_server import create_app
from apcir.interactive.store import Store

store = Store("/data/rech/huiyuche/rali_searcher.db")
# seed an admin once (accounts are admin-created — there is no open signup):
if store.verify_user("admin", "CHANGE_ME") is None:
    try: store.create_user("admin", "CHANGE_ME", is_admin=True)
    except ValueError: pass

app = create_app(PipelineConfig(), eager_load=False, store=store)
uvicorn.run(app, host="127.0.0.1", port=8500, workers=1)   # ONE worker: huge resident state, GPU serialized
```

Run on the host that has the indexes + GPUs (octal40 for the full ClueWeb qwen index — its fp16
**load peak ~320G** exceeds octal31's free RAM, so the capacity guard refuses it on octal31; octal31
can serve the smaller QReCC dev indexes). Then publish with ONE of:

## 2A. SSH local port-forward (lab-internal, simplest, zero public exposure)

Each user tunnels `localhost → the service` through the iro gateway. From a laptop (even at home, as
long as you can SSH to iro):

```bash
ssh -N -L 8500:localhost:8500 -J arcade@iro.umontreal.ca <user>@octal40   # octal31/octal40
# then open http://localhost:8500 in the browser; log in with username/password.
```

- Pros: nothing is exposed publicly; no extra software. Pros for trust: only people who can SSH to
  the lab can reach it.
- Cons: every user must run an SSH command — **not** "just open a URL and type a password".

## 2B. Cloudflare Tunnel (a public URL; "just open a URL + log in", works from home; **recommended for non-SSH users**)

The host dials OUT to Cloudflare (outbound HTTPS only — no public IP / inbound hole needed) and gets a
public `https://…` URL that proxies down the tunnel to the local service.

```bash
# one-time: install cloudflared (a single static binary), then:
cloudflared tunnel --url http://127.0.0.1:8500          # quick: prints a https://<random>.trycloudflare.com URL
# or, for a stable named tunnel + your own domain:
#   cloudflared tunnel login && cloudflared tunnel create rali && \
#   cloudflared tunnel route dns rali searcher.example.com && \
#   cloudflared tunnel run --url http://127.0.0.1:8500 rali
```

- Pros: users (incl. you at home) just open the URL and enter their app username/password — **zero
  SSH**. App auth (admin-created accounts + bearer tokens) is the access gate.
- Requires: octal40/31 can make outbound HTTPS to Cloudflare (university networks usually allow it —
  test with `curl -sS https://cloudflare.com >/dev/null && echo outbound-ok`). Optionally add
  Cloudflare Access in front for a second auth layer.

## 2C. Public-IP reverse proxy (if you control a public host)

If you have a VM/server WITH a public IP, keep a reverse SSH tunnel from the index host to it and let
it reverse-proxy (nginx/caddy) `https://your-host → 127.0.0.1:8500`:

```bash
# on the index host (octal40), persistent reverse tunnel to the public host:
autossh -M 0 -N -R 8500:localhost:8500 user@your-public-host
# on the public host, proxy 443 -> localhost:8500 (nginx/caddy), TLS via Let's Encrypt.
```

## Notes / threat model

- Accounts are **admin-created** (`store.create_user(..., is_admin=True/False)`); there is no open
  signup. Passwords are PBKDF2-HMAC-SHA256; bearer tokens expire after 30 days.
- Login throttling is intentionally **not** implemented — keep the service network-restricted
  (SSH tunnel / Cloudflare Access / a private host). Add throttling if you ever expose it broadly.
- Always run a **single** uvicorn worker (the resident indexes + GPU search are not multi-worker safe).
- Back up the one SQLite `.db` file; that is all user/session/PTKB state.
