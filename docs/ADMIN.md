# Authentication and the Administration API

The server is one local organization with one user: whoever owns the data
directory. It implements the parts of the OpenAI Administration API
(`/v1/organization/*`) that mean something for that, and refuses the rest
with a `400` whose `code` is `unsupported` and whose message gives the reason.

## Authentication

Off by default. On `127.0.0.1` that means any local process can use the
server; web pages a local browser opens cannot (see [Browsers](#browsers)).
Bound to another address, anyone who can reach it can; `serve` prints a
warning then.

```bash
local-ai admin create-key --name ops   # prints sk-admin-... once on stdout
local-ai admin status                  # authentication: on
```

The first admin key turns authentication on, for a running server too (the
state is read from the store on every request). From then on every request,
`/health` included, needs `Authorization: Bearer <key>`:

| key | authorizes |
|---|---|
| admin key `sk-admin-...` | only `/v1/organization/*` |
| project key `sk-proj-...` | every other route; usage is attributed to its project and key |
| `serve --api-key KEY` (optional) | every route except `/v1/organization/*`; usage is unattributed |

A missing or invalid key is a `401` (`invalid_api_key`); the wrong kind of key
is a `403` (`insufficient_permissions`). Expired keys, keys of archived
projects and keys of deleted service accounts are refused. Deleting every
admin key leaves authentication on; `local-ai admin create-key` makes a new
one, and `local-ai admin disable-auth` turns authentication off (keys are
kept; `enable-auth` turns it back on). While authentication is off the
Administration API is refused, because anyone local could otherwise mint keys.

Project keys are created through service accounts, as in the OpenAI API:

```bash
ADMIN=sk-admin-...
curl -s localhost:8080/v1/organization/projects -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' -d '{"name":"apps"}'
curl -s localhost:8080/v1/organization/projects/$PROJECT/service_accounts \
  -H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json' \
  -d '{"name":"my-app"}'        # .api_key.value is the sk-proj-... key, shown once
```

### Browsers

A web page can reach a local server in two ways: by sending requests whose
answers it cannot read (a `text/plain` POST needs no CORS preflight, which is
enough to start a generation or delete a file), and by DNS rebinding, where
`evil.example` re-points its name at `127.0.0.1` and then reads the server as
its own origin (Ollama's CVE-2024-28224). Every request, on every HTTP
version, is checked before authentication:

- **Host.** While bound to a loopback address, a request is refused (`403`)
  unless its `Host` (HTTP/2 and HTTP/3 `:authority`) is `localhost`, a
  `*.localhost` name, a `*.internal` name (Docker's `host.docker.internal`;
  the TLD is reserved for private use, so public DNS cannot rebind it) or an
  IP address. A rebinding page always sends its own name. Bound to another
  address the server can be reached under names it cannot know, and the host
  is not checked.
- **Origin.** While authentication is off, a request whose `Origin` is
  another site, or that has no `Origin` but `Sec-Fetch-Site: cross-site`, is
  refused (`403`). Admitted without configuration: `http(s)` pages on
  `localhost`, `*.localhost`, `127.0.0.0/8` or `[::1]` (any port, like
  Ollama's default `OLLAMA_ORIGINS`), and origins that are not `http(s)`
  (`tauri://`, `vscode-webview://`, `chrome-extension://`, ...), which only
  local applications and extensions have. `null` (sandboxed frames, `file:`
  pages) can come from any site and is refused. `serve --allow-origin
  ORIGIN` admits an exact origin (repeatable; `*` admits every origin, `null`
  admits `file:` pages). Once authentication is on, an authorized request is
  admitted from any origin: the page was given a key.

CLIs, SDKs and agent harnesses send no `Origin` or `Sec-Fetch-*` headers and
address the server as `localhost` or an IP, so neither check affects them.
An admitted request with an `Origin` gets `Access-Control-Allow-Origin`
reflecting it, and an admitted CORS preflight is answered `204` with the
methods and headers it asked for (with authentication on, every origin's
preflight is admitted, since the request itself still needs a key), so
browser applications can read the answers.

Other local servers differ: Ollama checks the host the same way and admits
only its `OLLAMA_ORIGINS` (localhost and app schemes by default); llama.cpp's
server reflects every origin unless `--cors-origins` is set, and limits it to
localhost once its file and shell tools are on; LM Studio sends no CORS
headers unless "Enable CORS" is switched on. Neither check here can be turned
off; `--allow-origin '*'` comes closest to llama.cpp's default, and the host
check still applies.

### Key storage

A key is a prefix, 16 hex digits of lookup, `_`, and 64 hex digits (256 bits)
of OS randomness. The services store (`~/Library/Application
Support/local-ai/services`, owner-only) keeps the lookup, the redacted form
(`sk-admin...xyz`), and an HMAC-SHA-256 of the whole key under a fresh 256-bit
salt; never the key. Verification recomputes the HMAC and compares tags with
`ring::hmac::verify` (constant time). The `--api-key` key is compared the same
way, under a random per-process HMAC key. A 256-bit random key needs no slow
password hash. `last_used_at` is written at most once a minute per key, as a
best-effort write: if another writer holds the database for more than 50 ms,
or the write fails, it is skipped and the request is still authenticated.
Checks share no lock; each reads through a pooled database connection.

## Endpoints

Implemented:

- `admin_api_keys`: create (the only response with `value`), list (`after`,
  `limit`, `order`), retrieve, delete.
- `projects`: create, list (`include_archived`), retrieve, modify (`name`,
  `external_key_id`), archive. `geography` and `residency` are refused: every
  byte stays on this machine.
- `projects/{id}/service_accounts`: create (with a first key unless
  `create_service_account_only`), list, retrieve, update (`name`, `role`),
  delete (its keys go with it); `.../{id}/api_keys` creates another key.
  Roles are stored and echoed but grant nothing different: every project key
  may use every non-administrative API. Key `scopes` are refused for the same
  reason.
- `projects/{id}/api_keys`: list (`owner_project_access`), retrieve, delete.
- `users`: the one local owner (no email); modify, delete and roles are
  refused. `invites`: always an empty list; sending one is refused.
- `audit_logs`: `api_key.created`/`deleted`, `project.created`/`updated`/
  `archived`, `service_account.created`/`updated`/`deleted`, newest first,
  with the `effective_at[gt|gte|lt|lte]`, `project_ids[]`, `event_types[]`,
  `actor_ids[]`, `resource_ids[]`, `after` and `before` filters.
  `actor_emails[]` matches nothing (the owner has no email).
- `usage/*`: every family, `1m`/`1h`/`1d` buckets aligned to UTC from
  `start_time` to `end_time` (default: the end of the bucket holding now),
  every bucket listed (empty ones with no results), `limit`, `page`, the
  documented filters and `group_by`. The server writes one record per
  finished request, attributed to the calling project and key:
  - `completions`: every finished generation (Chat Completions, Responses,
    including a client that went away: the generation is drained to its final
    count; each compaction and reasoning-summary generation as a record of its
    own), input, cached and output tokens;
  - `completions` with `batch: true`: every batch line, attributed to the
    project key that created the batch (it is stored with it, since a batch
    outlives the request that created it);
  - `embeddings`: input tokens of each `POST /v1/embeddings`;
  - `moderations`: input tokens (every scored window) of each
    `POST /v1/moderations`;
  - `audio_speeches`: input characters of each `POST /v1/audio/speech`,
    recorded once the whole input is synthesized;
  - `audio_transcriptions`: seconds of decoded audio (a started second counts
    as one) of each `POST /v1/audio/transcriptions` and
    `/v1/audio/translations`, as the hosted API counts both.

  `vector_stores` reports the current storage level in the bucket holding
  now. Families the server cannot produce (images, web search, code
  interpreter, file search calls) are always empty.
  Group fields the server does not record (`service_tier`, `size`, `source`,
  `context_level`, `vector_store_id`) are `null`; `api_source` is
  `unlabeled`.
- `costs`: daily buckets with no results. Nothing is billed locally, so there
  are no cost line items; zero-amount items would suggest a price exists.

Refused (`400 unsupported`): project rate limits (none are enforced; the
engine queue's `503` with `Retry-After` is the only bound), certificates
(no mutual TLS), project users and groups, groups, roles, spend alerts and
limits, data retention, model and hosted-tool permissions, external storage.

## Limits

- Work that runs after its request ended is attributed to the project key
  that created it, which is stored with it: batch lines in the services
  database, and a pending durable background Response in its record beside
  its request. Work created with the `--api-key` key or with authentication
  off is recorded unattributed.
- A failed or cancelled speech, transcription, embeddings or moderations
  request records nothing; a generation records what it produced.
- Usage records are kept indefinitely (one small row per request).
- The TCP listener speaks plain HTTP (TLS covers only HTTP/3, when
  configured): on a non-loopback address, keys cross the network in clear
  text.
- A process running as the owner can read the store and turn authentication
  off; authentication protects against other users and network peers, not
  against the owner's own processes with file access.
