# Authentication and the Administration API

The server is one local organization with one user: whoever owns the data
directory. It implements the parts of the OpenAI Administration API
(`/v1/organization/*`) that mean something for that, and refuses the rest
with a `400` whose `code` is `unsupported` and whose message gives the reason.

## Authentication

Off by default. On `127.0.0.1` that means any local process can use the
server, and so can any web page a local browser opens that is allowed to
reach the port. Bound to another address, anyone who can reach it can; `serve`
prints a warning then.

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

### Key storage

A key is a prefix, 16 hex digits of lookup, `_`, and 64 hex digits (256 bits)
of OS randomness. The services store (`~/Library/Application
Support/local-ai/services`, owner-only) keeps the lookup, the redacted form
(`sk-admin...xyz`), and an HMAC-SHA-256 of the whole key under a fresh 256-bit
salt; never the key. Verification recomputes the HMAC and compares tags with
`ring::hmac::verify` (constant time). The `--api-key` key is compared the same
way, under a random per-process HMAC key. A 256-bit random key needs no slow
password hash. `last_used_at` is written at most once a minute per key.

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
  finished generation (Chat Completions, Completions, Responses, including a
  client that went away: the generation is drained to its final count) and per
  embeddings request. `vector_stores` reports the current storage level in the
  bucket holding now. Families the server cannot produce (images, web search,
  code interpreter, file search calls) are always empty, as are moderations
  and audio until those routes record usage (`admin::record` in the server).
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

- Not recorded: batch lines, eval runs, compaction and reasoning-summary
  generations, and background Responses resumed after a restart.
- Usage records are kept indefinitely (one small row per request).
- The TCP listener speaks plain HTTP (TLS covers only HTTP/3, when
  configured): on a non-loopback address, keys cross the network in clear
  text.
- A process running as the owner can read the store and turn authentication
  off; authentication protects against other users and network peers, not
  against the owner's own processes with file access.
