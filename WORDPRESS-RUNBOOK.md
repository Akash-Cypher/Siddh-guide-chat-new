# Ask Sid — WordPress Runbook

Everything on the WordPress side: installing the plugin, configuring the API key,
placing the widget, and diagnosing the failures we have actually hit.

Backend, crawler, ECR and App Runner are covered in [README.md](README.md).

---

## How the pieces fit

```
Browser ──> WordPress plugin (proxy) ──> AWS App Runner backend ──> Bedrock + DynamoDB
            /wp-json/siddh/v1/chat        nyrqbf2z3k.ap-south-1.awsapprunner.com
```

The browser never talks to AWS directly and never sees the API key. WordPress
holds the key server-side and forwards each message. This is why the plugin must
be installed for **anything** to work — including the standalone test page, which
also calls `/wp-json/siddh/v1/chat`.

| Piece | Lives in | Controls |
|---|---|---|
| Chat bubble, header, styling | plugin `assets/` | how it looks, session behaviour, instant answers |
| REST proxy `/wp-json/siddh/v1/chat` | plugin PHP | authentication, forwarding |
| Answer content, routing, refusals | AWS backend | *what it says* |

**Consequence:** re-uploading the plugin never changes what the bot answers.
Answer behaviour changes only when the backend is deployed (git push → CI → ECR →
App Runner auto-deploy).

---

## Build the plugin zip

The zip is a build artifact and is gitignored — rebuild it before every handover.

```bash
cd "<repo root>"
python -c "
import zipfile, os
src = 'frontend/siddh-guide-chat'
files = ['siddh-guide-chat.php',
         'assets/siddh-guide-chat-widget.js',
         'assets/style.css',
         'assets/siddhanta-logo.png']
with zipfile.ZipFile('siddh-guide-chat.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(os.path.join(src, f), 'siddh-guide-chat/' + f)
"
python -m pytest tests/test_session_lifecycle.py -q
```

`test_release_zip_matches_the_sources` fails if the zip is stale. **Do not skip
it** — a zip built before a widget change silently ships the old script while
every other test still passes. That has happened.

The zip must contain exactly one top-level folder, `siddh-guide-chat/`. Zipping
the *contents* instead of the folder makes WordPress reject the upload.

`siddh-guide-chat-test.html` is deliberately **not** in the zip; inside the plugin
folder it would be publicly reachable on the live site.

---

## Install or update the plugin

1. **Plugins → Add New Plugin → Upload Plugin** → choose `siddh-guide-chat.zip`
2. **Install Now** → WordPress shows a version comparison → **Replace current with uploaded**
3. Confirm the list shows **Ask Sid**, the expected version, **Active**

If no "Replace" option appears: deactivate and delete the old plugin, then upload
fresh and activate. Deleting the plugin does **not** delete the API key if the key
is in `wp-config.php` — but it *does* if the key was only in the plugin's settings
(see below).

Never edit plugin files through **Plugins → Plugin File Editor**. A syntax error
there white-screens the site, admin included.

### Blast radius

The plugin adds a shortcode, a REST route and an admin screen. It does not touch
WooCommerce, checkout, orders, or the theme. Worst case the chat stops working —
deactivating it restores everything instantly.

---

## Configure the API key

The plugin resolves the key in this order:

1. `SIDDH_CHAT_API_KEY` constant in `wp-config.php` (preferred for production)
2. `siddh_chat_api_key` WordPress option, set via the admin screen

It is **never** hardcoded in the plugin. An older build did hardcode it, which is
why replacing that plugin broke the chat with `401 Unauthorized` — the key went
with the file it was buried in.

### Where to get the value

AWS Console → **App Runner** → `siddh-guide-chat` → **Configuration** →
**Environment variables** → `CHAT_API_KEY` (37 characters).

Copy it straight from AWS into WordPress. Never put it in chat, email, a ticket,
or a commit.

### Where to paste it

`https://<site>/wp-admin/admin.php?page=siddh-guide-kb` — the **Ask Sid KB** menu
item. Scroll to **Settings**:

| Field | Value |
|---|---|
| Backend URL | `https://nyrqbf2z3k.ap-south-1.awsapprunner.com` (no trailing slash) |
| API key | the 37-character `CHAT_API_KEY` |

Click **Save settings**. The banner at the top must change from
*"⚠️ API key not set"* to *"● API key configured"*.

### ⚠️ Browser autofill will corrupt this form

The form has a text field followed by a password field, so password managers fill
it with your **email and login password**. This has happened on this exact screen.

Before saving, confirm:

- Backend URL is a `https://…awsapprunner.com` URL — **not an email address**
- API key was typed/pasted by you — clear the box completely (click in, Ctrl+A,
  Delete) until no dots remain, then paste

Saving the autofilled values points the bot at a nonexistent backend and stores
your personal password as the API key.

Leaving the API key blank keeps the currently stored value — blank does not erase it.

---

## Place the widget

Shortcode: `[siddh_guide_chat]`

| Where | How |
|---|---|
| One page | Pages → Add New → **Shortcode** block → `[siddh_guide_chat]` → Publish |
| Elementor page | **Shortcode** widget → `[siddh_guide_chat]` |
| Every page | WPCode → PHP Snippet, auto-insert *Site Wide Footer* (below) |

```php
add_action('wp_footer', function () {
    echo do_shortcode('[siddh_guide_chat]');
});
```

**Place it once per page only.** A page with both the shortcode and the site-wide
snippet renders two widgets with duplicate element IDs — two buttons, and every
message sent twice.

The block renders nothing where you put it; the widget is `position: fixed` and
floats bottom-right regardless.

---

## Always clear the cache

After **every** plugin update, purge the **full page cache** (WP Rocket /
LiteSpeed / Cloudflare — whatever is active), then check in an incognito window.

The plugin versions its own asset URLs by file modification time, so the browser
fetches fresh CSS/JS on its own. But the **cached page HTML still points at the
old URL** until the page cache is purged. This is the single most common reason
an update "does nothing".

---

## Verify a deployment

```bash
# 1. Backend alive and on the expected build
curl.exe https://nyrqbf2z3k.ap-south-1.awsapprunner.com/health

# 2. Which widget build WordPress is actually serving
curl.exe https://<site>/wp-content/plugins/siddh-guide-chat/assets/siddh-guide-chat-widget.js

# 3. End-to-end through the proxy, exactly as a visitor does (no API key needed)
curl.exe -X POST https://<site>/wp-json/siddh/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"hello\",\"session_id\":\"diag-0001\"}"
```

Step 3 returning `200` with an `answer` means the whole chain works. `401` means
the API key is missing or wrong in WordPress.

In PowerShell use `curl.exe`, not `curl` — bare `curl` is an alias for
`Invoke-WebRequest`, which rejects a URL without a scheme.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| *"Sorry, something went wrong"* on every message | proxy got a non-200 from the backend | Run verify step 3. `401` → API key. |
| Proxy returns `401 Unauthorized` | key missing/wrong in WordPress | Re-enter it on the Ask Sid KB screen (watch autofill) |
| No chat bubble at all | plugin inactive, shortcode missing, or stale cache | Check Plugins → Active; check the shortcode is on the page; purge cache |
| Update "did nothing" | page cache | Purge full page cache, then Ctrl+F5 |
| Bubble works but looks old | old plugin still installed | Plugins list should read **Ask Sid**, not *Siddh Guide Chat* |
| Answers ignore earlier messages | page was refreshed | By design — see Conversation model |
| Response field says `"continuity": "degraded"` | backend could not reach DynamoDB | Check App Runner logs for `CONTINUITY:` |
| Chat unrecoverably broken | — | Plugins → Deactivate **Ask Sid**. Site unaffected. |
| Site white-screens | bad file edit | Hosting file manager → rename `wp-content/plugins/siddh-guide-chat` to `…-off` |

---

## Conversation model (explain this to support staff)

A conversation lasts exactly as long as the loaded page.

- Closing and reopening the chat bubble **keeps** the conversation
- Refreshing the page **starts a new one**
- A second tab is a **separate** conversation
- Nothing is stored in the visitor's browser; no transcript is ever restored

This is deliberate, for privacy: on a shared or public computer the next visitor
must never see the previous person's chat. It is reported as a bug regularly.
**It is not one.** Do not "fix" it by adding `sessionStorage` — `tests/test_session_lifecycle.py`
fails if anyone does.

When the bubble is reopened the window looks empty even though the conversation
is intact. Judge continuity by whether it still *knows* earlier answers, not by
what is on screen.

---

## Known data gaps

Some Siksha listings have blank fields. The bot states they are not listed rather
than inventing values — correct behaviour, but it looks like a failure in a demo.

- **14 of 32 courses have no duration.** Verified present for *Vedic Mathematics
  Foundations* and *Indic Reasoning and Debating*.
- **No refund policy, free-trial or fee-waiver content** exists in the knowledge
  base. Avoid these in demos.

The fix is content, not code: fill the fields on Siksha, then **Ask Sid KB →
Update Chatbot Knowledge Now**.

---

## Knowledge base refresh

The bot answers only from the crawled website and course catalog.

- Automatic ~90 seconds after any page/post/product is published or updated
- Automatic once a day
- Manual: **Ask Sid KB → Update Chatbot Knowledge Now** (needs the API key set)

Allow about a minute, then reload a chat and ask about the new content.

---

## Running the tests locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q
```

### Windows: `ImportError … cygrpc … Application Control policy`

Some managed Windows machines block gRPC's native DLL, so `chromadb` — and
therefore `backend/main.py` — cannot be imported. The frontend tests still run:

```bash
python -m pytest tests/test_session_lifecycle.py -q
```

To run the full suite, stub chromadb (the tests never touch a real vector store):

```python
# run_tests.py
import sys, types
stub = types.ModuleType("chromadb"); config = types.ModuleType("chromadb.config")
config.Settings = type("Settings", (), {"__init__": lambda s, *a, **k: None})
stub.config = config
sys.modules.setdefault("chromadb", stub)
sys.modules.setdefault("chromadb.config", config)
import pytest; sys.exit(pytest.main(sys.argv[1:]))
```

```bash
python run_tests.py tests -q
```

CI installs and imports the real chromadb, so this only affects local runs.
