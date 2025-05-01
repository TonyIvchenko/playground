# Shared Browser Tokens

`shared/browser-tokens.css` is the repo-level source of truth for the base design
tokens used by the browser apps.

Load it from browser apps with:

```html
<link rel="stylesheet" href="/shared/browser-tokens.css">
```

The shared stylesheet defines the base token groups for:

- font roles
- spacing scale
- type scale
- radii
- color roles
- elevation shadows

It also includes the shared top-of-page browser header classes:

- `app-header`
- `app-header-top`
- `app-header-copy`
- `app-kicker`
- `app-title`
- `app-subtitle`
- `app-header-badge`

For empty states, browser apps should keep the copy pattern consistent too:

- use `Waiting for <thing>` for compact pills and badges
- use `<action> to begin.` for the companion instruction line
- avoid `No ... yet` or `No ... loaded` phrasing when the app is simply idle

For browser-app error states, keep both the styling and wording consistent:

- use `app-status` or `app-help` plus `is-error` for status and helper text
- use `app-pill` plus `is-error` for compact pill-style error badges
- prefer `Couldn't <action>.` wording over `Failed to ...` or `... failed`

For browser-model loading states, keep the download pattern consistent too:

- use `app-status` or `app-help` plus `is-loading` while browser models are loading
- use `app-pill` plus `is-loading` for compact model-loading badges
- prefer `Loading browser <thing> model…` for active initialization
- prefer `Downloading browser <thing> model…` when a first-run asset download is in progress

For browser-app fallback badges, use one shared degraded-mode pattern too:

- use `app-pill` plus `is-fallback` for fallback-mode badges
- prefer the exact label `Fallback mode active`
- keep the badge separate from the error message so the badge communicates mode and the status line communicates cause

Apps can still define their own local aliases, but those aliases should map back
to the shared roles here instead of inventing a brand new base token set each
time.
