# Shared Browser Styles

`shared/browser-tokens.css` is the repo-level source of truth for the base design
tokens used by the browser apps.

`shared/browser-starter.css` is the companion starter stylesheet for the repeated
page shell and surface scaffolding shared by the browser-only apps.

Load both shared stylesheets from browser apps with:

```html
<link rel="stylesheet" href="/shared/browser-tokens.css">
<link rel="stylesheet" href="/shared/browser-starter.css">
```

The starter stylesheet defines the small shared shell and surface helpers for:

- `app-starter-shell`
- `app-starter-surface`
- `app-starter-hero`
- `app-starter-panel`

Use those classes for the top-level page shell, hero block, and large panel
surfaces before introducing one-off page-level `main`, `.hero`, or `.panel`
scaffolding in each app.

The shared stylesheet defines the base token groups for:

- font roles
- spacing scale
- type scale
- radii
- color roles
- elevation shadows

It also owns the shared browser font-loading and default type contract:

- the Google Fonts import for the shared body, display, accent, and mono families lives here
- browser apps should let `body` inherit the shared `--font-body` default instead of redefining page-level font stacks
- browser apps should use the shared typography utilities for exceptions:
  - `app-font-body`
  - `app-font-display`
  - `app-font-accent`
  - `app-font-mono`
- browser apps should not keep local `@import` font blocks in each `index.html`

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

For browser-app status copy, keep the verbs and rhythm consistent too:

- use `Loading …` or `Downloading …` while work is in progress
- use `<thing> ready.` when a model, image, mix, or other artifact is ready
- use the shared `Fallback mode active` badge for degraded-but-usable mode instead of inventing new fallback labels in the status line
- use `Couldn't …` for failures instead of drifting back to `Failed to …`
- prefer browser-model phrasing like `Browser text model ready.` over vague status text like `Model ready.`

For browser-app fallback badges, use one shared degraded-mode pattern too:

- use `app-pill` plus `is-fallback` for fallback-mode badges
- prefer the exact label `Fallback mode active`
- keep the badge separate from the error message so the badge communicates mode and the status line communicates cause

For browser-app buttons, keep the keyboard interaction contract shared too:

- give clickable buttons the shared `app-button` class
- keep `app-button:focus-visible` available so keyboard users get the same focus ring across apps
- layer local `secondary`, `ghost`, or `tab` styles on top of `app-button` instead of replacing the base class
- use real `<button type="button">` controls for in-page actions instead of clickable generic elements

For browser-app mobile layout, keep the spacing and collapse behavior shared too:

- use `app-shell` on the top-level app container so mobile padding and gap collapse together
- use `app-panel` on major hero and panel surfaces so panel padding tightens consistently on small screens
- use `app-stack-md` for major grids that should collapse to one column at the shared medium breakpoint
- use `app-two-up-md` for dense visual grids that should settle into two columns at the shared medium breakpoint
- prefer the shared `64rem` and `48rem` breakpoints over one-off pixel breakpoints in each app

For motion-heavy browser apps, honor reduced-motion preferences too:

- use `window.matchMedia("(prefers-reduced-motion: reduce)")` in animation or camera loops
- simplify non-essential motion when reduced motion is active instead of only slowing it down a little
- prefer static redraws, direct jumps, or lower-frequency updates for decorative visuals
- keep the user-facing status copy honest when reduced motion changes how an effect behaves

For browser-app footers, keep the caveat and privacy contract shared too:

- use one shared footer shell with `app-footer`, `app-footer-grid`, `app-footer-block`, `app-footer-kicker`, and `app-footer-copy`
- keep the three footer labels consistent: `Caveats`, `Privacy`, and `Local only`
- keep the copy short and app-specific instead of repeating generic boilerplate
- use the footer for disclaimers and privacy expectations, not for live status updates

For browser-app help affordances, keep the disclosure pattern shared too:

- use one shared drawer shell with `app-help-drawer`, `app-help-toggle`, `app-help-grid`, `app-help-block`, `app-help-kicker`, and `app-help-list`
- prefer the exact summary label `Help & Shortcuts`
- keep the three drawer labels consistent: `How to use`, `Best results`, and `Shortcuts`
- use `app-keycap` and `app-key-row` for real keyboard shortcuts when an app has them
- if an app has no special keyboard behavior yet, say that plainly instead of inventing fake shortcuts

Apps can still define their own local aliases, but those aliases should map back
to the shared roles here instead of inventing a brand new base token set each
time.
