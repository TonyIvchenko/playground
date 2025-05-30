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

Apps can still define their own local aliases, but those aliases should map back
to the shared roles here instead of inventing a brand new base token set each
time.
