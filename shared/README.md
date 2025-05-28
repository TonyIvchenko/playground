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

Apps can still define their own local aliases, but those aliases should map back
to the shared roles here instead of inventing a brand new base token set each
time.
