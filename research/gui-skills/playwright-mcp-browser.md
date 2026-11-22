# Playwright MCP (browser) — GUI skills

- **Version tested:** `@playwright/mcp` 0.0.83, headless Chromium, in `sandboxes/browser`
- **Last verified:** 2026-11-05, browser rerun (`results/raw/br_results.json`)
- **Control path:** Playwright MCP tools `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`

## Launch and focus

- `browser_navigate` with a plain URL: `http://127.0.0.1:8000/`. Never wrap it in backticks or quotes;
  `` `http://...` `` became `https://%60http//...` and loaded Chrome's error page. — verified 2026-11-05

## Layout

- `browser_navigate` in this version returns a **link to a snapshot file**, not the page. Call
  `browser_snapshot` with **no arguments** to read the page. — verified 2026-11-05
- The snapshot is an accessibility tree; every element carries `[ref=...]`, e.g. `textbox "Name" [ref=f3e5]`.

## Tasks

### Click, type, submit — verified 2026-11-05

1. `browser_snapshot` (no arguments).
2. Find the element and copy its ref **bare**: `f3e5`.
3. `browser_click` / `browser_type` with `target: "f3e5"`. Playwright resolves it to the role and name
   (`getByRole('textbox', { name: 'Name' })`).
4. After a click that navigates, `browser_snapshot` again before reading anything from the new page.

## Keyboard shortcuts

`browser_type` with `submit: true` presses Enter after typing.

## Dialogs and interruptions

None seen on the test site.

## Pitfalls

- `#f3e5`, `[ref=f3e5]`, `#name`, `name_field` and `text=Contact` style guesses either fail or match the
  wrong thing; use the bare ref from the latest snapshot. — verified 2026-11-05
- `browser_snapshot` arguments such as `target`, `filename: "x.png"` or `"snapshot.json"` failed with
  "does not match any elements" or "outside allowed roots". Call it with no arguments. — verified 2026-11-05
- The page **title** is not the **heading**: read the `heading` element from a snapshot. Qwen answered
  "Contact" (title) for "Contact us" (heading). — verified 2026-11-05
- Refs change between snapshots (`e5` on one page, `f3e5` on the next); never reuse a ref from an earlier page.
