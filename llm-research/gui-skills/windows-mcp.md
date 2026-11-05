# Windows-MCP (0.8.8) — GUI skills

- **Version tested:** Windows-MCP 0.8.8 (server reports 4.1.6) on Windows 11 Enterprise Evaluation, `win11-gui`
- **Last verified:** 2026-11-05, GUI baseline run (`results/raw/gui_results.json`)
- **Control path:** Windows-MCP over streamable HTTP, `Authorization: Bearer <key>`

## Rules that decide whether a task works

1. **`Type` must say where to type.** It needs `loc` (`[x, y]` screen coordinates) or `label` (an element number
   from the last `Snapshot`), plus `text`. Without either it fails with "Either loc or label must be provided",
   and nothing is typed. Every Notepad and Explorer failure in the baseline was this. — verified 2026-11-05
2. **Call `Snapshot` with no arguments** to see the screen: it lists interactive elements with labels and
   coordinates. `Snapshot` with `region: "Desktop"` failed with "Error capturing desktop state". — verified 2026-11-05
3. **Read every tool result.** A result starting "Error calling tool" means nothing happened; fix the call and
   retry. Never report success after an error. Two models said "done" after only errors. — verified 2026-11-05
4. `App` with `{"mode": "launch", "name": "Notepad"}` (or "Calculator", "File Explorer") opens an app. — verified 2026-11-05
5. `Shortcut` with `{"shortcut": "ctrl+s"}` presses a key combination. — verified 2026-11-05
6. Windows-MCP pauses itself with `USER_CONTROL` while a person uses the desktop (the noVNC viewer counts).
   Close the viewer; `ControlStatus` should read `ready`. — verified 2026-11-05

## Tasks

### Type into Notepad and save — unverified as a whole
1. `App` launch Notepad, then `Snapshot`.
2. Find the text area in the snapshot; `Type` with its `label` (or `loc`) and the text.
3. `Shortcut` ctrl+s, then `Snapshot` to find the file-name box in the Save dialog.
4. `Type` the full path into that box with its `label` and `press_enter: true`.
5. `Snapshot` again: the window title should show the file name.

### Calculator — verified 2026-11-05 (Qwen2.5-VL)
1. `App` launch Calculator, `Snapshot`, then `Click` the digit and operator buttons by label, then `=`.
2. Read the display from a `Snapshot`; report only the number it shows.

### New folder in Explorer — unverified
1. `App` launch File Explorer; `Snapshot`; `Click` "Desktop" in the navigation pane.
2. `Shortcut` ctrl+shift+n creates "New folder" with its name selected.
3. `Snapshot`, then `Type` the name into the selected name box by `label`, `press_enter: true`.

## Pitfalls
- Big-Tiger-Gemma clicked Calculator buttons by coordinates and reported 5490 for 123 × 45 (true 5535).
- Tiger-Gemma 9B wrote pyautogui code or invented tool names ("App Open") instead of calling tools.
