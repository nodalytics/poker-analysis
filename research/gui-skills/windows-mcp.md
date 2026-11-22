# Windows-MCP (0.8.8) — GUI skills

- **Version tested:** Windows-MCP 0.8.8 (server reports 4.1.6) on Windows 11 Enterprise Evaluation, `win11-gui`
- **Last verified:** 2026-11-05: GUI runs (`results/raw/gui_results*.json`) and the task sequences below, stepped
  through by hand (`scripts/gui_probe.py`)
- **Control path:** Windows-MCP over streamable HTTP, `Authorization: Bearer <key>`

## Rules that decide whether a task works

1. **`Type` must say where to type.** It needs `loc` (`[x, y]` screen coordinates) or `label`, plus `text`. Without
   either it fails with "Either loc or label must be provided", and nothing is typed. Every Notepad and Explorer
   failure in the baseline was this. — verified 2026-11-05
   **A label is always the integer element number from the last Snapshot, never a name:** `label: 12`, not
   `label: "multiply"`, and not `label: 123` for the digits 1-2-3 (those failed with "Input should be a valid
   integer" and "Label 123 out of range"). Find each element's number in the Snapshot and use one per call; the
   same applies to `Click`. — verified 2026-11-05
2. **Call `Snapshot` with no arguments** to see the screen: it lists interactive elements with labels and
   coordinates. `Snapshot` with `region: "Desktop"` failed with "Error capturing desktop state". — verified 2026-11-05
3. **Read every tool result.** A result starting "Error calling tool" means nothing happened; fix the call and
   retry. Never report success after an error. Two models said "done" after only errors. — verified 2026-11-05
4. `App` with `{"mode": "launch", "name": "Notepad"}` (or "Calculator", "File Explorer") opens an app. — verified 2026-11-05
5. `Shortcut` with `{"shortcut": "ctrl+s"}` presses a key combination. — verified 2026-11-05
6. Windows-MCP pauses itself with `USER_CONTROL` while a person uses the desktop (the noVNC viewer counts).
   Close the viewer; `ControlStatus` should read `ready`. — verified 2026-11-05

## Tasks

### Type into Notepad and save — verified 2026-11-05 (by hand through Windows-MCP)
1. `App` launch Notepad, then `Snapshot`. Find `document "Text editor"` and its coordinates.
2. `Type` with `loc` set to those coordinates and the text.
3. `Shortcut` ctrl+s, then `Snapshot`. The Save dialog's `edit "File name:"` is **already focused and pre-filled**
   (with the first line of the text, e.g. "hello from the model.txt").
4. `Type` the full path with `loc` set to that edit box, **`clear: true`** (or the path is appended to the pre-filled
   name) and `press_enter: true`.
5. `Snapshot`: the window title must now read `gui_test.txt - Notepad`. Only then is it saved.

### Calculator — verified 2026-11-05 (Qwen2.5-VL, Big-Tiger-Gemma)
1. `App` launch Calculator, `Snapshot`, then `Click` each button by its element number from the Snapshot, one digit
   or operator at a time (1, 2, 3, ×, 4, 5, =).
2. Read the display from a `Snapshot`; report only the number it shows.

### New folder in Explorer — verified 2026-11-05 (by hand through Windows-MCP)
1. `App` launch File Explorer, `Snapshot`, find `edit "Address Bar"`.
2. `Type` the folder path (e.g. `C:\Users\tester\Desktop`) into the address bar with `clear: true` and
   `press_enter: true`. The window title becomes `Desktop - File Explorer`. A single `Click` on "Desktop" in the Home
   view only selects it and does not open it.
3. `Shortcut` ctrl+shift+n, then `Snapshot`: there is now an `edit "New folder"` marked `[focused]`.
4. `Type` the new name with `loc` set to that focused edit box, `clear: true` and `press_enter: true`.
5. `Snapshot`: the list must show the new name, not "New folder".

## Pitfalls
- Windows 11 Notepad reopens unsaved tabs after it is closed or killed, so it can start with old text.
  Check the window title is "Untitled - Notepad" before typing. — verified 2026-11-05
- Notepad: Qwen typed the text by `label` correctly, then reported the file saved; no gui_test.txt existed
  anywhere under C:\Users\tester. The Save dialog step is the open problem. — verified 2026-11-05
- A skills file can derail a model: with this file in its system prompt, Big-Tiger-Gemma stopped after its first
  tool call on every task, where it passed Calculator without it. Test a skills file per model. — verified 2026-11-05
- Big-Tiger-Gemma clicked Calculator buttons by coordinates and reported 5490 for 123 × 45 (true 5535).
- Tiger-Gemma 9B wrote pyautogui code or invented tool names ("App Open") instead of calling tools.

## Scripting the VM (non-GUI tools) — verified 2026-11-07
- `PowerShell` takes `{"command", "timeout"}`; **the timeout defaults to 30 s**, so pass it for installs and builds.
  Output comes wrapped as `Response: ... Status Code: n`: parse out the part between them.
- A command line is capped at about 32 KB (Windows), so do not push file contents through `PowerShell`. Upload with
  `FileSystem` `{"mode": "write", "append": true, "content": <base64 chunk>}` (2 MB chunks worked), then decode
  inside Windows with `[IO.File]::WriteAllBytes(path, [Convert]::FromBase64String((Get-Content -Raw f.b64)))`.
- Download by reading byte ranges through PowerShell (`[IO.File]::OpenRead`, `Seek`, `[Convert]::ToBase64String`);
  1.5 MB chunks moved a 69 MB .exe in about 2 minutes.
- The VM has internet (pypi.org, python.org) and about 110 GB free. Python 3.14 is in `C:\Python314`;
  `scripts/win_build.py` adds a per-user Python 3.12 for builds that need older wheels (eval7).
- Hold `~/windows11/in-use` for the whole run (win_build.py does) and remove build folders afterwards.
