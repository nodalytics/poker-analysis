# Windows 11 shell — GUI skills

- **Version tested:** Windows 11 Enterprise Evaluation in `win11-gui` (dockurr/windows 6.06)
- **Last verified:** 2026-11-05 through the noVNC web viewer (Playwright on `127.0.0.1:8006`); entries marked
  `unverified` are standard Windows 11 behaviour not yet tried on the VM
- **Control path:** Windows-MCP (UI Automation tree first, screenshots when the tree is empty); noVNC as a fallback

## Driving it through noVNC (fallback when Windows-MCP is down) — verified 2026-11-05

- The canvas sits at y=130 in a 913px-wide page; the taskbar search box is at about (390, 684).
- **Click the canvas once before typing**, or keystrokes go to the desktop (they selected the Recycle Bin).
- **Shifted characters are dropped** by `keyboard.type`: `:` became `;`, `&` became `7`, capitals became lowercase,
  `|` became `\`. Send them as `keyboard.down('Shift')`, `press('Semicolon' | 'Quote' | 'Digit7' | 'Backslash' |
  'Period' | 'Comma' | 'Minus' | 'Digit5' | 'Key<X>')`, `up('Shift')`. Plain letters, digits, `\`, `/`, `-`, `.` are fine.
- Open a command prompt: click the search box, type `cmd`, wait 2s, then click **Open** in the right pane at about
  (657, 368). Pressing Enter too soon after typing does nothing. The prompt opens as Administrator.
- Run several commands as separate lines rather than joining them with `&`.

## Launch and focus

- Start menu: `Win`, type the app name, `Enter`. Works for most apps without knowing their path. — unverified
- Run dialog: `Win+R`, type a command (`notepad`, `explorer`, `cmd`, `ms-settings:`), `Enter`. — unverified
- Switch windows: `Alt+Tab`; show the desktop: `Win+D`. — unverified
- Snap a window: `Win+Left` / `Win+Right`; maximise: `Win+Up`. — unverified

## Layout

| Region | UI Automation | Notes |
|---|---|---|
| Taskbar | `ControlType: Pane`, class `Shell_TrayWnd` | unverified |
| Start menu search box | `ControlType: Edit` in the Start pane | unverified |
| File Explorer address bar | `ControlType: Edit`, focus with `Ctrl+L` or `Alt+D` | unverified |

## Tasks

### Open an app and type into it — unverified

1. `Win+R`, type `notepad`, `Enter`.
2. Wait for a window whose title ends in `Notepad`.
3. Type the text; `Ctrl+S` opens Save As. Type the full path in the file name box, `Enter`.

### Open a folder in File Explorer — unverified

1. `Win+E`, then `Ctrl+L`, type the path, `Enter`.

### Open a Settings page directly — unverified

1. `Win+R`, type an `ms-settings:` URI (for example `ms-settings:display`), `Enter`.

## Keyboard shortcuts

| Keys | Does |
|---|---|
| `Win` | Start menu |
| `Win+R` | Run dialog |
| `Win+E` | File Explorer |
| `Win+I` | Settings |
| `Alt+F4` | Close the active window |
| `Ctrl+Shift+Esc` | Task Manager |

## Dialogs and interruptions

- First logon can show "Let's finish setting up your device"; close it or choose "Skip for now". — unverified
- Notepad in Windows 11 may reopen previous tabs; check the active tab before typing. — unverified

## Pitfalls

- The `\\host.lan\Data` share (container path `/tmp/smb`) did not show files copied in while Windows was running;
  Windows kept listing only `readme.txt`. Put files in `oem/` before an install instead. — verified 2026-11-05
- First boot shows a Windows Security "Update virus protection" toast and the Start menu; neither blocks input.
  — verified 2026-11-05
