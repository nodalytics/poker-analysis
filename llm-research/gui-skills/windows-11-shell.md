# Windows 11 shell — GUI skills

- **Version tested:** Windows 11 Enterprise Evaluation in `win11-gui` (dockurr/windows 6.06)
- **Last verified:** not yet; every entry below is **unverified** standard Windows 11 behaviour, to be
  confirmed on the VM through Windows-MCP and marked as it is
- **Control path:** Windows-MCP (UI Automation tree first, screenshots when the tree is empty)

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

None recorded yet.
