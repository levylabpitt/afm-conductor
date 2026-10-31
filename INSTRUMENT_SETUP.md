# Setting up an instrument PC by hand

This does by hand what `install-config-conductor.cmd` and `install-config-hardware.cmd` (one time, in that order) and `run-hardware-twin.cmd`
(every day) do. Use it if you prefer to do the steps yourself, or if a script fails.
Commands are for a Windows Command Prompt or PowerShell opened in this folder.

## One-time setup

### 1. Install uv

uv installs Python and the program's libraries for you. No administrator rights needed.

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close the window and open a new one so `uv` is found. Check with `uv --version`.

### 2. Install the libraries

```
uv sync --extra hardware --extra geometry
```

This includes `pywin32`, which the program uses to talk to Igor.

### 3. Swap Igor.exe for Igor 6.37

The Asylum software ships its own Igor 6.38 inside the Asylum folder. Igor 6.38
has no COM interface, which this program needs; Igor 6.37 has it. Only `Igor.exe`
is replaced. Your Asylum licence and experiments are not touched.

1. Close Igor completely (check the system tray).
2. Find the Asylum `Igor.exe`. It is normally at
   `C:\AsylumResearch\v<number>\RealTime\Igor Pro Folder\Igor.exe`
   (the drive may be `D:`). In File Explorer, right-click it, choose
   Properties, then Details, and note the file version (6.38 expected).
3. Download the WaveMetrics Igor 6.37 installer:
   https://www.wavemetrics.net/Downloads/Win/setupIgor6.exe
4. Run it and click through with the default choices. It installs a separate
   copy, normally in `C:\Program Files\WaveMetrics\Igor Pro Folder\`
   (or `C:\Program Files (x86)\WaveMetrics\Igor Pro Folder\`). The existing
   Asylum licence carries over; there is nothing to enter.
5. Check that the new `Igor.exe` reports version 6.37 (Properties, Details).
6. In the Asylum `Igor Pro Folder`, rename `Igor.exe` to `Igor.exe.6.38-backup`.
   If a backup with that name already exists, leave it alone and delete the
   current `Igor.exe` instead. Windows may ask for administrator permission.
7. Copy the 6.37 `Igor.exe` into the Asylum `Igor Pro Folder`.
8. Check the Asylum `Igor.exe` now reports version 6.37.

To undo: close Igor, delete the Asylum `Igor.exe`, and rename
`Igor.exe.6.38-backup` back to `Igor.exe`.

### 4. Install SOCKIT and the program's Igor procedures

Igor must be closed. Replace the folder with your Asylum `Igor Pro Folder`
(the folder that contains `Igor.exe`).

```
uv run --extra hardware scripts\install_sockit.py --download --no-igor --folder "C:\AsylumResearch\v19\RealTime\Igor Pro Folder"
uv run --extra hardware scripts\install_afm_ipf.py --folder "C:\AsylumResearch\v19\RealTime\Igor Pro Folder"
```

The first downloads SOCKIT from WaveMetrics and copies it into `Igor Extensions`
and `More Extensions`. The second copies the `.ipf` files from this folder's
`igor` directory into the Igor `Igor Procedures` and `User Procedures` folders.
Copying those files there yourself does the same job: see the top of each script
for which file goes where.

## Every day

1. Set the two environment variables, then start the service. It opens the
   cockpit in your browser by itself. Keep this window open.

   ```
   set AFM_LITHO_BACKEND=hardware
   set AFM_LITHO_WIRE_VERSION=3
   uv run --no-sync --extra hardware afm-conductor
   ```

   In PowerShell, use `$env:AFM_LITHO_BACKEND = "hardware"` and
   `$env:AFM_LITHO_WIRE_VERSION = "3"` instead of `set`.

2. Open Igor with the Asylum software loaded.

3. In a second window, in this folder, connect Igor to the service:

   ```
   uv run --no-sync --extra hardware scripts\start_bridge.py
   ```

   It compiles the bridge in Igor and dials the service. Wait for the message
   that the bridge is live.

To stop, double-click `Stop AFM.cmd`.
