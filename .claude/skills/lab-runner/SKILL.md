---
name: lab-runner
description: Run anything on the lab machine (the GPU box this repo trains and evaluates on) safely and so it survives — training runs, exploiters, sequential tests, evals, builds, model pulls, tunnels to lab services. Use it whenever a command will ssh to the lab, needs a GPU, will run longer than a couple of minutes, or touches lab services (Ollama, Kestrel, the coding-model server), even if the user only says "run it on the lab", "kick off nmXX", "test this on the GPU" or "check the lab".
---

# Running work on the lab

The lab is shared: other people's services and jobs run there, and some of our jobs are timed by the wall clock.
Most of what went wrong on 2026-11-23 was not the code. One SSH connection dropped twice, and this machine went to
sleep and froze an evaluation for six hours. Lost connections were scored as model failures. This skill exists to
avoid repeating that.

## Connect once, reuse it

- The address, login and key are in `.secrets/lab.env` (`LAB`, `LAB_KEY`), which is gitignored. Never print them,
  and never put them in a doc, commit or message.
- Open one ControlMaster connection and send every command through it. Many short connections get refused for about
  an hour, while the lab keeps serving existing ones.
  ```sh
  . .secrets/lab.env
  C=/tmp/claude-$(id -u)/cm-lab; rm -f $C
  timeout 45 ssh -fN -M -o ControlPath=$C -o ControlPersist=yes -o ConnectTimeout=30 -o ServerAliveInterval=20 \
      -o IdentitiesOnly=yes -i "$LAB_KEY" "$LAB"
  ssh -o ControlPath=$C x 'nvidia-smi'          # later commands: any host name works through the socket
  ```
- **Tunnels** go through the same connection (`ssh -o ControlPath=$C -O forward -L 11435:127.0.0.1:11435 x`).
  When the connection dies, the tunnel dies with it. A long local loop must check the connection and the tunnel
  before each step and reopen them: `ssh -O check`, then the server's health URL.
- **Keep this machine awake** for anything local that drives the lab for long:
  `systemd-inhibit --what=sleep:idle:handle-lid-switch --why="lab job" <command>`.

## Run jobs so they outlive the session

- Long jobs run **detached on the lab** and write to files:
  `cd ~/research && (setsid nohup <cmd> > out.log 2>&1 &)`. The parentheses matter: `a && setsid nohup b &`
  backgrounds the whole chain and holds the SSH session open.
- **Wait on output, not on processes.** Use `until grep -q "finished" queue.log; do sleep 60; done`. Never use
  `pkill -f` or `pgrep -f` in the same command as the job: the pattern matches its own command line, so the wait
  never ends, or the kill takes down your own session.
- **Stop exactly what you started,** by its PID, or by the port it listens on
  (`ss -ltnp | grep :PORT`). Leave every other process and container alone.

## GPUs

- Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. In that order, 0 = RTX 3090, 1 = RTX 4090, 2 = RTX PRO 6000 (96 GB). Without
  it, CUDA's numbering differs from nvidia-smi's.
- The **4090 hosts the coding-model server** (`coder-ollama`, ragflow-lab). Don't train there.
- **One timed job per GPU.** The 30-minute exploiter and similar jobs are limited by wall-clock time, so sharing a GPU
  quietly shrinks their training and biases the reading. When runs land on GPUs of different speeds, give each model
  the same number of seeds on each GPU.
- The trainer and evaluators are CUDA-only. CPU checks run with `CUDA_VISIBLE_DEVICES=` and `OMP_NUM_THREADS=4`.

## Don't disturb what's there

- `~/research` is what other jobs read. Test changed code from a scratch copy (`~/claude-work/<name>/`), and only
  copy into `~/research` when nothing running depends on the old version.
- `lab.sh sync` uses `rsync --delete` and removes files other agents put in `~/research`, so prefer copying
  individual files.
- Ports in use: 8000, 8790, 8890, 8088. Bind new services to `127.0.0.1` only, and reach them through a tunnel.
- Don't push to `nodalytics/ragflow-lab` while an evaluation is using the coding-model server. Its CD redeploys within
  5 minutes and restarts the server.
- Python on the lab: `~/research/.venv-gpu/bin/python` (torch, CUDA). The `.venv` there has no torch.

## After a run

- Copy results into the repo's `research/results/<topic>/`. Replace lab paths (`/home/<user>`) with `~`, and scan for
  addresses, user names and keys before committing.
- Analyse locally, not on the lab.
- A run that failed for infrastructure reasons (a lost connection, a dead tunnel, a full disk) is rerun, not scored.
  Keep its partial output under a name that says so.
