---
name: publish-safety
description: Commit, push, release or rewrite history in this PUBLIC repo (github.com/nodalytics/poker-analysis) and its public sibling nodalytics/ragflow-lab without leaking anything or breaking attribution rules. Use it before every git commit or push, before publishing results, logs, videos or docs, when copying files from the lab or another project into the repo, and whenever the user asks to remove something from the repo "and its history", purge a file, or clean up contributors.
---

# Publishing safely

Both repos are public. Anything pushed can be cached, forked or indexed within minutes. Deleting it later doesn't
undo that, and even a history rewrite leaves the old commits reachable by hash until GitHub Support purges them.
On 2026-11-22/23 this repo needed three history rewrites:
- a private note copied in from another project;
- `Co-Authored-By: Claude` trailers on six early commits, which made Claude show as a contributor;
- the trading scripts.

Each was cheap to prevent and expensive to undo.

## Before every commit

1. **Stage by path, never `git add -A`.** Other sessions share these checkouts, and `-A` sweeps up their files.
2. **Scan the staged diff** for anything private. Check the patterns below, then read the diff yourself.
   ```sh
   git diff --cached | grep -n -i -E '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|@[a-z0-9-]+\.[a-z]|BEGIN .*PRIVATE|sk-[a-z0-9]|ghp_|token=|password|/home/[a-z]'
   ```
   - **Never commit:** lab addresses, logins and key files (`.secrets/`, `*.pem`); API keys (the coding server's
     `sk-lab-...` keys); the Kestrel access token; the Windows VM credentials; personal email addresses; notes from
     other projects.
   - Results copied from the lab carry `/home/<user>/` paths. Replace them with `~`.
3. **Commit as ojies:**
   `git -c user.name=ojies -c user.email=68871517+ojies@users.noreply.github.com commit ...`. Never add a
   `Co-Authored-By` or other attribution trailer, even when a tool suggests one: GitHub turns it into a listed
   contributor.
4. **Lint:** from `research/`, run `uv run ruff check scripts` and `uv run ruff format --check scripts`, with no path
   filter.
5. **Media:** check every frame or screenshot of a video or GIF for the lab address, tokens, keys or a terminal
   prompt showing the login, before it goes anywhere.

## Pushing

- `main` is pushed directly here. For `ragflow-lab`, a push deploys to the lab within 5 minutes, so don't push while
  something depends on the coding-model server.
- After pushing, confirm with `git status -sb` that you're level with the remote, and check open PRs if branches
  moved.

## Removing something from history (only when the user asks)

1. Fetch, and make a backup first:
   `git bundle create .secrets/backup-before-<what>-<date>.bundle --all`.
2. List every path the material ever had, including old directory names:
   `git log --all --name-only --format= | sort -u | grep ...`.
3. Rewrite only our refs: `main`, live branches and every tag.
   ```sh
   git filter-repo --force --invert-paths --path <p1> --path <p2> ... \
       --refs refs/heads/main refs/heads/<branch> $(git tag | sed 's#^#refs/tags/#')
   ```
   - For commit messages (for example, trailers), use `--message-callback` instead.
   - Create local branches for remote-only branches first. A branch deleted on GitHub can't be rewritten; PR refs
     (`refs/pull/N/head`) are read-only.
4. Verify before pushing:
   - the material is gone from the rewritten refs (search them, not `--all`, which still includes the stale remote
     refs);
   - `main`'s tree is unchanged when the material was already deleted;
   - commit counts match;
   - every other ref differs only in the removed paths.
5. `git push --force` the branches, then `--tags`. Check that PRs are still mergeable and that releases are still
   attached to their tags.
6. Tell the user what only they can do: ask GitHub Support to purge the old SHAs and PR refs, and reset any other
   clones (including the lab's).
