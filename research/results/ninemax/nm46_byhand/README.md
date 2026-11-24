# nm46 (by hand): nm25's recipe with more aggressive opponents

A training run started by hand on the lab on 2026-11-24 at 17:38 UTC, outside the pipeline: no run file, so no
pre-registration and no checks. Kept here so it can be examined, and registered properly if it is worth it. The name
nm46 was later wanted for the A3 follow-up (`research/docs/2026-11-23-review-and-plan.md`); that run takes the next free
name instead.

- `command.txt`: the exact command; `config.json`: the settings the trainer recorded; `progress.log`: its evaluations
  every 30 minutes (copied when saved; the lab's `blueprint/nm46/progress.log` is the full one).

What differs from nm25 (both start a tournament 9-max network, mtt menu, ante 1, depths 10-25bb):

| Setting | nm25 | nm46 (by hand) |
|---|---|---|
| start | nm20 | nm25 |
| rule players in the pool | reg, reg, tag, maniac, station | maniac, maniac, tag, station, reg |
| share of tables against the pool | 0.5 | 0.6 |
| exploiter share | 0.25 | 0.35 |
| exploiter restart | every 60 min | every 45 min |
| snapshots kept | every 2 h | every 1 h |

No league, no prioritised opponents and no archetype styles (nm45 has all three). At its first two evaluations it
read +132.6 and +132.3 bb/100 against the calling station, against nm25's level of about +118, with the exploiter near
zero. It trained on the 3090 at about 42,000 hands a second.

To judge it as the pipeline would, register a run file with this recipe's settings (`research/pipeline/README.md`)
and nm45's checks; the station check is the one this recipe aims at.

The same session first ran this recipe as `blueprint/nm45` at 17:09 UTC, in the pipeline's live nm45 folder. It was
stopped before it saved a checkpoint, but it replaced nm45's `config.json` with these settings; the pipeline's file was
restored from the interrupted nm45 run, trained with the identical recipe.
