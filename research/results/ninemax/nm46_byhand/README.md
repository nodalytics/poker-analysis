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

## nm45's checks, run on nm46 (2026-11-27)

Run by hand with the pipeline's own commands and sizes (`checks/`, one log and one JSON a check), against nm25, plus
two direct comparisons with nm45. These were not pre-registered for nm46, so they describe it; registering the recipe
as a run (nm47) is what would adopt it.

| Check | nm45 (pipeline, 2026-11-24) | nm46 |
|---|---|---|
| vs calling station (difference vs nm25, bb/100) | -16.53 [-17.24, -15.83], fail | +1.15 [+0.58, +1.71] |
| vs maniac | -3.54 [-3.97, -3.12], fail | +0.92 [+0.57, +1.28] |
| vs reg | -0.11 [-0.41, +0.18] | +0.00 [-0.30, +0.30] |
| vs tag | +0.20 [-0.14, +0.53] | +0.06 [-0.27, +0.39] |
| 25 styles | mean +0.17, worst -1.55 (maniac) | mean -0.03, worst -1.38 (style06) |
| no-harm, 40,000 deals vs nm25 | -0.06, no difference | -0.23, no difference |
| bluffers: jam-any, jam-40, raise-any | -53.1, -12.1, -12.3 | -51.5, -13.6, -10.5 |
| exploiter, 4 seeds of 30 minutes (limit +1.2) | +0.5 | running |

Direct, nm46 against nm45: +17.68 [+17.00, +18.36] more against the station, +4.46 [+4.03, +4.90] against the maniac,
reg and tag level; at one table, 40,000 deals, -0.19 bb/100, no difference.

Both networks are on the lab's Kestrel as practice models (nm25 stays the default), installed as verified model
packages in its data volume, so no rebuild was needed.
