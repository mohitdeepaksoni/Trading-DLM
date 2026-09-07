# Pre-registration rule for event-rotation files (fixed 2026-09-07)

The tariff event harness is a same-day instrument (research/11): Agora's producer-price response ranks the day-0
reaction of sector ETFs and nothing after. A live file therefore only counts as evidence if it demonstrably existed
before the market could react. The rule:

1. **Source text first.** The official action text (Federal Register notice, White House fact sheet, court order — the
   quoted passage that states rates, scope and effective date) is saved verbatim. Its sha256 goes into the file.
2. **Mechanical mapping.** Text → ISIC shock dict follows the fixed conventions of `emporion/structure/events_us_tariffs.py`
   (partner trade shares, goods-only scope, exemptions as listed). No judgement after seeing prices.
3. **Stamp before the open.** `scripts/event_rotation.py --shock-json … --source-text …` is run and the file is chained
   and OpenTimestamps-stamped **before 13:30 UTC** (09:30 New York) on the event's first trading day. The stamp time
   is the evidence. Files built later carry `post_hoc: true` inside their hash and are excluded from every live tally.
4. **Everything from the historical table is post-hoc** by construction (built 2026-09-05 with hindsight); it is the
   calibration set, never the live record.
5. **One file per event.** A revised action is a new event with its own text and stamp; the earlier file stands.
6. **Tally.** Live evidence = mean day-0 book excess return and rank-IC over stamped, non-post-hoc files, reported with
   n and the permutation null exactly as the harness does; nothing is certified below n = 20 live events.
