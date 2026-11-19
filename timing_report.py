"""
timing_report.py  --  latency + speed summary of a play session

    python3 timing_report.py decisions.jsonl

Reads the log written by tetris_test.py and prints:
  * how long after a piece was FIRST SEEN the buttons were dispatched (vision + decision latency)
  * camera-frame age at dispatch (live camera runs only)
  * AI time, planned button time, measured fall speed
  * how often the planned presses were "tight" against the time the piece has before it lands
Send me this output (and a raw recording) if the bot misplaces pieces at higher levels.
"""
import json
import statistics
import sys


def stats(name, vals, unit):
    vals = [v for v in vals if v is not None]
    if not vals:
        print("  %-42s n/a" % name)
        return
    vals.sort()
    p90 = vals[min(len(vals) - 1, int(0.9 * len(vals)))]
    print("  %-42s median %7.2f  p90 %7.2f  max %7.2f %s   (n=%d)" % (
        name, statistics.median(vals), p90, vals[-1], unit, len(vals)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "decisions.jsonl"
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        print("no decisions in", path)
        return
    print("%d decisions in %s\n" % (len(recs), path))
    stats("first seen -> buttons dispatched", [r.get("first_seen_to_decision_s") for r in recs], "s")
    stats("camera frame age at dispatch", [r.get("frame_age_ms") for r in recs], "ms")
    stats("AI search time", [r.get("ai_ms") for r in recs], "ms")
    stats("planned button time (last strike)", [r.get("planned_press_s") for r in recs], "s")
    stats("measured fall speed", [r.get("sec_per_row") for r in recs], "s/row")
    tight = 0
    known = 0
    for r in recs:
        spr, plan = r.get("sec_per_row"), r.get("planned_press_s")
        if spr and plan is not None:
            known += 1
            room = 14 * spr - (r.get("first_seen_to_decision_s") or 0.0)      # ~14 rows of travel on a low stack
            if plan > 0.7 * room:
                tight += 1
    if known:
        print("\n  presses were 'tight' against the fall time in %d of %d decisions (%.0f%%)" % (tight, known, 100.0 * tight / known))
    print("  time-budgeted decisions: %d of %d" % (sum(1 for r in recs if r.get("time_budgeted")), len(recs)))
    used = sum(1 for r in recs if r.get("next_used"))
    print("  decisions that used the NEXT piece (lookahead): %d of %d" % (used, len(recs)))
    hl = [r.get("vision_health") for r in recs if r.get("vision_health")]
    if hl:
        c = [h["contrast"] for h in hl]
        print("  lighting contrast (block - background brightness): min %.0f  median %.0f   (LOW if < 70)" % (min(c), statistics.median(c)))


if __name__ == "__main__":
    main()
