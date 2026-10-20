"""Trial design (A) and CONSORT flow (B), rebuilt from the schema and consort panel makers.

Both panels measure their own height from the text, so the rows use height="auto".
Swap the text and numbers for your own trial; positions follow the panel width.

Run:  python trial_design_figure.py [--out out] [--formats png svg]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from nejm_plots import (Band, Bracket, Composite, Event, Exit, Item, Para, RunIn, Theme,
                        consort_panel, trial_schema_panel)

GREEN = "#3A9142"
BLUE_BAND, BLUE_POST, BLUE_TXT, BLUE_POST_TXT = "#BCD8EE", "#DAEAF6", "#1F5C99", "#3D7DBF"
GREY_BAND, GREY_POST, GREY_TXT, GREY_POST_TXT = "#E3E3E3", "#F1F1F1", "#4D4D4D", "#666666"
MINUS, GE, EN = "−", "≥", "–"


def trial_design():
    criteria = [
        Para("Screening", weight="bold", size=8.2),
        Para("People living with HIV with a history of suboptimal HIV control", hang=2.1),
        Item("1.", "Documented detectable HIV-1 viral load >1000 copies/ml in the preceding 2 yr "
                   f"despite receiving oral ART for {GE}3 mo", indent=1.0),
        Item("2.", "History of being lost to follow-up (>4 wk elapsed since a scheduled clinic "
                   "appointment or drug refill in preceding 2 yr)", indent=1.0),
        Item("3.", f"Unlinked to HIV care despite {GE}3 mo elapsing since HIV diagnosis", indent=1.0),
        Para("Anyone receiving an NNRTI will be required to switch to dolutegravir before or "
             "during screening", hang=1.4, space_before=0.5),
    ]
    run_in = RunIn(-4, -0.18, [
        Para("Screening Phase", weight="bold", align="center", size=8.0),
        Para("Virologically suppressed (HIV-1 viral load <200 copies/ml) for "
             f"{GE}3 mo at the end of screening", align="center", space_before=0.15),
        Para("2 NRTIs+DTG", weight="bold", align="center", size=8.0, space_before=0.5),
    ])
    bands = [
        Band("CAB LA, 600 mg+RPV LA, 900 mg\nIntramuscularly every 8 wk\n(N=270)", 1, 24,
             fc=BLUE_BAND, color=BLUE_TXT,
             post=Band("4-yr Post-trial access\nto CAB LA+RPV LA\nevery 8 wk", 24, 72, fc=BLUE_POST,
                       color=BLUE_POST_TXT, weight="normal")),
        Band("2 NRTIs+DTG\n(N=270)", 0, 24, fc=GREY_BAND, color=GREY_TXT,
             post=Band("Transition to standard care", 24, 72, fc=GREY_POST, color=GREY_POST_TXT,
                       weight="normal")),
    ]
    events = [
        Event(-0.5, "Confirm viral load\n<200 copies/ml", width=1.1),
        Event(0, "Randomization\n(storage of plasma\nand PBMCs)", width=1.1, dx=-0.27, lift=0.36),
        Event(12, [Para("Primary End Point:", weight="bold", align="center"),
                   Para("HIV viral load <50 copies/ml", align="center"),
                   Para("Secondary end points: virologic nonresponse, confirmed virologic failure, "
                        "safety, HIV disease progression, drug resistance, acceptability, treatment "
                        "preference", align="center")], width=1.62, lift=0.0),
        Event(24, "Termination", width=0.9),
    ]
    anchors = [(-4, 0.0, f"{MINUS}4\nmo"), (-0.5, 0.1335, f"{MINUS}2\nwk"), (0, 0.241, "0"),
               (1, 0.26, "1\nmo"), (12, 0.495, "12\nmo"), (24, 0.749, "24\nmo"), (72, 1.0, "72\nmo")]
    return trial_schema_panel(
        anchors, bands, events=events, breaks=[(-4, -0.5), (-0.5, 0, 0.3), (24, 72)], run_in=run_in,
        randomization=0, bracket=Bracket(0, 1, "Optional 1-mo\noral lead-in"), criteria=criteria,
        axis_title="Time Relative to Randomization", title="Trial Design")


def flow():
    return consort_panel(
        pre=["845 Participants were assessed for eligibility",
             Exit("305 Were excluded (may have >1 reason for exclusion)",
                  [(87, "Did not meet required viral suppression criteria"),
                   (84, "Had evidence of chronic HBV (HBsAg-positive) or had previous infection "
                        f"without immunity (anti-HBc{EN}positive and anti-HBs{EN}negative)"),
                   (28, "Were not receiving first-line regimen"),
                   (129, "Had other reason")]),
             "540 Underwent randomization"],
        arms=[["271 Were assigned to receive long-acting therapy",
               Exit("3 Switched to standard oral treatment during oral lead-in phase"),
               "268 Began long-acting therapy and were included in the ITT-exposed analysis",
               Exit("24 Were excluded (may have >1 reason for exclusion)",
                    [(4, "Did not follow originally assigned treatment through week 48"),
                     (5, "Had protocol deviation related to drug administration"),
                     (19, "Had protocol deviation related to visit window")]),
               "244 Were included in the per-protocol analysis"],
              ["269 Were assigned to continue oral therapy",
               None,
               "269 Were included in the ITT-exposed analysis",
               Exit("21 Were excluded (may have >1 reason for exclusion)",
                    [(2, "Did not follow originally assigned treatment through week 48"),
                     (2, "Had protocol deviation related to drug administration"),
                     (18, "Had protocol deviation related to visit window")]),
               "248 Were included in the per-protocol analysis"]],
        title="Screening, Randomization, and Analysis")


def build(theme=None, width=7.1):
    c = Composite(width=width, theme=theme or Theme.nejm())
    c.row([trial_design()], height="auto")
    c.row([flow()], height="auto")
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--formats", nargs="+", default=["png", "svg"])
    a = ap.parse_args()
    out = Path(a.out)
    build().save(out / "fig9_trial_design_consort", formats=a.formats)
    # each panel on its own
    th = Theme.nejm()
    trial_design().save(out / "panels" / "trial_design", width=7.1, height="auto", theme=th, formats=a.formats)
    flow().save(out / "panels" / "consort", width=7.1, height="auto", theme=th, formats=a.formats)
    print("done")
