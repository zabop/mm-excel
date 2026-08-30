"""Generate PESINC / PESSEV subsample data for several treatments at once.

Usage:
    python pessev_wrapper.py --pesinc 5,10,25,5 --pessev 1,3,5,1
                             [--sample-count 200] [--pessev-spread 60]
                             [--threshold 10]

    --pesinc         comma separated incidences as percentages, one per
                     simulation, e.g. 5,10,25,5
    --pessev         comma separated mean severities as percentages over *all*
                     samples, one per simulation, e.g. 1,3,5,1. Paired with
                     --pesinc by position, so the two lists need the same
                     length.
    --sample-count   number of subsamples per simulation, default 200. One
                     value, shared by every simulation.
    --pessev-spread  coefficient of variation of severity among infected
                     samples, as a percentage, default 60. One value, shared by
                     every simulation. See pessev.py for what the number means.
    --threshold      severity threshold as a percentage. Optional, and only
                     when it is given does the threshold table appear. One
                     value, shared by every simulation.

Writes one JSON document to stdout and another to stderr, so
`python pessev_wrapper.py --pesinc 5,10 --pessev 1,3 > out.json` keeps the data
clean. stdout carries the data: a `simulations` list holding each simulation's
pesinc_i and pessev_i columns, in the order the arguments named them. stderr
carries the summary: the same list, but with the counts and means that describe
each simulation rather than the samples themselves. Nothing is seeded, so every
invocation differs.

With --threshold, every summary entry also carries a threshold_counts object
splitting that simulation's samples into unaffected, below the threshold, and at
or above it. Without it, threshold_counts is null.

A run that gives up writes {"error": "..."} to stderr and exits non-zero, so the
stderr document parses whether the run succeeded or not.
"""

import argparse
import json
import math
import sys

from pessev import (
    POSSIBLE_PESSEV_VALUES,
    calculate_pesinc_i,
    calculate_pessev_i,
    fail,
    summarise,
)


class JsonArgumentParser(argparse.ArgumentParser):
    """argparse, but its own errors are JSON as well.

    Bad arguments are reported by argparse itself, which would otherwise print
    a usage block and leave stderr unparseable.
    """

    def error(self, message):
        fail(f"could not read the arguments: {message}")


def comma_separated_floats(text):
    """Read "5,10,25" as [5.0, 10.0, 25.0], for argparse's type=."""
    values = []
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            raise argparse.ArgumentTypeError(f"empty value in {text!r}")
        try:
            values.append(float(entry))
        except ValueError:
            raise argparse.ArgumentTypeError(f"{entry!r} is not a number")
    return values


def parse_args(argv):
    parser = JsonArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pesinc",
        type=comma_separated_floats,
        required=True,
        metavar="A,B,C",
        help="incidences as percentages, one per simulation",
    )
    parser.add_argument(
        "--pessev",
        type=comma_separated_floats,
        required=True,
        metavar="A,B,C",
        help="mean severities as percentages, one per simulation",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=200,
        metavar="N",
        help="subsamples per simulation (default: %(default)s)",
    )
    parser.add_argument(
        "--pessev-spread",
        type=float,
        default=60.0,
        metavar="CV",
        help="coefficient of variation of severity, as a percentage "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="PCT",
        help="severity threshold as a percentage; adds a table counting each "
        "simulation's unaffected, below threshold and at or above threshold "
        "samples (default: no such table)",
    )
    return parser.parse_args(argv)


def validate(pesinc_values, pessev_values, sample_count, spread, threshold):
    """Reject bad input before a single row reaches stdout.

    Everything here is checked up front rather than per simulation as it is
    generated, so a bad third simulation cannot leave two simulations' worth of
    rows on stdout before it gives up.
    """
    if len(pesinc_values) != len(pessev_values):
        fail(
            f"--pesinc has {len(pesinc_values)} values and --pessev has "
            f"{len(pessev_values)}; they are paired by position, so both lists "
            f"must be the same length"
        )
    if sample_count < 1:
        fail("--sample-count must be at least 1")
    if spread < 0:
        fail("--pessev-spread cannot be negative")
    if threshold is not None and not 0 <= threshold <= 100:
        fail("--threshold is a severity percentage and must lie between 0 and 100")

    allowed = sorted({value for value in POSSIBLE_PESSEV_VALUES if value > 0})
    for index, (pesinc, pessev) in enumerate(
        zip(pesinc_values, pessev_values), start=1
    ):
        if not 0 <= pesinc <= 100 or not 0 <= pessev <= 100:
            fail(
                f"simulation {index}: PESINC and PESSEV are percentages and must "
                f"lie between 0 and 100"
            )

        # the algorithm distributes whole percent points, so the requested mean
        # has to land on an integer number of them
        target = pessev * sample_count
        if not math.isclose(target, round(target)):
            fail(
                f"simulation {index}: PESSEV * SAMPLE_COUNT = {target} is not a "
                f"whole number of percent points, so the mean cannot be matched"
            )

        # the same reachability check calculate_pessev_i makes, repeated here
        # because the infected count does not depend on the random draws
        infected_count = int(sample_count * pesinc / 100)
        lowest = infected_count * allowed[0]
        highest = infected_count * allowed[-1]
        if not lowest <= round(target) <= highest:
            fail(
                f"simulation {index}: PESSEV {pessev:g}% is out of reach: "
                f"{infected_count} infected samples, each between {allowed[0]}% "
                f"and {allowed[-1]}%, hold the mean between "
                f"{lowest / sample_count:.2f}% and {highest / sample_count:.2f}%"
            )


def run_one(index, pesinc, pessev, sample_count, spread):
    """The pesinc_i / pessev_i columns for one simulation."""
    pesinc_i = calculate_pesinc_i(sample_count, pesinc)
    try:
        pessev_i = calculate_pessev_i(pesinc_i, pessev, spread)
    except ValueError as error:  # validate() should have caught this already
        fail(f"simulation {index}: {error}")

    assert all(value in POSSIBLE_PESSEV_VALUES for value in pessev_i)
    assert all(sev == 0 for sev, inc in zip(pessev_i, pesinc_i) if inc == 0)
    assert all(sev > 0 for sev, inc in zip(pessev_i, pesinc_i) if inc == 1)
    return pesinc_i, pessev_i


def main(argv):
    args = parse_args(argv)
    validate(
        args.pesinc,
        args.pessev,
        args.sample_count,
        args.pessev_spread,
        args.threshold,
    )

    # stdout describes the samples, stderr describes the simulations; both are
    # assembled in full before either is written, so a run that gives up part
    # way through cannot leave half a document on stdout
    data = {
        "sample_count": args.sample_count,
        "pessev_spread": args.pessev_spread,
        "threshold": args.threshold,
        "simulations": [],
    }
    summary = {"threshold": args.threshold, "simulations": []}

    for index, (pesinc, pessev) in enumerate(zip(args.pesinc, args.pessev), start=1):
        pesinc_i, pessev_i = run_one(
            index, pesinc, pessev, args.sample_count, args.pessev_spread
        )
        data["simulations"].append(
            {
                "index": index,
                "pesinc": pesinc,
                "pessev": pessev,
                "pesinc_i": pesinc_i,
                "pessev_i": pessev_i,
            }
        )
        summary["simulations"].append(
            {
                "index": index,
                **summarise(
                    pesinc, pessev, args.sample_count, pessev_i, args.threshold
                ),
            }
        )

    print(json.dumps(data))
    print(json.dumps(summary), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
