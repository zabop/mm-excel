"""Generate PESINC / PESSEV subsample data for several treatments at once.

Usage:
    python pessev_wrapper.py --pesinc 5,10,25,5 --pessev 1,3,5,1
                             [--sample-count 200] [--pessev-spread 60]

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

Writes a tab separated simulation / pesinc_i / pessev_i table to stdout and a
summary per simulation to stderr, so
`python pessev_wrapper.py --pesinc 5,10 --pessev 1,3 > out.tsv` keeps the data
clean. The first column counts the simulations from one, so at the default
sample count rows 1-200 belong to simulation 1, rows 201-400 to simulation 2,
and so on. Nothing is seeded, so every invocation differs.
"""

import argparse
import math
import sys

from pessev import POSSIBLE_PESSEV_VALUES, calculate_pesinc_i, calculate_pessev_i


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
    parser = argparse.ArgumentParser(
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
    return parser.parse_args(argv)


def validate(pesinc_values, pessev_values, sample_count, spread):
    """Reject bad input before a single row reaches stdout.

    Everything here is checked up front rather than per simulation as it is
    generated, so a bad third simulation cannot leave two simulations' worth of
    rows on stdout before it gives up.
    """
    if len(pesinc_values) != len(pessev_values):
        sys.exit(
            f"--pesinc has {len(pesinc_values)} values and --pessev has "
            f"{len(pessev_values)}; they are paired by position, so both lists "
            f"must be the same length"
        )
    if sample_count < 1:
        sys.exit("--sample-count must be at least 1")
    if spread < 0:
        sys.exit("--pessev-spread cannot be negative")

    allowed = sorted({value for value in POSSIBLE_PESSEV_VALUES if value > 0})
    for index, (pesinc, pessev) in enumerate(
        zip(pesinc_values, pessev_values), start=1
    ):
        if not 0 <= pesinc <= 100 or not 0 <= pessev <= 100:
            sys.exit(
                f"simulation {index}: PESINC and PESSEV are percentages and must "
                f"lie between 0 and 100"
            )

        # the algorithm distributes whole percent points, so the requested mean
        # has to land on an integer number of them
        target = pessev * sample_count
        if not math.isclose(target, round(target)):
            sys.exit(
                f"simulation {index}: PESSEV * SAMPLE_COUNT = {target} is not a "
                f"whole number of percent points, so the mean cannot be matched"
            )

        # the same reachability check calculate_pessev_i makes, repeated here
        # because the infected count does not depend on the random draws
        infected_count = int(sample_count * pesinc / 100)
        lowest = infected_count * allowed[0]
        highest = infected_count * allowed[-1]
        if not lowest <= round(target) <= highest:
            sys.exit(
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
        sys.exit(f"simulation {index}: {error}")

    assert all(value in POSSIBLE_PESSEV_VALUES for value in pessev_i)
    assert all(sev == 0 for sev, inc in zip(pessev_i, pesinc_i) if inc == 0)
    assert all(sev > 0 for sev, inc in zip(pessev_i, pesinc_i) if inc == 1)
    return pesinc_i, pessev_i


def print_summary(index, pesinc, pessev, sample_count, pessev_i):
    """The report pessev.py writes for a single run, one block per simulation."""
    print(
        f"=== simulation {index}: PESINC {pesinc:g}%, PESSEV {pessev:g}% ===",
        file=sys.stderr,
    )

    infected = [severity for severity in pessev_i if severity > 0]
    achieved = sum(pessev_i) / sample_count
    print(f"infected samples : {len(infected)} of {sample_count}", file=sys.stderr)
    print(
        f"mean severity    : {achieved:.4f}% (target {pessev:.4f}%, "
        f"off by {achieved - pessev:+.4f})",
        file=sys.stderr,
    )
    if not infected:
        return

    # PESSEV / PESINC is only reachable when PESINC * SAMPLE_COUNT is whole,
    # otherwise the infected count is truncated and the two disagree slightly
    print(
        f"mean on infected : {sum(infected) / len(infected):.4f}% "
        f"(target {pessev / pesinc * 100:.4f}%)",
        file=sys.stderr,
    )
    print(f"severity spread  : {min(infected)}% .. {max(infected)}%", file=sys.stderr)
    for value in sorted(set(infected)):
        print(f"  {value:>3}% {infected.count(value):>4}", file=sys.stderr)


def main(argv):
    args = parse_args(argv)
    validate(args.pesinc, args.pessev, args.sample_count, args.pessev_spread)

    print("simulation\tpesinc_i\tpessev_i")
    for index, (pesinc, pessev) in enumerate(zip(args.pesinc, args.pessev), start=1):
        pesinc_i, pessev_i = run_one(
            index, pesinc, pessev, args.sample_count, args.pessev_spread
        )
        for incidence, severity in zip(pesinc_i, pessev_i):
            print(f"{index}\t{incidence}\t{severity}")
        print_summary(index, pesinc, pessev, args.sample_count, pessev_i)


if __name__ == "__main__":
    main(sys.argv[1:])
