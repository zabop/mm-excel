"""Test homogeneity of variance between the simulations making up a treatment.

Usage:
    python pessev_wrapper.py --pesinc 48,10,48,10 --pessev 20,3,20,3 \\
        | python heterogeneity_tests.py --simulation-count 2

    --simulation-count  how many consecutive simulations form one treatment.
                        With k, simulations 1..k are treatment 1, k+1..2k are
                        treatment 2, and so on. At least 2, and it has to divide
                        the number of simulations exactly.
    --levene-center     what Levene centres each group on: mean, median or
                        trimmed. Default mean, which is NOT scipy's default;
                        see LEVENE_CENTERS below for why the median centre is
                        blind on this particular data.

Reads the stdout document of pessev_wrapper.py on stdin and writes one JSON
document to stdout, holding a Levene and a Bartlett test per treatment. Within a
treatment the groups compared are the simulations, and each group is that
simulation's whole pessev_i column, zeros included: an uninfected subsample is
an observed severity of zero, not a missing one.

A run that gives up writes {"error": "..."} to stderr and exits non-zero, so the
stderr document parses whether the run succeeded or not.
"""

import argparse
import json
import math
import statistics
import sys
import warnings

from scipy.stats import bartlett, levene

from pessev import JsonArgumentParser, fail

# scipy defaults to "median" (the Brown-Forsythe variant), but that centre is
# degenerate on this data and the default here is "mean" instead. Whenever
# PESINC is below 50 the majority of a column is zero, so the median is zero,
# |x - median| collapses to x, and its mean is exactly PESSEV -- which the
# generator pins exactly. Every group then has the same mean absolute deviation
# and the statistic comes out 0.0 with a p-value of 1.0 no matter how far the
# variances really are apart. Measured: columns of variance 448 and 1421 score
# 0.0 on the median centre and 25.7 (p=6e-07) on the mean.
LEVENE_CENTERS = ("mean", "median", "trimmed")
DEFAULT_LEVENE_CENTER = "mean"


def finite(value):
    """A float that JSON can carry, or None.

    json.dumps writes a bare NaN or Infinity token for non-finite floats, which
    is not valid JSON and makes JSON.parse throw in the browser. Both tests
    reach non-finite results on data this generator really produces -- a
    simulation with PESINC 0 has an all-zero pessev_i, which leaves Bartlett
    dividing by a zero variance and Levene centring on all-zero residuals.
    """
    number = float(value)
    return number if math.isfinite(number) else None


def run_test(test, groups):
    """One test over the groups, as a JSON-safe object.

    scipy raises on input it cannot use at all and returns nan for input that is
    merely degenerate; both end up as a null statistic with the reason attached,
    so a treatment that cannot be tested says so instead of vanishing.
    """
    try:
        # scipy warns on stderr when a degenerate group makes it divide by zero,
        # which would put loose text in a stream that has to stay one JSON
        # document. The degenerate case is expected here and already reported
        # through the note below, so the warning has nothing left to add.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value = test(*groups)
    except ValueError as error:
        return {"statistic": None, "p_value": None, "note": str(error)}

    statistic, p_value = finite(statistic), finite(p_value)
    if statistic is None or p_value is None:
        # every group constant, or every group identical: the ratio the test
        # forms is 0/0, and scipy hands back nan rather than raising
        return {
            "statistic": statistic,
            "p_value": p_value,
            "note": "undefined for these groups, which carry no variance to compare",
        }
    return {"statistic": statistic, "p_value": p_value}


def analyse(data, simulation_count, levene_center=DEFAULT_LEVENE_CENTER):
    """Levene and Bartlett per treatment, over the simulations that form it."""
    simulations = data.get("simulations") if isinstance(data, dict) else None
    if not simulations:
        fail(
            "the input does not look like pessev_wrapper.py output: no "
            "simulations to group"
        )

    if simulation_count < 2:
        fail(
            f"--simulation-count is {simulation_count}, so each treatment would "
            f"hold a single group and there would be nothing to compare; it "
            f"must be at least 2"
        )
    if len(simulations) % simulation_count:
        fail(
            f"{len(simulations)} simulations do not divide evenly into "
            f"treatments of {simulation_count}"
        )

    treatments = []
    for start in range(0, len(simulations), simulation_count):
        members = simulations[start : start + simulation_count]
        # float, not the int the generator produces: scipy 1.18 derives the
        # dtype of its internal group-size array from the samples, and an
        # integer one cannot hold the nan it writes into it
        groups = [[float(v) for v in member["pessev_i"]] for member in members]
        treatments.append(
            {
                "treatment": len(treatments) + 1,
                "simulations": [member["index"] for member in members],
                "group_sizes": [len(group) for group in groups],
                "variances": [
                    finite(statistics.variance(group)) if len(group) > 1 else None
                    for group in groups
                ],
                "levene": run_test(
                    lambda *args: levene(*args, center=levene_center), groups
                ),
                "bartlett": run_test(bartlett, groups),
            }
        )

    return {
        "simulation_count": simulation_count,
        "treatment_count": len(treatments),
        "levene_center": levene_center,
        "treatments": treatments,
    }


def run(payload_text, simulation_count, levene_center=DEFAULT_LEVENE_CENTER):
    """The whole job, shared by the command line and the web page."""
    if levene_center not in LEVENE_CENTERS:
        fail(
            f"--levene-center must be one of {', '.join(LEVENE_CENTERS)}, "
            f"not {levene_center!r}"
        )
    try:
        data = json.loads(payload_text)
    except json.JSONDecodeError as error:
        fail(f"could not read the simulation data: {error}")

    # allow_nan=False turns a non-finite value that slipped past finite() into a
    # loud failure here rather than an unparseable document downstream
    print(
        json.dumps(
            analyse(data, simulation_count, levene_center), allow_nan=False
        )
    )


def parse_args(argv):
    parser = JsonArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--simulation-count",
        type=int,
        required=True,
        metavar="K",
        help="consecutive simulations forming one treatment",
    )
    parser.add_argument(
        "--levene-center",
        choices=LEVENE_CENTERS,
        default=DEFAULT_LEVENE_CENTER,
        help="what Levene centres each group on (default: %(default)s; see the "
        "module docstring for why this is not scipy's default)",
    )
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    run(sys.stdin.read(), args.simulation_count, args.levene_center)


if __name__ == "__main__":
    main(sys.argv[1:])
