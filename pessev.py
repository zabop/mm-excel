"""Generate PESINC / PESSEV subsample data for one treatment.

Usage:
    python pessev.py PESINC PESSEV [SAMPLE_COUNT] [PESSEV_SPREAD] [SEED]

    PESINC         incidence as a percentage, e.g. 48
    PESSEV         mean severity as a percentage over *all* samples, e.g. 20
    SAMPLE_COUNT   number of subsamples, default 200
    PESSEV_SPREAD  coefficient of variation of severity among infected samples,
                   as a percentage, default 60. 0 = every infected sample
                   identical, ~60 realistic, >=150 heavily skewed towards mild
                   infections. A cv is a ratio, so this one may exceed 100.
    SEED           random seed, default 20260828

Writes a tab separated pesinc_i / pessev_i table to stdout and a summary to
stderr, so `python pessev.py 48 20 > out.tsv` keeps the data clean. Everything
is in whole percent points: pesinc_i is a 0/1 flag and pessev_i is an integer
severity, so a severity of 5% appears in the table as 5.
"""

import itertools
import math
import random
import sys

# whole percent points, which is also the unit the table is written in -- integer
# severities are what lets the mean come out exact instead of drifting over a few
# hundred float additions
POSSIBLE_PESSEV_VALUES = [
    0, 1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100,
]  # fmt: skip


def calculate_pesinc_i(sample_count, pesinc):
    """Shuffled list of `sample_count` flags, `pesinc` percent of them set."""
    samples_with_pests_count = int(sample_count * pesinc / 100)
    pesinc_i = [1] * samples_with_pests_count + [0] * (
        sample_count - samples_with_pests_count
    )
    return sorted(pesinc_i, key=lambda _: random.random())


def _one_step_moves(values, allowed, rank):
    """Map every reachable one-rung delta to the indices that can make it."""
    moves = {}
    for i, value in enumerate(values):
        for step in (-1, 1):
            neighbour = rank[value] + step
            if 0 <= neighbour < len(allowed):
                moves.setdefault(allowed[neighbour] - value, []).append(i)
    return moves


def _match_target_sum(values, allowed, target_sum, max_moves=4):
    """Walk values along the allowed ladder until they sum to exactly target_sum.

    The one-rung gaps are 2, 5 and 10 percent points; 2 and 5 are coprime, so
    any integer residual can be closed by a short combination of moves.
    """
    rank = {value: i for i, value in enumerate(allowed)}
    total = sum(values)

    # coarse: keep taking the single move that shrinks the residual the most,
    # which brings it below one rung of the ladder
    while total != target_sum:
        residual = target_sum - total
        moves = _one_step_moves(values, allowed, rank)
        best = min(moves, key=lambda delta: (abs(residual - delta), -abs(delta)))
        if abs(residual - best) >= abs(residual):
            break
        i = random.choice(moves[best])
        values[i] += best
        total += best

    # fine: close what is left with a combination of moves on distinct samples,
    # e.g. a residual of 1 becomes +5, -2, -2. Greedy single moves stall here.
    residual = target_sum - total
    moves = _one_step_moves(values, allowed, rank)
    for count in range(1, max_moves + 1):
        for combo in itertools.combinations_with_replacement(sorted(moves), count):
            if sum(combo) != residual:
                continue
            picked, plan = set(), []
            for delta in combo:
                i = next((i for i in moves[delta] if i not in picked), None)
                if i is None:
                    break
                picked.add(i)
                plan.append((i, delta))
            else:
                for i, delta in plan:
                    values[i] += delta
                return values

    return values  # closest reachable; the caller reports the leftover residual


def calculate_pessev_i(
    pesinc_i, pessev, spread, possible_values=POSSIBLE_PESSEV_VALUES
):
    """Severity per subsample, averaging exactly `pessev` percent over the list.

    Zero wherever pesinc_i is zero, non-zero wherever it is one. `spread` is a
    coefficient of variation in percent, i.e. 60 means a cv of 0.6.
    """
    allowed = sorted({value for value in possible_values if value > 0})
    infected = [i for i, incidence in enumerate(pesinc_i) if incidence == 1]
    target_sum = round(pessev * len(pesinc_i))

    # every infected sample carries at least the smallest allowed value and at
    # most the largest, which pins how low and how high the mean can reach
    lowest = len(infected) * allowed[0]
    highest = len(infected) * allowed[-1]
    if not lowest <= target_sum <= highest:
        raise ValueError(
            f"PESSEV {pessev:g}% is out of reach: {len(infected)} infected "
            f"samples, each between {allowed[0]}% and {allowed[-1]}%, hold the mean "
            f"between {lowest / len(pesinc_i):.2f}% and {highest / len(pesinc_i):.2f}%"
        )

    pessev_i = [0] * len(pesinc_i)
    if not infected:
        return pessev_i

    mean_infected = target_sum / len(infected)
    if spread <= 0:
        draws = [mean_infected] * len(infected)
    else:
        cv = spread / 100
        shape = 1 / cv**2  # a gamma with this shape has cv == spread
        draws = [random.gammavariate(shape, mean_infected / shape) for _ in infected]

    values = [min(allowed, key=lambda a: (abs(a - draw), a)) for draw in draws]
    values = _match_target_sum(values, allowed, target_sum)

    for i, value in zip(infected, values):
        pessev_i[i] = value
    return pessev_i


def main(argv):
    if not 3 <= len(argv) <= 6:
        sys.exit(__doc__)

    try:
        pesinc = float(argv[1])
        pessev = float(argv[2])
        sample_count = int(argv[3]) if len(argv) > 3 else 200
        spread = float(argv[4]) if len(argv) > 4 else 60.0
        seed = int(argv[5]) if len(argv) > 5 else 20260828
    except ValueError as error:
        sys.exit(f"could not read the arguments: {error}")

    if not 0 <= pesinc <= 100 or not 0 <= pessev <= 100:
        sys.exit("PESINC and PESSEV are percentages and must lie between 0 and 100")
    if sample_count < 1:
        sys.exit("SAMPLE_COUNT must be at least 1")
    if spread < 0:
        sys.exit("PESSEV_SPREAD cannot be negative")

    # the algorithm distributes whole percent points, so the requested mean has
    # to land on an integer number of them
    target_sum = pessev * sample_count
    if not math.isclose(target_sum, round(target_sum)):
        sys.exit(
            f"PESSEV * SAMPLE_COUNT = {target_sum} is not a whole number of "
            f"percent points, so the mean cannot be matched"
        )

    random.seed(a=seed)
    pesinc_i = calculate_pesinc_i(sample_count, pesinc)
    try:
        pessev_i = calculate_pessev_i(pesinc_i, pessev, spread)
    except ValueError as error:
        sys.exit(str(error))

    assert all(value in POSSIBLE_PESSEV_VALUES for value in pessev_i)
    assert all(sev == 0 for sev, inc in zip(pessev_i, pesinc_i) if inc == 0)
    assert all(sev > 0 for sev, inc in zip(pessev_i, pesinc_i) if inc == 1)

    print("pesinc_i\tpessev_i")
    for incidence, severity in zip(pesinc_i, pessev_i):
        print(f"{incidence}\t{severity}")

    infected = [severity for severity in pessev_i if severity > 0]
    achieved = sum(pessev_i) / sample_count
    print(f"infected samples : {len(infected)} of {sample_count}", file=sys.stderr)
    print(
        f"mean severity    : {achieved:.4f}% (target {pessev:.4f}%, "
        f"off by {achieved - pessev:+.4f})",
        file=sys.stderr,
    )
    if not infected:
        return pesinc_i, pessev_i

    # PESSEV / PESINC is only reachable when PESINC * SAMPLE_COUNT is whole,
    # otherwise the infected count is truncated and the two disagree slightly
    print(
        f"mean on infected : {sum(infected) / len(infected):.4f}% "
        f"(target {pessev / pesinc * 100:.4f}%)",
        file=sys.stderr,
    )
    print(
        f"severity spread  : {min(infected)}% .. {max(infected)}%", file=sys.stderr
    )
    for value in sorted(set(infected)):
        print(f"  {value:>3}% {infected.count(value):>4}", file=sys.stderr)

    # the CLI ignores this, but the web page copies the columns from it instead
    # of parsing its own stdout back out of the DOM
    return pesinc_i, pessev_i


if __name__ == "__main__":
    main(sys.argv)
