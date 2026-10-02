"""Contract the whole d = 5 cultivation circuit in every run its detectors allow.

``Cultivation.ipynb`` (Part 6) shows that the reduced d = 5 circuit, written in the
variables of the faces of the authors' final check, is T-dag|+> fed into the colour
code of those faces, and checks the whole circuit against that state by tensor
contraction in 4 runs. This script does the same for all 1024 runs that the derived
detectors allow (they fix 117 of the circuit's 127 Boolean variables in terms of the
other ten), in the order of ``consistent_assignments``. One line per run goes to
standard output, or to ``--log``; the script stops at the first run that differs.

A run takes a median of 18 s, from 4 s to about half an hour (PyZX's tensor
contraction after its own simplification), so all 1024 take about nine hours on a
laptop. ``--start`` and ``--end`` pick a range of runs, to split the work:

    python examples/cultivation_d5_all_runs.py                          # all 1024 runs
    python examples/cultivation_d5_all_runs.py --start 0 --end 512 --log a.log
    python examples/cultivation_d5_all_runs.py --start 512 --log b.log
"""

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")                         # interactive_cultivation_d5_3d picks a window backend on import

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from interactive_cultivation_d5_3d import CIRCUIT, faces, textbook  # noqa: E402
from zxqec.gpf import boolean_variables, consistent_assignments, reduce_gpf, same_map  # noqa: E402
from zxqec.qprogram import program_from_stim  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", type=int, default=0, help="first run (default 0)")
    ap.add_argument("--end", type=int, default=None, help="one past the last run (default: all)")
    ap.add_argument("--log", help="append one line per run to this file instead of printing it")
    args = ap.parse_args()

    with open(CIRCUIT) as f:
        p, _ = program_from_stim(f.read())
    g = p.quantum_graph()
    res = reduce_gpf(g)
    qubits = [int(q) for q in p.code_at(p.num_stages - 1).data().qubits]
    want = textbook(qubits, faces())           # T-dag|+> in the colour code of the faces
    variables = sorted(set(boolean_variables(g)) | {str(v) for c in res.constraints for v in c.free_vars()})
    runs = consistent_assignments(res.constraints, variables)
    end = len(runs) if args.end is None else min(args.end, len(runs))

    out = open(args.log, "a") if args.log else sys.stdout
    print("{} Boolean variables, {} runs allowed; checking runs {}..{}".format(
        len(variables), len(runs), args.start, end - 1), file=out, flush=True)
    for i in range(args.start, end):
        t0 = time.time()
        ok = same_map(g.substitute_variables(runs[i]), want)
        print(i, ok, "{:.0f}s".format(time.time() - t0), file=out, flush=True)
        if not ok:
            print("DIFFERS in run", i, file=out, flush=True)
            sys.exit(1)
    print("ALL RUNS {}..{} EQUAL".format(args.start, end - 1), file=out, flush=True)


if __name__ == "__main__":
    main()
