#!/usr/bin/env python3
"""
Rehash an AHF halos catalogue to append direct host/subhalo information.

Fixes included:
- halo IDs are parsed as integers directly, never via float
- automatic detection of AHF ID encoding:
    standard: ID = 1e12 * isnap + (ihalo + 1)
    MPI     : ID = 1e16 * isnap + 1e10 * rank + (ihalo + 1)

For each halo:
- main halo: its centre does not lie within the search radius of any larger halo
             (where "larger" follows the user's convention: smaller haloid)
- subhalo:   its centre lies within the search radius of one or more larger halos

Two host-assignment schemes are written:

1) Standard centre-in-r200 test:
       dist <= r200_host
   producing columns:
       sub_count  sub_id

2) Expanded test:
       dist <= sqrt(r200_host^2 + 0.5 * r200_sub^2)
   producing columns:
       sub_count_wide  sub_id_wide

Among all larger halos that contain a halo centre, the assigned host is the
smallest such host halo, i.e. the containing host with the largest haloid.
Main halos get host id 0.

The script preserves the original file line-by-line and appends four extra columns:
    sub_count  sub_id  sub_count_wide  sub_id_wide
"""


import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

STD_FACTOR = 10**12
MPI_FACTOR = 10**16
MPI_RANK_FACTOR = 10**10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append host/subhalo columns to an AHF halos file."
    )
    p.add_argument("input_file", type=Path, help="Input AHF_halos file")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./AHF_new"),
        help="Directory for output file (default: ./AHF_new)",
    )
    p.add_argument(
        "--boxsize",
        type=float,
        default=1.0e6,
        help="Periodic box size in kpc (default: 1e6)",
    )
    p.add_argument("--id-col", type=int, default=0, help="Halo ID column (default: 0)")
    p.add_argument("--x-col", type=int, default=5, help="x column (default: 5)")
    p.add_argument("--y-col", type=int, default=6, help="y column (default: 6)")
    p.add_argument("--z-col", type=int, default=7, help="z column (default: 7)")
    p.add_argument("--r200-col", type=int, default=11, help="r200 column (default: 11)")
    return p.parse_args()


def is_data_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith("#")


def detect_id_scheme(haloids: np.ndarray) -> str:
    if haloids.size == 0:
        raise ValueError("Cannot detect ID scheme from an empty catalogue.")

    snap_std = haloids // STD_FACTOR
    snap_mpi = haloids // MPI_FACTOR

    uniq_std = np.unique(snap_std)
    uniq_mpi = np.unique(snap_mpi)

    plausible_std = np.all((uniq_std >= 0) & (uniq_std < 10000))
    plausible_mpi = np.all((uniq_mpi >= 0) & (uniq_mpi < 10000))

    good_std = plausible_std and uniq_std.size == 1
    good_mpi = plausible_mpi and uniq_mpi.size == 1

    if good_std and not good_mpi:
        return "standard"
    if good_mpi and not good_std:
        return "mpi"

    if good_std and good_mpi:
        if np.median(haloids) >= 10**16:
            return "mpi"
        return "standard"

    raise ValueError(
        "Could not determine haloid encoding automatically. "
        f"Unique snap estimates from //1e12: {uniq_std[:10]} (n={uniq_std.size}); "
        f"from //1e16: {uniq_mpi[:10]} (n={uniq_mpi.size})."
    )


def read_ahf_file(
    path: Path,
    id_col: int,
    x_col: int,
    y_col: int,
    z_col: int,
    r200_col: int,
) -> Tuple[List[str], List[int], np.ndarray, np.ndarray, np.ndarray]:
    all_lines = path.read_text().splitlines()
    data_line_idx: List[int] = []
    centres = []
    haloids = []
    r200 = []

    max_col = max(id_col, x_col, y_col, z_col, r200_col)

    for i, line in enumerate(all_lines):
        if not is_data_line(line):
            continue
        parts = line.split()
        if len(parts) <= max_col:
            raise ValueError(
                f"Data line {i+1} has only {len(parts)} columns, "
                f"but column {max_col} was requested."
            )
        data_line_idx.append(i)
        haloids.append(int(parts[id_col]))
        centres.append([float(parts[x_col]), float(parts[y_col]), float(parts[z_col])])
        r200.append(float(parts[r200_col]))

    if not haloids:
        raise ValueError("No data rows found in input file.")

    return (
        all_lines,
        data_line_idx,
        np.asarray(centres, dtype=np.float64),
        np.asarray(haloids, dtype=np.int64),
        np.asarray(r200, dtype=np.float64),
    )


def periodic_distances(points: np.ndarray, centre: np.ndarray, boxsize: float) -> np.ndarray:
    delta = np.abs(points - centre)
    delta = np.minimum(delta, boxsize - delta)
    return np.sqrt(np.sum(delta * delta, axis=1))


def find_hosts_standard(
    haloids: np.ndarray,
    centres: np.ndarray,
    r200: np.ndarray,
    boxsize: float,
) -> np.ndarray:
    n = haloids.size
    sub_id = np.zeros(n, dtype=np.int64)

    if np.any(r200 < 0.0):
        raise ValueError("Negative r200 encountered.")

    tree = cKDTree(centres, boxsize=boxsize)
    search_radius = float(np.max(r200))

    for i in range(n):
        cand = np.asarray(tree.query_ball_point(centres[i], r=search_radius), dtype=np.int64)
        cand = cand[cand != i]
        if cand.size == 0:
            continue

        cand = cand[haloids[cand] < haloids[i]]
        if cand.size == 0:
            continue

        dist = periodic_distances(centres[cand], centres[i], boxsize)
        cand = cand[dist <= r200[cand]]
        if cand.size == 0:
            continue

        best = cand[np.argmax(haloids[cand])]
        sub_id[i] = haloids[best]

    return sub_id


def find_hosts_wide(
    haloids: np.ndarray,
    centres: np.ndarray,
    r200: np.ndarray,
    boxsize: float,
) -> np.ndarray:
    n = haloids.size
    sub_id = np.zeros(n, dtype=np.int64)

    if np.any(r200 < 0.0):
        raise ValueError("Negative r200 encountered.")

    max_r = float(np.max(r200))
    max_search_radius = float(np.sqrt(max_r * max_r + 0.5 * max_r * max_r))
    tree = cKDTree(centres, boxsize=boxsize)

    for i in range(n):
        cand = np.asarray(tree.query_ball_point(centres[i], r=max_search_radius), dtype=np.int64)
        cand = cand[cand != i]
        if cand.size == 0:
            continue

        cand = cand[haloids[cand] < haloids[i]]
        if cand.size == 0:
            continue

        dist = periodic_distances(centres[cand], centres[i], boxsize)
        thresh = np.sqrt(r200[cand] * r200[cand] + 0.5 * r200[i] * r200[i])
        cand = cand[dist <= thresh]
        if cand.size == 0:
            continue

        best = cand[np.argmax(haloids[cand])]
        sub_id[i] = haloids[best]

    return sub_id


def make_counts(haloids: np.ndarray, sub_id: np.ndarray) -> np.ndarray:
    counts = Counter(sub_id[sub_id > 0].tolist())
    return np.array([counts.get(int(hid), 0) for hid in haloids], dtype=np.int64)


def make_output_path(input_file: Path, output_dir: Path) -> Path:
    name = input_file.name
    if name.endswith(".AHF_halos"):
        outname = name[:-10] + ".AHF_subhalos"
    else:
        outname = name + ".subhalos"
    return output_dir / outname


def write_output(
    all_lines: List[str],
    data_line_idx: List[int],
    sub_count: np.ndarray,
    sub_id: np.ndarray,
    sub_count_wide: np.ndarray,
    sub_id_wide: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    line_map = {line_idx: row_idx for row_idx, line_idx in enumerate(data_line_idx)}

    first_data_line = data_line_idx[0]
    header_line_to_patch = None
    for i in range(first_data_line - 1, -1, -1):
        if all_lines[i].strip().startswith("#"):
            header_line_to_patch = i
            break

    out_lines = []
    for i, line in enumerate(all_lines):
        if i == header_line_to_patch:
            out_lines.append(f"{line} sub_count sub_id sub_count_wide sub_id_wide")
            continue

        if i in line_map:
            j = line_map[i]
            out_lines.append(
                f"{line} {int(sub_count[j])} {int(sub_id[j])} "
                f"{int(sub_count_wide[j])} {int(sub_id_wide[j])}"
            )
        else:
            out_lines.append(line)

    output_path.write_text("\n".join(out_lines) + "\n")


def main() -> None:
    args = parse_args()

    print(f"Input file : {args.input_file}")
    output_path = make_output_path(args.input_file, args.output_dir)
    print(f"Output file: {output_path}")
    print(f"Boxsize    : {args.boxsize:g} kpc")

    all_lines, data_line_idx, centres, haloids, r200 = read_ahf_file(
        args.input_file,
        id_col=args.id_col,
        x_col=args.x_col,
        y_col=args.y_col,
        z_col=args.z_col,
        r200_col=args.r200_col,
    )

    print(f"Read {haloids.size} halos")

    scheme = detect_id_scheme(haloids)
    if scheme == "standard":
        snap = int(np.unique(haloids // STD_FACTOR)[0])
        print(f"Detected haloid scheme: standard (1e12), snapshot = {snap}")
    else:
        snap = int(np.unique(haloids // MPI_FACTOR)[0])
        ranks = np.unique((haloids % MPI_FACTOR) // MPI_RANK_FACTOR)
        print(
            f"Detected haloid scheme: MPI (1e16), snapshot = {snap}, "
            f"nranks present = {ranks.size}"
        )

    sub_id = find_hosts_standard(haloids, centres, r200, args.boxsize)
    sub_count = make_counts(haloids, sub_id)
    print(
        f"Standard criterion: main halos = {np.count_nonzero(sub_id == 0)}, "
        f"subhalos = {np.count_nonzero(sub_id != 0)}"
    )

    sub_id_wide = find_hosts_wide(haloids, centres, r200, args.boxsize)
    sub_count_wide = make_counts(haloids, sub_id_wide)
    print(
        f"Wide criterion    : main halos = {np.count_nonzero(sub_id_wide == 0)}, "
        f"subhalos = {np.count_nonzero(sub_id_wide != 0)}"
    )

    write_output(
        all_lines=all_lines,
        data_line_idx=data_line_idx,
        sub_count=sub_count,
        sub_id=sub_id,
        sub_count_wide=sub_count_wide,
        sub_id_wide=sub_id_wide,
        output_path=output_path,
    )

    print("Done.")


if __name__ == "__main__":
    main()
