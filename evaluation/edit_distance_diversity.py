import argparse
import os
import statistics
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

from tqdm import tqdm

try:
    from Levenshtein import distance as fast_levenshtein_distance  # type: ignore
except ImportError:
    fast_levenshtein_distance = None


def read_nonempty_sequences(file_path: str) -> List[str]:
    sequences: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            seq = line.strip()
            if seq:
                sequences.append(seq)
    return sequences


def levenshtein_distance_python(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Use two-row DP to keep memory linear in min(len(a), len(b)).
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, replace))
        prev = cur
    return prev[-1]


def select_engine(engine: str) -> str:
    if engine == "auto":
        return "levenshtein" if fast_levenshtein_distance is not None else "python"
    if engine == "levenshtein" and fast_levenshtein_distance is None:
        raise ValueError(
            "engine=levenshtein requested but python-Levenshtein is not installed. "
            "Install it with: pip install python-Levenshtein"
        )
    return engine


def compute_distance(a: str, b: str, engine: str) -> int:
    if engine == "levenshtein":
        assert fast_levenshtein_distance is not None
        return int(fast_levenshtein_distance(a, b))
    return levenshtein_distance_python(a, b)


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def merge(self, other: "RunningStats") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            return

        total = self.count + other.count
        delta = other.mean - self.mean
        self.mean = self.mean + delta * other.count / total
        self.m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / total
        self.count = total

    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        return (self.m2 / (self.count - 1)) ** 0.5


def compute_chunk_stats(
    sequences: List[str],
    start_i: int,
    end_i: int,
    engine: str,
) -> Tuple[int, float, float, int, float, float]:
    edit_stats = RunningStats()
    norm_stats = RunningStats()

    for i in range(start_i, end_i):
        s1 = sequences[i]
        for j in range(i + 1, len(sequences)):
            s2 = sequences[j]
            d = compute_distance(s1, s2, engine)
            denom = max(len(s1), len(s2))
            nd = d / denom if denom > 0 else 0.0
            edit_stats.update(float(d))
            norm_stats.update(nd)

    return (
        edit_stats.count,
        edit_stats.mean,
        edit_stats.m2,
        norm_stats.count,
        norm_stats.mean,
        norm_stats.m2,
    )


def pairwise_edit_stats_single_process(
    sequences: List[str],
    engine: str,
) -> Tuple[int, float, float, float, float]:
    total_pairs = len(sequences) * (len(sequences) - 1) // 2
    if total_pairs == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    edit_stats = RunningStats()
    norm_stats = RunningStats()

    with tqdm(total=total_pairs, desc="Computing edit distance") as pbar:
        for i in range(len(sequences)):
            s1 = sequences[i]
            for j in range(i + 1, len(sequences)):
                s2 = sequences[j]
                d = compute_distance(s1, s2, engine)
                denom = max(len(s1), len(s2))
                nd = d / denom if denom > 0 else 0.0
                edit_stats.update(float(d))
                norm_stats.update(nd)
                pbar.update(1)

    return edit_stats.count, edit_stats.mean, edit_stats.std(), norm_stats.mean, norm_stats.std()


def generate_i_ranges(num_sequences: int, chunk_size: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for start_i in range(0, num_sequences, chunk_size):
        end_i = min(start_i + chunk_size, num_sequences)
        ranges.append((start_i, end_i))
    return ranges


def _worker_chunk(args: Tuple[List[str], int, int, str]) -> Tuple[int, float, float, int, float, float]:
    sequences, start_i, end_i, engine = args
    return compute_chunk_stats(sequences, start_i, end_i, engine)


def pairwise_edit_stats_multi_process(
    sequences: List[str],
    engine: str,
    num_workers: int,
    chunk_size: int,
) -> Tuple[int, float, float, float, float]:
    total_pairs = len(sequences) * (len(sequences) - 1) // 2
    if total_pairs == 0:
        return 0, 0.0, 0.0, 0.0, 0.0

    ranges = generate_i_ranges(len(sequences), chunk_size)
    tasks = [(sequences, start_i, end_i, engine) for start_i, end_i in ranges]

    edit_stats = RunningStats()
    norm_stats = RunningStats()

    with Pool(processes=num_workers) as pool:
        with tqdm(total=total_pairs, desc="Computing edit distance") as pbar:
            for count_e, mean_e, m2_e, count_n, mean_n, m2_n in pool.imap_unordered(_worker_chunk, tasks):
                local_edit = RunningStats()
                local_edit.count = count_e
                local_edit.mean = mean_e
                local_edit.m2 = m2_e

                local_norm = RunningStats()
                local_norm.count = count_n
                local_norm.mean = mean_n
                local_norm.m2 = m2_n

                edit_stats.merge(local_edit)
                norm_stats.merge(local_norm)
                pbar.update(count_e)

    return edit_stats.count, edit_stats.mean, edit_stats.std(), norm_stats.mean, norm_stats.std()

    total_pairs = len(sequences) * (len(sequences) - 1) // 2
    if total_pairs == 0:
        return edit_distances, normalized_distances

    for i, j in tqdm(combinations(range(len(sequences)), 2), total=total_pairs, desc="Computing edit distance"):
        s1, s2 = sequences[i], sequences[j]
        d = levenshtein_distance(s1, s2)
        denom = max(len(s1), len(s2))
        nd = d / denom if denom > 0 else 0.0
        edit_distances.append(d)
        normalized_distances.append(nd)

    return edit_distances, normalized_distances


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate edit-distance diversity for sequences in a file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to file containing one sequence per line.")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="If > 0, only use first N non-empty lines for faster estimation.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="auto",
        choices=["auto", "python", "levenshtein"],
        help="Distance backend: auto prefers python-Levenshtein if installed.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes. Use >1 to parallelize pair computation.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Number of outer-loop indices per worker task chunk.",
    )
    args = parser.parse_args()

    sequences = read_nonempty_sequences(args.input_file)
    if args.sample_size > 0:
        sequences = sequences[: args.sample_size]

    engine = select_engine(args.engine)

    print(f"Number of sequences used: {len(sequences)}")
    print(f"Distance engine: {engine}")
    print(f"Workers: {args.num_workers}")

    if len(sequences) < 2:
        print("Need at least 2 sequences to compute diversity.")
        return

    if args.num_workers <= 0:
        raise ValueError("num_workers must be >= 1")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")

    if args.num_workers == 1:
        num_pairs, mean_edit, std_edit, mean_norm, std_norm = pairwise_edit_stats_single_process(
            sequences=sequences,
            engine=engine,
        )
    else:
        max_workers = cpu_count() or 1
        workers = min(args.num_workers, max_workers)
        num_pairs, mean_edit, std_edit, mean_norm, std_norm = pairwise_edit_stats_multi_process(
            sequences=sequences,
            engine=engine,
            num_workers=workers,
            chunk_size=args.chunk_size,
        )

    if num_pairs == 0:
        print("No pairwise distances computed.")
        return

    mean_similarity = 1.0 - mean_norm

    print(f"Number of pairs: {num_pairs}")
    print(f"Mean edit distance: {mean_edit:.4f}")
    print(f"Edit distance std: {std_edit:.4f}")
    print(f"Mean normalized edit distance (diversity): {mean_norm:.4f}")
    print(f"Mean normalized similarity (1 - diversity): {mean_similarity:.4f}")
    print(f"Normalized edit distance std: {std_norm:.4f}")

    output_file = os.path.join(os.path.dirname(args.input_file), "edit_distance_diversity.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Number of sequences used: {len(sequences)}\n")
        f.write(f"Distance engine: {engine}\n")
        f.write(f"Workers: {args.num_workers}\n")
        f.write(f"Number of pairs: {num_pairs}\n")
        f.write(f"Mean edit distance: {mean_edit:.4f}\n")
        f.write(f"Edit distance std: {std_edit:.4f}\n")
        f.write(f"Mean normalized edit distance (diversity): {mean_norm:.4f}\n")
        f.write(f"Mean normalized similarity (1 - diversity): {mean_similarity:.4f}\n")
        f.write(f"Normalized edit distance std: {std_norm:.4f}\n")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
