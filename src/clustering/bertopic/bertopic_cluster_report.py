"""Write clustering variation metrics to a text report file."""

import math
from pathlib import Path
from datetime import datetime


METRICS_DOCUMENTATION = """
METRIC DEFINITIONS
-------------------------------------------------

1. PURITY (per-topic and weighted)
   - Per-topic purity: For each cluster, the fraction of documents that have the
     *dominant* emotion (the most frequent emotion in that cluster).
     Formula: purity(topic k) = max_emotion count_in_topic_k(emotion) / size(topic k).
   - Mean purity: Average of per-topic purities over all topics (each topic weighted equally).
   - Weighted purity: Purity weighted by cluster size.
     Formula: sum over topics of [ purity(topic k) * (size(topic k) / total docs) ].
   - Interpretation: High purity = each cluster is dominated by one emotion. Does not
     measure whether clusters separate different emotions well (e.g. all-positive vs
     all-negative still gives moderate purity).

2. NMI (Normalized Mutual Information)
   - Measures how much knowing the cluster ID reduces uncertainty about the emotion label.
   - Formula: NMI(emotion; cluster) = MI(emotion, cluster) / average(H(emotion), H(cluster)).
     MI = mutual information, H = entropy. We use sklearn's normalized_mutual_info_score
     with average_method="arithmetic".
   - Range: 0 (cluster and emotion independent) to 1 (cluster perfectly predicts emotion).
   - Interpretation: High NMI = clusters are informative for emotion (e.g. positive vs
     negative, or finer separation). Good for "do clusters help classify emotion?"

3. HOMOGENEITY
   - Each cluster contains only one class (emotion)?
   - Formula: 1 - H(emotion | cluster) / H(emotion). Homogeneity is 1 when every cluster
     has documents of only one emotion.
   - Range: 0 to 1. High = each cluster is "pure" in terms of emotion.

4. COMPLETENESS
   - Each emotion is assigned to only one cluster?
   - Formula: 1 - H(cluster | emotion) / H(cluster). Completeness is 1 when all documents
     of a given emotion fall in a single cluster.
   - Range: 0 to 1. High = each emotion is concentrated in one cluster (no split across
     many clusters).

5. TOPICS AND OUTLIERS
   - Topics: Number of distinct topic IDs excluding -1.
   - Outliers: Number of documents assigned topic -1 (e.g. by HDBSCAN when not in any cluster).

"""


def write_report(all_results: list, out_path: Path) -> None:
    """Write all variation results to a single text file for comparison."""
    with open(out_path, "w") as f:
        f.write("BERTopic Variation Comparisons\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(METRICS_DOCUMENTATION)
        f.write("\n")

        # Summary table (no truncation; column widths from data; * = best in column)
        col_topics = 7
        col_outliers = 9
        col_purity = 11  # 10 for value + 1 for marker
        col_nmi = 8
        col_homog = 8
        col_compl = 8
        col_name = max(len(r["name"]) for r in all_results) if all_results else 40
        sep_len = col_name + col_topics + col_outliers + col_purity + col_nmi + col_homog + col_compl + 6
        valid = [r for r in all_results if "error" not in r]
        max_purity = max((r.get("weighted_purity", r.get("mean_purity", 0)) for r in valid), default=0) if valid else 0
        nmi_vals = [r.get("nmi", 0) for r in valid if not (isinstance(r.get("nmi"), float) and math.isnan(r.get("nmi", 0)))]
        homog_vals = [r.get("homogeneity", 0) for r in valid if not (isinstance(r.get("homogeneity"), float) and math.isnan(r.get("homogeneity", 0)))]
        compl_vals = [r.get("completeness", 0) for r in valid if not (isinstance(r.get("completeness"), float) and math.isnan(r.get("completeness", 0)))]
        max_nmi = max(nmi_vals, default=0)
        max_homog = max(homog_vals, default=0)
        max_compl = max(compl_vals, default=0)
        f.write("-" * sep_len + "\n")
        f.write(
            f"{'Variation':<{col_name}} {'Topics':>{col_topics}} {'Outliers':>{col_outliers}} "
            f"{'Purity':>{col_purity}} {'NMI':>{col_nmi}} {'Homog':>{col_homog}} {'Compl':>{col_compl}}\n"
        )
        f.write("-" * sep_len + "\n")
        for r in all_results:
            purity = r.get("weighted_purity", r.get("mean_purity", 0))
            nmi = r.get("nmi", 0)
            homog = r.get("homogeneity", 0)
            compl = r.get("completeness", 0)
            if isinstance(nmi, float) and math.isnan(nmi):
                nmi = 0.0
            if isinstance(homog, float) and math.isnan(homog):
                homog = 0.0
            if isinstance(compl, float) and math.isnan(compl):
                compl = 0.0
            p_mark = "*" if (valid and purity == max_purity) else " "
            n_mark = "*" if (valid and not math.isnan(nmi) and nmi == max_nmi) else " "
            h_mark = "*" if (valid and not math.isnan(homog) and homog == max_homog) else " "
            c_mark = "*" if (valid and not math.isnan(compl) and compl == max_compl) else " "
            f.write(
                f"{r['name']:<{col_name}} {r['n_topics']:>{col_topics}} {r['n_outliers']:>{col_outliers}} "
                f"{purity:>{col_purity - 1}.4f}{p_mark} {nmi:>{col_nmi - 1}.4f}{n_mark} "
                f"{homog:>{col_homog - 1}.4f}{h_mark} {compl:>{col_compl - 1}.4f}{c_mark}\n"
            )
        f.write("(* = best in column)\n")
        f.write("\n")

        # Per-variation detail
        for r in all_results:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"VARIATION: {r['name']}\n")
            f.write("=" * 80 + "\n")
            if "error" in r:
                f.write(f"ERROR: {r['error']}\n\n")
                continue
            _nmi = r.get("nmi", 0)
            _homog = r.get("homogeneity", 0)
            _compl = r.get("completeness", 0)
            if isinstance(_nmi, float) and math.isnan(_nmi):
                _nmi = 0.0
            if isinstance(_homog, float) and math.isnan(_homog):
                _homog = 0.0
            if isinstance(_compl, float) and math.isnan(_compl):
                _compl = 0.0
            f.write(
                f"Topics: {r['n_topics']}  |  Outliers: {r['n_outliers']}  |  "
                f"Purity: {r['weighted_purity']:.4f}  |  NMI: {_nmi:.4f}\n"
            )
            f.write(
                f"  Homogeneity: {_homog:.4f}  |  Completeness: {_compl:.4f}\n"
            )
            f.write("\n")

            for info in r["topic_infos"]:
                f.write(f"  --- Topic {info['topic_id']} (n={info['size']}) ---\n")
                f.write(f"      Dominant emotion: {info['dominant_emotion']} (purity={info['purity']:.4f})\n")
                f.write(f"      Topic label: {info['top_words']}\n")
                f.write(f"      Emotion distribution: {info['emotion_dist']}\n")
                f.write("\n")
            f.write("\n")

    print(f"Results written to {out_path}")
