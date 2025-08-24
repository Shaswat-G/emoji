# -----------------------------------------------------------------------------
# Script: utc_throughput_analyzer_v2.py
# Purpose: Analyze emoji proposal processing metrics for the Unicode Technical Consortium (UTC), comparing pre- and post-2017 eras and accepted vs rejected proposals.
# Features: Loads datasets, computes metrics, performs statistical comparisons, generates reports and visualizations.
# Inputs: single_concept_accepted_proposals_v2.xlsx, rejected_proposal_dataset.xlsx, utc_register_with_llm_extraction.xlsx, emoji_proposal_email_matches.csv
# Outputs: Markdown reports, PNG visualizations, processed metrics CSV (in throughput_analysis_v2 folder)
# -----------------------------------------------------------------------------

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from utc_proposal_triangulator import (
    analyze_proposal_context,
    normalize_doc_num,
    safe_literal_eval,
    track_proposal_through_time,
)

warnings.filterwarnings("ignore")

CUT_OFF_DATE = datetime(2017, 1, 1)  # Date to classify proposals before/after


class UTCThroughputAnalyzerV2:
    def compare_eras_by_status(self, metrics_df):
        """
        Compare metrics for pre/post 2017, split by status (accepted/rejected/overall)
        Returns dict with keys: 'overall', 'accepted', 'rejected'
        """
        results = {}
        # Overall
        results["overall"] = self.compare_eras(metrics_df)
        # Accepted only
        results["accepted"] = self.compare_eras(
            metrics_df[metrics_df["status"] == "accepted"]
        )
        # Rejected only
        results["rejected"] = self.compare_eras(
            metrics_df[metrics_df["status"] == "rejected"]
        )
        return results

    def create_era_visualizations_by_status(
        self, metrics_df, era_comparisons, output_dir
    ):
        """
        Create visualizations for pre/post 2017, split by status (overall, accepted, rejected)
        """
        for key, label in zip(
            ["overall", "accepted", "rejected"],
            ["All Proposals", "Accepted Proposals", "Rejected Proposals"],
        ):
            df = (
                metrics_df
                if key == "overall"
                else metrics_df[metrics_df["status"] == key]
            )
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            df.boxplot(column="processing_days", by="era", ax=axes[0])
            axes[0].set_title(f"Processing Time by Era: {label}")
            axes[0].set_ylabel("Processing Days")
            axes[0].set_xlabel("Era")
            df.boxplot(column="reference_count", by="era", ax=axes[1])
            axes[1].set_title(f"Reference Count by Era: {label}")
            axes[1].set_ylabel("Reference Count")
            axes[1].set_xlabel("Era")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"era_comparison_boxplots_{key}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        print(f"Era visualizations by status saved to {output_dir}")

    def generate_era_report_by_status(self, era_comparisons, output_dir):
        """
        Generate Markdown report for pre/post 2017, split by status (overall, accepted, rejected)
        """
        report_path = os.path.join(output_dir, "era_comparison_by_status_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(
                "# Era Comparison: Overall, Accepted, and Rejected Proposals (V2)\n\n"
            )
            f.write(
                f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            )
            for key, label in zip(
                ["overall", "accepted", "rejected"],
                ["All Proposals", "Accepted Proposals", "Rejected Proposals"],
            ):
                comparison = era_comparisons[key]
                f.write(f"## {label}\n\n")
                f.write("**Sample Size:**\n")
                f.write(
                    f"- Pre-2017: {comparison['overview']['pre_2017_count']} proposals\n"
                )
                f.write(
                    f"- Post-2017: {comparison['overview']['post_2017_count']} proposals\n"
                )
                f.write(
                    f"- Total: {comparison['overview']['total_proposals']} proposals\n\n"
                )

                f.write("### Detailed Metrics Comparison\n\n")
                f.write(
                    "| Metric | Pre-2017 Mean | Post-2017 Mean | Change | P-Value | Significant |\n"
                )
                f.write(
                    "|--------|---------------|----------------|---------|---------|-------------|\n"
                )
                for metric, data in comparison.items():
                    if metric != "overview":
                        change = "✓ Improved" if data['improved'] else "✗ Not Improved"
                        significant = "Yes" if data['statistically_significant'] else "No"
                        f.write(
                            f"| {metric.replace('_', ' ').title()} | {data['pre_2017']['mean']:.2f} | {data['post_2017']['mean']:.2f} | {change} | {data['p_value']:.4f} | {significant} |\n"
                        )
                f.write("\n")

            f.write("*See accompanying visualizations for charts.*\n")
        print(f"Era comparison by status report generated: {report_path}")
        return report_path

    def __init__(self):
        """Initialize the analyzer with data paths"""
        self.base_path = os.getcwd()
        self.cutoff_date = CUT_OFF_DATE

        # Load data
        self._load_data()

    def _load_data(self):
        """Load all required datasets"""
        print("Loading datasets...")

        # Load accepted proposals (single concept v2, only 'normal')
        accepted_path = os.path.join(
            self.base_path, "single_concept_accepted_proposals_v2.xlsx"
        )
        accepted_df = pd.read_excel(accepted_path)
        # Filter for normal nature using v2 column
        self.accepted_proposals_df = accepted_df[
            accepted_df["nature_v2"] == "normal"
        ].copy()

        # Load rejected proposals (only 'normal')
        rejected_path = os.path.join(self.base_path, "rejected_proposal_dataset.xlsx")
        rejected_df = pd.read_excel(rejected_path)
        self.rejected_proposals_df = rejected_df[
            rejected_df["nature"] == "normal"
        ].copy()

        # Load UTC document registry
        utc_doc_reg_path = os.path.join(
            self.base_path, "utc_register_with_llm_extraction.xlsx"
        )
        self.utc_doc_reg_df = pd.read_excel(utc_doc_reg_path)

        columns_to_eval = [
            "doc_type",
            "extracted_doc_refs",
            "emoji_chars",
            "unicode_points",
            "emoji_keywords_found",
            "emoji_shortcodes",
            "people",
            "emoji_references",
            "entities",
        ]
        for col in columns_to_eval:
            if col in self.utc_doc_reg_df.columns:
                self.utc_doc_reg_df[col] = self.utc_doc_reg_df[col].apply(
                    safe_literal_eval
                )

        # Load email matches (only for accepted proposals)
        utc_email_path = os.path.join(
            self.base_path, "emoji_proposal_email_matches.csv"
        )
        self.email_match_df = pd.read_csv(utc_email_path)
        for col in columns_to_eval:
            if col in self.email_match_df.columns:
                self.email_match_df[col] = self.email_match_df[col].apply(
                    safe_literal_eval
                )

        # Convert date columns
        self._convert_dates()

        print(
            f"Loaded {len(self.accepted_proposals_df)} accepted proposals (normal, v2)"
        )
        print(f"Loaded {len(self.rejected_proposals_df)} rejected proposals (normal)")
        print(f"Loaded {len(self.utc_doc_reg_df)} UTC documents")
        print(f"Loaded {len(self.email_match_df)} email matches (accepted only)")

    def _convert_dates(self):
        """Convert date columns to datetime objects"""
        # Convert UTC document dates
        if "date" in self.utc_doc_reg_df.columns:
            self.utc_doc_reg_df["date"] = pd.to_datetime(
                self.utc_doc_reg_df["date"], errors="coerce"
            )

        # Convert email dates
        if "date" in self.email_match_df.columns:
            self.email_match_df["date"] = pd.to_datetime(
                self.email_match_df["date"], errors="coerce"
            )

        # Convert v2 acceptance dates
        if "acceptance_date_v2" in self.accepted_proposals_df.columns:
            self.accepted_proposals_df["acceptance_date_v2"] = pd.to_datetime(
                self.accepted_proposals_df["acceptance_date_v2"], errors="coerce"
            )

    def analyze_proposal_processing(self, proposal_id, start_date, end_date):
        """
        Analyze metrics for a proposal based strictly on UTC document references between precomputed first and last date.
        Returns:
            dict: Processing metrics including reference count, dormancy, people/entities/emoji, etc.
        """
        # Filter UTC document register for this proposal and date range
        timeline = track_proposal_through_time(proposal_id, self.utc_doc_reg_df)
        if timeline.empty:
            return None
        timeline = timeline[
            (timeline["date"] >= pd.to_datetime(start_date))
            & (timeline["date"] <= pd.to_datetime(end_date))
        ].copy()

        metrics = {
            "proposal_id": proposal_id,
            "reference_count": len(timeline),
        }

        # Dormancy analysis
        if not timeline.empty and "date" in timeline.columns:
            timeline_dates = timeline["date"].dropna().sort_values()
            if len(timeline_dates) > 1:
                gaps = timeline_dates.diff().dropna()
                metrics["max_dormancy_days"] = gaps.max().days if not gaps.empty else 0
            else:
                metrics["max_dormancy_days"] = 0

        # People and entity metrics
        context = analyze_proposal_context(timeline)
        metrics["unique_people"] = len(context["people_involved"])
        metrics["unique_entities"] = len(context["entities_involved"])
        metrics["unique_emoji"] = len(context["emoji_mentioned"])

        return metrics

    def classify_proposals_by_era(self):
        """
        Classify proposals into pre-2017 and post-2017 based on submission date.
        For each proposal, analyze metrics strictly based on UTC document references between precomputed first and last date.
        """
        proposal_metrics = []

        print("Analyzing accepted proposals (normal, v2)...")
        for _, row in tqdm(
            self.accepted_proposals_df.iterrows(), total=len(self.accepted_proposals_df)
        ):
            proposal_id = normalize_doc_num(
                row["proposal_doc_num"] if "proposal_doc_num" in row else row["doc_num"]
            )
            first_date = row["date"] if "date" in row else None
            # Use v2 acceptance date
            last_date = (
                row["acceptance_date_v2"] if "acceptance_date_v2" in row else None
            )
            if pd.notnull(first_date) and pd.notnull(last_date):
                metrics = self.analyze_proposal_processing(
                    proposal_id, first_date, last_date
                )
                if metrics:
                    metrics.update(
                        {
                            "status": "accepted",
                            "first_date": pd.to_datetime(first_date),
                            "last_date": pd.to_datetime(last_date),
                            "era": (
                                "pre_2017"
                                if pd.to_datetime(first_date) < self.cutoff_date
                                else "post_2017"
                            ),
                            # Use v2 processing time if available, otherwise calculate
                            "processing_days": (
                                row["processing_time_v2"]
                                if "processing_time_v2" in row
                                and pd.notnull(row["processing_time_v2"])
                                else (
                                    pd.to_datetime(last_date)
                                    - pd.to_datetime(first_date)
                                ).days
                            ),
                        }
                    )
                    proposal_metrics.append(metrics)

        print("Analyzing rejected proposals (normal)...")
        for _, row in tqdm(
            self.rejected_proposals_df.iterrows(), total=len(self.rejected_proposals_df)
        ):
            proposal_id = normalize_doc_num(row["doc_num"])
            first_date = row["date"] if "date" in row else None
            last_date = row["rejection_date"] if "rejection_date" in row else None
            if pd.notnull(first_date) and pd.notnull(last_date):
                metrics = self.analyze_proposal_processing(
                    proposal_id, first_date, last_date
                )
                if metrics:
                    metrics.update(
                        {
                            "status": "rejected",
                            "first_date": pd.to_datetime(first_date),
                            "last_date": pd.to_datetime(last_date),
                            "era": (
                                "pre_2017"
                                if pd.to_datetime(first_date) < self.cutoff_date
                                else "post_2017"
                            ),
                            "processing_days": (
                                pd.to_datetime(last_date) - pd.to_datetime(first_date)
                            ).days,
                        }
                    )
                    proposal_metrics.append(metrics)

        return pd.DataFrame(proposal_metrics)

    def compare_status(self, metrics_df):
        """
        Compare metrics for accepted vs rejected proposals (overall, no era split)
        """
        print("\nComparing accepted vs rejected proposals...")
        accepted = metrics_df[metrics_df["status"] == "accepted"]
        rejected = metrics_df[metrics_df["status"] == "rejected"]

        comparison_results = {
            "overview": {
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "total_proposals": len(metrics_df),
            }
        }

        numeric_metrics = [
            "processing_days",
            "reference_count",
            "max_dormancy_days",
            "unique_people",
            "unique_entities",
        ]

        for metric in numeric_metrics:
            if metric in accepted.columns and metric in rejected.columns:
                acc_vals = accepted[metric].dropna()
                rej_vals = rejected[metric].dropna()

                if len(acc_vals) > 0 and len(rej_vals) > 0:
                    stat, p_val = stats.mannwhitneyu(
                        acc_vals, rej_vals, alternative="two-sided"
                    )
                    comparison_results[metric] = {
                        "accepted": {
                            "mean": acc_vals.mean(),
                            "std": acc_vals.std(),
                            "median": acc_vals.median(),
                        },
                        "rejected": {
                            "mean": rej_vals.mean(),
                            "std": rej_vals.std(),
                            "median": rej_vals.median(),
                        },
                        "p_value": p_val,
                        "statistically_significant": p_val < 0.05,
                    }

        return comparison_results

    def create_status_visualizations(self, metrics_df, status_comparison, output_dir):
        """
        Create visualizations for accepted vs rejected proposals (overall)
        """
        plt.figure(figsize=(12, 8))
        metrics_df.boxplot(column="processing_days", by="status")
        plt.title("Processing Time: Accepted vs Rejected Proposals (V2)")
        plt.ylabel("Processing Days")
        plt.xlabel("Proposal Status")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "accepted_vs_rejected_processing_time.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(12, 8))
        metrics_df.boxplot(column="reference_count", by="status")
        plt.title("Reference Count: Accepted vs Rejected Proposals (V2)")
        plt.ylabel("Reference Count")
        plt.xlabel("Proposal Status")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "accepted_vs_rejected_reference_count.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Accepted vs Rejected visualizations saved to {output_dir}")

    def generate_status_report(self, metrics_df, status_comparison, output_dir):
        """
        Generate Markdown report for accepted vs rejected proposals (overall)
        """
        report_path = os.path.join(output_dir, "accepted_vs_rejected_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Accepted vs Rejected Proposals Analysis (V2)\n\n")
            f.write(
                f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            )

            overview = status_comparison["overview"]
            f.write("## Overview\n\n")
            f.write(f"- **Accepted Proposals:** {overview['accepted_count']}\n")
            f.write(f"- **Rejected Proposals:** {overview['rejected_count']}\n")
            f.write(f"- **Total Proposals:** {overview['total_proposals']}\n\n")

            f.write("## Statistical Comparisons\n\n")
            f.write(
                "| Metric | Accepted Mean | Rejected Mean | Change | P-Value | Significant |\n"
            )
            f.write(
                "|--------|---------------|---------------|--------|---------|-------------|\n"
            )
            for metric, data in status_comparison.items():
                if metric != "overview":
                    direction = "Accepted Lower" if data['accepted']['mean'] < data['rejected']['mean'] else "Rejected Lower"
                    significant = "Yes" if data['statistically_significant'] else "No"
                    f.write(
                        f"| {metric.replace('_', ' ').title()} | {data['accepted']['mean']:.2f} | {data['rejected']['mean']:.2f} | {direction} | {data['p_value']:.4f} | {significant} |\n"
                    )

        print(f"Accepted vs Rejected report generated: {report_path}")
        return report_path

    def compare_eras(self, metrics_df):
        """
        Compare processing metrics between pre-2017 and post-2017 eras
        """
        print("\nComparing processing efficiency between eras...")

        pre_2017 = metrics_df[metrics_df["era"] == "pre_2017"]
        post_2017 = metrics_df[metrics_df["era"] == "post_2017"]

        comparison_results = {
            "overview": {
                "pre_2017_count": len(pre_2017),
                "post_2017_count": len(post_2017),
                "total_proposals": len(metrics_df),
            }
        }

        # Metrics to compare
        numeric_metrics = [
            "processing_days",
            "reference_count",
            "max_dormancy_days",
            "unique_people",
            "unique_entities",
        ]

        for metric in numeric_metrics:
            if metric in pre_2017.columns and metric in post_2017.columns:
                pre_vals = pre_2017[metric].dropna()
                post_vals = post_2017[metric].dropna()

                if len(pre_vals) > 0 and len(post_vals) > 0:
                    stat, p_val = stats.mannwhitneyu(
                        pre_vals, post_vals, alternative="two-sided"
                    )
                    comparison_results[metric] = {
                        "pre_2017": {
                            "mean": pre_vals.mean(),
                            "std": pre_vals.std(),
                            "median": pre_vals.median(),
                        },
                        "post_2017": {
                            "mean": post_vals.mean(),
                            "std": post_vals.std(),
                            "median": post_vals.median(),
                        },
                        "p_value": p_val,
                        "improved": self._metric_improved(
                            metric, pre_vals.mean(), post_vals.mean()
                        ),
                        "statistically_significant": p_val < 0.05,
                    }

        return comparison_results

    def _metric_improved(self, metric, pre_value, post_value):
        """Determine if a metric improved (lower is better for some metrics)"""
        improvement_metrics = [
            "processing_days",
            "max_dormancy_days",
        ]  # Lower is better
        if metric in improvement_metrics:
            return post_value < pre_value
        else:
            return post_value > pre_value

    def create_visualizations(self, metrics_df, comparison_results, output_dir):
        """
        Create comprehensive visualizations of the throughput analysis
        """
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Processing Time Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Processing days by era
        metrics_df.boxplot(column="processing_days", by="era", ax=axes[0, 0])
        axes[0, 0].set_title("Processing Time by Era (V2)")
        axes[0, 0].set_ylabel("Processing Days")
        axes[0, 0].set_xlabel("Era")

        # Reference count by era
        metrics_df.boxplot(column="reference_count", by="era", ax=axes[0, 1])
        axes[0, 1].set_title("Reference Count by Era (V2)")
        axes[0, 1].set_ylabel("Number of References")
        axes[0, 1].set_xlabel("Era")

        # Dormancy by era
        metrics_df.boxplot(column="max_dormancy_days", by="era", ax=axes[1, 0])
        axes[1, 0].set_title("Max Dormancy by Era (V2)")
        axes[1, 0].set_ylabel("Max Dormancy Days")
        axes[1, 0].set_xlabel("Era")

        # People involvement by era
        metrics_df.boxplot(column="unique_people", by="era", ax=axes[1, 1])
        axes[1, 1].set_title("People Involvement by Era (V2)")
        axes[1, 1].set_ylabel("Unique People Count")
        axes[1, 1].set_xlabel("Era")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "era_comparison_boxplots.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Timeline of proposal submissions
        plt.figure(figsize=(14, 8))

        # Group by year and era
        yearly_counts = (
            metrics_df.groupby([metrics_df["first_date"].dt.year, "era"])
            .size()
            .unstack(fill_value=0)
        )
        yearly_counts.plot(kind="bar", stacked=True, alpha=0.8)

        plt.axvline(
            x=2017, color="red", linestyle="--", alpha=0.7, label="2017 Process Change"
        )
        plt.title("Emoji Proposal Submissions Over Time (V2)")
        plt.xlabel("Year")
        plt.ylabel("Number of Proposals")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "proposals_timeline.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Visualizations saved to {output_dir}")

    def analyze_proposal_volume_patterns(self, metrics_df):
        """
        Analyze proposal volume patterns by year and month for context
        Returns dict with yearly counts, monthly rates, and composition analysis
        """
        print("Analyzing proposal volume patterns...")

        # 1. Yearly proposal counts by status and era
        yearly_analysis = {}

        # Overall yearly counts
        yearly_counts = metrics_df.groupby([
            metrics_df["first_date"].dt.year,
            "status"
        ]).size().unstack(fill_value=0)

        yearly_analysis["yearly_counts_by_status"] = yearly_counts

        # Yearly counts by era
        yearly_era_counts = metrics_df.groupby([
            metrics_df["first_date"].dt.year,
            "era",
            "status"
        ]).size().unstack(fill_value=0)

        yearly_analysis["yearly_counts_by_era_status"] = yearly_era_counts

        # 2. Monthly incoming rates
        monthly_analysis = {}

        # Create year-month column
        metrics_df["year_month"] = metrics_df["first_date"].dt.to_period("M")

        # Monthly counts by status
        monthly_counts = metrics_df.groupby(["year_month", "status"]).size().unstack(fill_value=0)
        monthly_analysis["monthly_counts_by_status"] = monthly_counts

        # Calculate monthly rates by era
        pre_2017_months = metrics_df[metrics_df["era"] == "pre_2017"]["year_month"].nunique()
        post_2017_months = metrics_df[metrics_df["era"] == "post_2017"]["year_month"].nunique()

        pre_2017_proposals = len(metrics_df[metrics_df["era"] == "pre_2017"])
        post_2017_proposals = len(metrics_df[metrics_df["era"] == "post_2017"])

        monthly_analysis["rates"] = {
            "pre_2017": {
                "total_months": pre_2017_months,
                "total_proposals": pre_2017_proposals,
                "proposals_per_month": pre_2017_proposals / pre_2017_months if pre_2017_months > 0 else 0
            },
            "post_2017": {
                "total_months": post_2017_months,
                "total_proposals": post_2017_proposals,
                "proposals_per_month": post_2017_proposals / post_2017_months if post_2017_months > 0 else 0
            }
        }

        # Monthly rates by status and era
        for era in ["pre_2017", "post_2017"]:
            era_data = metrics_df[metrics_df["era"] == era]
            era_months = era_data["year_month"].nunique()

            for status in ["accepted", "rejected"]:
                status_count = len(era_data[era_data["status"] == status])
                monthly_analysis["rates"][era][f"{status}_per_month"] = status_count / era_months if era_months > 0 else 0

        # 3. Composition analysis (acceptance rates by era)
        composition_analysis = {}

        for era in ["pre_2017", "post_2017"]:
            era_data = metrics_df[metrics_df["era"] == era]
            total = len(era_data)
            accepted = len(era_data[era_data["status"] == "accepted"])
            rejected = len(era_data[era_data["status"] == "rejected"])

            composition_analysis[era] = {
                "total": total,
                "accepted": accepted,
                "rejected": rejected,
                "acceptance_rate": (accepted / total * 100) if total > 0 else 0,
                "rejection_rate": (rejected / total * 100) if total > 0 else 0
            }

        return {
            "yearly": yearly_analysis,
            "monthly": monthly_analysis,
            "composition": composition_analysis
        }

    def create_volume_visualizations(self, metrics_df, volume_patterns, output_dir):
        """
        Create visualizations for proposal volume patterns
        """
        # 1. Yearly proposal counts stacked by status
        plt.figure(figsize=(14, 8))
        yearly_counts = volume_patterns["yearly"]["yearly_counts_by_status"]
        yearly_counts.plot(kind="bar", stacked=True, alpha=0.8, figsize=(14, 8))
        plt.axvline(x=yearly_counts.index.get_loc(2017) if 2017 in yearly_counts.index else len(yearly_counts)//2,
                   color="red", linestyle="--", alpha=0.7, label="2017 Process Change")
        plt.title("Annual Proposal Volumes by Status (V2)")
        plt.xlabel("Year")
        plt.ylabel("Number of Proposals")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "annual_proposal_volumes.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Monthly proposal rates comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Monthly rates by era
        rates = volume_patterns["monthly"]["rates"]
        eras = ["pre_2017", "post_2017"]
        proposals_per_month = [rates[era]["proposals_per_month"] for era in eras]
        accepted_per_month = [rates[era]["accepted_per_month"] for era in eras]
        rejected_per_month = [rates[era]["rejected_per_month"] for era in eras]

        x = range(len(eras))
        width = 0.25

        ax1.bar([i - width for i in x], proposals_per_month, width, label="Total", alpha=0.8)
        ax1.bar(x, accepted_per_month, width, label="Accepted", alpha=0.8)
        ax1.bar([i + width for i in x], rejected_per_month, width, label="Rejected", alpha=0.8)

        ax1.set_xlabel("Era")
        ax1.set_ylabel("Proposals per Month")
        ax1.set_title("Monthly Proposal Rates by Era (V2)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["Pre-2017", "Post-2017"])
        ax1.legend()

        # Acceptance rates by era
        composition = volume_patterns["composition"]
        acceptance_rates = [composition[era]["acceptance_rate"] for era in eras]
        rejection_rates = [composition[era]["rejection_rate"] for era in eras]

        ax2.bar(x, acceptance_rates, width*2, label="Acceptance Rate", alpha=0.8, color="green")
        ax2.bar(x, rejection_rates, width*2, bottom=acceptance_rates, label="Rejection Rate", alpha=0.8, color="red")

        ax2.set_xlabel("Era")
        ax2.set_ylabel("Percentage")
        ax2.set_title("Proposal Outcome Composition by Era (V2)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["Pre-2017", "Post-2017"])
        ax2.legend()
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "proposal_volume_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Volume analysis visualizations saved to {output_dir}")

    def generate_report(self, metrics_df, comparison_results, output_dir):
        """
        Generate comprehensive Markdown report including volume analysis
        """
        # First analyze volume patterns
        volume_patterns = self.analyze_proposal_volume_patterns(metrics_df)

        report_path = os.path.join(output_dir, "throughput_analysis_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# UTC Throughput Analysis Report (V2)\n\n")
            f.write(
                f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            )

            f.write("## Analysis Overview\n\n")
            f.write(
                "This analysis compares emoji proposal processing efficiency between:\n"
            )
            f.write("- **Pre-2017 Era:** Before process improvements\n")
            f.write("- **Post-2017 Era:** After process improvements\n\n")
            f.write("**Data Sources (V2):**\n")
            f.write(
                "- Accepted proposals: `single_concept_accepted_proposals_v2.xlsx`\n"
            )
            f.write("- Uses `acceptance_date_v2` and `processing_time_v2` columns\n")
            f.write("- Filtered for `nature_v2 == 'normal'` proposals only\n\n")

            overview = comparison_results["overview"]
            f.write("**Sample Size:**\n")
            f.write(f"- Pre-2017: {overview['pre_2017_count']} proposals\n")
            f.write(f"- Post-2017: {overview['post_2017_count']} proposals\n")
            f.write(f"- Total: {overview['total_proposals']} proposals\n\n")

            # Volume Analysis Section
            f.write("## Proposal Volume Context\n\n")
            f.write("Understanding proposal volumes is crucial for interpreting efficiency metrics, as changes in processing efficiency might be influenced by changes in proposal volume or composition over time.\n\n")

            # Monthly rates table
            rates = volume_patterns["monthly"]["rates"]
            f.write("### Monthly Proposal Rates\n\n")
            f.write("| Era | Total Proposals/Month | Accepted/Month | Rejected/Month | Active Months |\n")
            f.write("|-----|---------------------|----------------|----------------|---------------|\n")
            for era in ["pre_2017", "post_2017"]:
                era_label = "Pre-2017" if era == "pre_2017" else "Post-2017"
                f.write(f"| {era_label} | {rates[era]['proposals_per_month']:.2f} | ")
                f.write(f"{rates[era]['accepted_per_month']:.2f} | ")
                f.write(f"{rates[era]['rejected_per_month']:.2f} | ")
                f.write(f"{rates[era]['total_months']} |\n")
            f.write("\n")

            # Composition analysis table
            composition = volume_patterns["composition"]
            f.write("### Proposal Outcome Composition\n\n")
            f.write("| Era | Total Proposals | Accepted | Rejected | Acceptance Rate | Rejection Rate |\n")
            f.write("|-----|----------------|----------|----------|-----------------|----------------|\n")
            for era in ["pre_2017", "post_2017"]:
                era_label = "Pre-2017" if era == "pre_2017" else "Post-2017"
                comp = composition[era]
                f.write(f"| {era_label} | {comp['total']} | {comp['accepted']} | ")
                f.write(f"{comp['rejected']} | {comp['acceptance_rate']:.1f}% | ")
                f.write(f"{comp['rejection_rate']:.1f}% |\n")
            f.write("\n")

            # Yearly counts summary
            yearly_counts = volume_patterns["yearly"]["yearly_counts_by_status"]
            f.write("### Annual Proposal Volumes (Top 10 Years)\n\n")
            if not yearly_counts.empty:
                yearly_totals = yearly_counts.sum(axis=1).sort_values(ascending=False).head(10)
                f.write("| Year | Total | Accepted | Rejected |\n")
                f.write("|------|-------|----------|----------|\n")
                for year in yearly_totals.index:
                    accepted = yearly_counts.loc[year, 'accepted'] if 'accepted' in yearly_counts.columns else 0
                    rejected = yearly_counts.loc[year, 'rejected'] if 'rejected' in yearly_counts.columns else 0
                    f.write(f"| {year} | {yearly_totals[year]} | {accepted} | {rejected} |\n")
                f.write("\n")

            f.write("**Key Volume Insights:**\n")
            rate_change = ((rates["post_2017"]["proposals_per_month"] - rates["pre_2017"]["proposals_per_month"]) / rates["pre_2017"]["proposals_per_month"] * 100) if rates["pre_2017"]["proposals_per_month"] > 0 else 0
            acceptance_change = composition["post_2017"]["acceptance_rate"] - composition["pre_2017"]["acceptance_rate"]

            f.write(f"- Monthly proposal rate {'increased' if rate_change > 0 else 'decreased'} by {abs(rate_change):.1f}% post-2017\n")
            f.write(f"- Acceptance rate {'increased' if acceptance_change > 0 else 'decreased'} by {abs(acceptance_change):.1f} percentage points post-2017\n")
            f.write("- This context is important when interpreting processing efficiency metrics below\n\n")

            f.write("## Processing Efficiency Metrics\n\n")
            f.write(
                "| Metric | Pre-2017 Mean ± SD | Post-2017 Mean ± SD | Improved | P-Value | Significant |\n"
            )
            f.write(
                "|--------|-------------------|---------------------|----------|---------|-------------|\n"
            )
            for metric, data in comparison_results.items():
                if metric != "overview":
                    change = "✓ Yes" if data['improved'] else "✗ No"
                    significant = "✓ Yes" if data['statistically_significant'] else "✗ No"
                    f.write(
                        f"| {metric.replace('_', ' ').title()} | {data['pre_2017']['mean']:.2f} ± {data['pre_2017']['std']:.2f} | {data['post_2017']['mean']:.2f} ± {data['post_2017']['std']:.2f} | {change} | {data['p_value']:.4f} | {significant} |\n"
                    )
            f.write("\n")

            f.write("## Methodology\n\n")
            f.write("- **Statistical tests:** Mann-Whitney U (non-parametric)\n")
            f.write("- **Significance threshold:** p < 0.05\n")
            f.write(
                "- **Improvement definition:** Lower processing time and dormancy = better; Higher engagement metrics = better\n"
            )
            f.write(
                "- **Era classification:** Based on proposal submission date vs 2017-01-01\n"
            )
            f.write("- **Volume normalization:** Raw metrics shown; volume context provided for interpretation\n\n")

            f.write("## Interpretation Notes\n\n")
            f.write("- **Volume effects:** Changes in efficiency metrics should be interpreted alongside volume changes\n")
            f.write("- **Composition effects:** Changes in acceptance rates may influence perceived processing efficiency\n")
            f.write("- **Temporal effects:** Longer time series in pre-2017 era may affect metric distributions\n")
            f.write("- **Process maturity:** Post-2017 improvements may reflect both procedural changes and institutional learning\n\n")

        print(f"Report generated: {report_path}")
        return report_path

    def run_full_analysis(self):
        """
        Run the complete throughput analysis
        """
        print("Starting UTC Throughput Analysis (V2)...")

        # Analyze all proposals
        metrics_df = self.classify_proposals_by_era()

        if metrics_df.empty:
            print("❌ No proposals found with required timeline data!")
            return None

        print(f"Analyzed {len(metrics_df)} proposals with timeline data")

        # Compare accepted vs rejected proposals (overall)
        status_comparison = self.compare_status(metrics_df)

        # Create output directory (V2)
        output_dir = os.path.join(self.base_path, "throughput_analysis_v2")
        os.makedirs(output_dir, exist_ok=True)

        # Generate status visualizations and report
        self.create_status_visualizations(metrics_df, status_comparison, output_dir)
        status_report_path = self.generate_status_report(
            metrics_df, status_comparison, output_dir
        )

        # Compare eras (pre/post 2017) for overall, accepted, rejected
        era_comparisons = self.compare_eras_by_status(metrics_df)

        # Analyze volume patterns for context
        volume_patterns = self.analyze_proposal_volume_patterns(metrics_df)

        # Generate visualizations and report for era analysis by status
        self.create_era_visualizations_by_status(
            metrics_df, era_comparisons, output_dir
        )

        # Generate volume analysis visualizations
        self.create_volume_visualizations(metrics_df, volume_patterns, output_dir)

        era_report_path = self.generate_era_report_by_status(
            era_comparisons, output_dir
        )

        # Generate main throughput analysis report (includes volume context)
        main_report_path = self.generate_report(
            metrics_df, era_comparisons["overall"], output_dir
        )

        # Save processed data
        data_path = os.path.join(output_dir, "proposal_metrics_2017_v2.csv")
        metrics_df.to_csv(data_path, index=False)

        print("\n✅ Analysis complete!")
        print(f"📊 Status Report: {status_report_path}")
        print(f"📊 Era Comparison Report: {era_report_path}")
        print(f"� Main Throughput Report (with volume context): {main_report_path}")
        print(f"�📈 Visualizations: {output_dir}")
        print(f"📋 Data: {data_path}")

        return {
            "metrics_df": metrics_df,
            "status_comparison": status_comparison,
            "era_comparisons": era_comparisons,
            "output_dir": output_dir,
        }


def main():
    """Main execution function"""
    analyzer = UTCThroughputAnalyzerV2()
    results = analyzer.run_full_analysis()
    return results


if __name__ == "__main__":
    main()
