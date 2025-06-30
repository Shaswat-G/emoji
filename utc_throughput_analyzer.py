# -----------------------------------------------------------------------------
# Script: utc_throughput_analyzer.py
# Summary: Analyzes UTC proposal processing efficiency before vs after 2017
#          to evaluate the impact of process standardization changes.
# Inputs:  emoji_proposal_table.csv, utc_register_with_llm_extraction.xlsx,
#          emoji_proposal_email_matches.csv, people_and_body_proposals.xlsx
# Outputs: Comparative analysis reports, visualizations, statistical tests
# Context: Evaluates whether UTC's 2017 process changes improved efficiency
# -----------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import ast
import re
from collections import defaultdict, Counter
from scipy import stats
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Import functions from the triangulator
from utc_proposal_triangulator import (
    normalize_doc_num,
    track_proposal_through_time,
    analyze_proposal_context,
    safe_literal_eval,
)


class UTCThroughputAnalyzer:
    def __init__(self):
        """Initialize the analyzer with data paths"""
        self.base_path = os.getcwd()
        self.cutoff_date = datetime(2017, 1, 1)

        # Load data
        self._load_data()

    def _load_data(self):
        """Load all required datasets"""
        print("Loading datasets...")

        # Load emoji proposals
        emoji_proposal_path = os.path.join(self.base_path, "emoji_proposal_table.csv")
        self.emoji_proposal_df = pd.read_csv(emoji_proposal_path, dtype=str)

        # Load people and body diversity proposals
        people_body_path = os.path.join(
            self.base_path, "people_and_body_proposals.xlsx"
        )
        people_body_df = pd.read_excel(people_body_path)
        self.people_body_proposals = set(people_body_df["proposals"].astype(str))
        print(
            f"Loaded {len(self.people_body_proposals)} people/body diversity proposals"
        )

        # Load UTC document registry
        utc_doc_reg_path = os.path.join(
            self.base_path, "utc_register_with_llm_extraction.xlsx"
        )
        self.utc_doc_reg_df = pd.read_excel(utc_doc_reg_path)

        # Parse Python objects in specified columns
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

        # Load email matches
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

        print(f"Loaded {len(self.emoji_proposal_df)} proposals")
        print(f"Loaded {len(self.utc_doc_reg_df)} UTC documents")
        print(f"Loaded {len(self.email_match_df)} email matches")

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

    def analyze_proposal_processing(self, proposal_id):
        """
        Analyze processing metrics for a single proposal

        Returns:
            dict: Processing metrics including timeline, velocity, etc.
        """
        # Get proposal timeline from UTC documents
        timeline = track_proposal_through_time(proposal_id, self.utc_doc_reg_df)

        if timeline.empty:
            return None

        # Get email matches for this proposal
        email_matches = self.email_match_df[
            self.email_match_df["proposal_doc_num"].apply(
                lambda x: (
                    normalize_doc_num(x) == normalize_doc_num(proposal_id)
                    if isinstance(x, str)
                    else False
                )
            )
        ]

        # Calculate metrics
        metrics = {
            "proposal_id": proposal_id,
            "reference_count": len(timeline) - 1,  # Exclude original proposal
            "email_count": len(email_matches),
        }

        # Date-based metrics
        if not timeline.empty and "date" in timeline.columns:
            timeline_dates = timeline["date"].dropna()
            if len(timeline_dates) > 0:
                metrics["first_date"] = timeline_dates.min()
                metrics["last_date"] = timeline_dates.max()
                metrics["processing_days"] = (
                    metrics["last_date"] - metrics["first_date"]
                ).days
                metrics["processing_years"] = metrics["processing_days"] / 365.25

                # Velocity metrics
                if metrics["processing_days"] > 0:
                    metrics["velocity_per_year"] = metrics["reference_count"] / max(
                        metrics["processing_years"], 1 / 365.25
                    )
                    metrics["velocity_per_month"] = metrics["reference_count"] / max(
                        metrics["processing_days"] / 30.44, 1 / 30.44
                    )
                else:
                    metrics["velocity_per_year"] = 0
                    metrics["velocity_per_month"] = 0

                # Dormancy analysis
                if len(timeline_dates) > 1:
                    sorted_dates = timeline_dates.sort_values()
                    gaps = [
                        (sorted_dates.iloc[i + 1] - sorted_dates.iloc[i]).days
                        for i in range(len(sorted_dates) - 1)
                    ]
                    metrics["max_dormancy_days"] = max(gaps) if gaps else 0
                    metrics["avg_gap_days"] = np.mean(gaps) if gaps else 0
                else:
                    metrics["max_dormancy_days"] = 0
                    metrics["avg_gap_days"] = 0

        # People and entity metrics
        context = analyze_proposal_context(timeline)
        metrics["unique_people"] = len(context["people_involved"])
        metrics["unique_entities"] = len(context["entities_involved"])
        metrics["unique_emoji"] = len(context["emoji_mentioned"])

        # Email metrics
        if not email_matches.empty and "date" in email_matches.columns:
            email_dates = email_matches["date"].dropna()
            if len(email_dates) > 0:
                metrics["first_email_date"] = email_dates.min()
                metrics["last_email_date"] = email_dates.max()
                metrics["email_span_days"] = (
                    metrics["last_email_date"] - metrics["first_email_date"]
                ).days

                # Email confidence
                if "confidence_score" in email_matches.columns:
                    metrics["avg_email_confidence"] = email_matches[
                        "confidence_score"
                    ].mean()
                    metrics["max_email_confidence"] = email_matches[
                        "confidence_score"
                    ].max()

        return metrics

    def classify_proposals_by_era(self):
        """
        Classify proposals into pre-2017 and post-2017 based on submission date
        """
        proposal_metrics = []

        print("Analyzing individual proposals...")
        for _, row in tqdm(
            self.emoji_proposal_df.iterrows(), total=len(self.emoji_proposal_df)
        ):
            proposal_id = normalize_doc_num(row["doc_num"])
            metrics = self.analyze_proposal_processing(proposal_id)

            if metrics and metrics.get("first_date"):
                # Classify by era
                metrics["era"] = (
                    "pre_2017"
                    if metrics["first_date"] < self.cutoff_date
                    else "post_2017"
                )
                metrics["proposal_title"] = row.get("proposal_title", "Unknown")
                metrics["proposer"] = row.get("proposer", "Unknown")

                # Check if it's a people/body diversity proposal
                metrics["is_people_body"] = self._is_people_body_proposal(proposal_id)

                proposal_metrics.append(metrics)

        return pd.DataFrame(proposal_metrics)

    def _is_people_body_proposal(self, proposal_id):
        """
        Determine if a proposal is related to people/body diversity
        using the curated list from people_and_body_proposals.xlsx
        """
        # Normalize the proposal ID for matching
        normalized_id = normalize_doc_num(proposal_id)

        # Check if this proposal is in our curated list
        return normalized_id in self.people_body_proposals

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
            "velocity_per_year",
            "max_dormancy_days",
            "unique_people",
            "unique_entities",
            "email_count",
            "avg_email_confidence",
        ]

        for metric in numeric_metrics:
            if metric in metrics_df.columns:
                pre_values = pre_2017[metric].dropna()
                post_values = post_2017[metric].dropna()

                if len(pre_values) > 0 and len(post_values) > 0:
                    # Statistical comparison
                    stat, p_value = stats.mannwhitneyu(
                        pre_values, post_values, alternative="two-sided"
                    )

                    comparison_results[metric] = {
                        "pre_2017_mean": pre_values.mean(),
                        "pre_2017_median": pre_values.median(),
                        "pre_2017_std": pre_values.std(),
                        "post_2017_mean": post_values.mean(),
                        "post_2017_median": post_values.median(),
                        "post_2017_std": post_values.std(),
                        "mannwhitney_u_stat": stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "improvement": (
                            "post_2017"
                            if self._metric_improved(
                                metric, pre_values.mean(), post_values.mean()
                            )
                            else "pre_2017"
                        ),
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
            return post_value > pre_value  # Higher is better

    def analyze_people_body_proposals(self, metrics_df):
        """
        Specific analysis for people/body diversity proposals
        """
        people_body_df = metrics_df[metrics_df["is_people_body"] == True]
        other_df = metrics_df[metrics_df["is_people_body"] == False]

        print(f"\nAnalyzing {len(people_body_df)} people/body diversity proposals...")

        # Compare people/body proposals across eras
        pb_pre_2017 = people_body_df[people_body_df["era"] == "pre_2017"]
        pb_post_2017 = people_body_df[people_body_df["era"] == "post_2017"]

        people_body_results = {
            "overview": {
                "total_people_body": len(people_body_df),
                "pb_pre_2017": len(pb_pre_2017),
                "pb_post_2017": len(pb_post_2017),
                "pb_percentage": len(people_body_df) / len(metrics_df) * 100,
            }
        }

        # Processing efficiency for people/body proposals
        if len(pb_pre_2017) > 0 and len(pb_post_2017) > 0:
            for metric in ["processing_days", "reference_count", "velocity_per_year"]:
                if metric in people_body_df.columns:
                    pre_values = pb_pre_2017[metric].dropna()
                    post_values = pb_post_2017[metric].dropna()

                    if len(pre_values) > 0 and len(post_values) > 0:
                        people_body_results[metric] = {
                            "pre_2017_mean": pre_values.mean(),
                            "post_2017_mean": post_values.mean(),
                            "improvement": self._metric_improved(
                                metric, pre_values.mean(), post_values.mean()
                            ),
                        }

        return people_body_results

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
        axes[0, 0].set_title("Processing Time by Era")
        axes[0, 0].set_ylabel("Processing Days")
        axes[0, 0].set_xlabel("Era")

        # Reference count by era
        metrics_df.boxplot(column="reference_count", by="era", ax=axes[0, 1])
        axes[0, 1].set_title("Reference Count by Era")
        axes[0, 1].set_ylabel("Number of References")
        axes[0, 1].set_xlabel("Era")

        # Velocity by era
        metrics_df.boxplot(column="velocity_per_year", by="era", ax=axes[1, 0])
        axes[1, 0].set_title("Processing Velocity by Era")
        axes[1, 0].set_ylabel("References per Year")
        axes[1, 0].set_xlabel("Era")

        # People involvement by era
        metrics_df.boxplot(column="unique_people", by="era", ax=axes[1, 1])
        axes[1, 1].set_title("People Involvement by Era")
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
        plt.title("Emoji Proposal Submissions Over Time")
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

        # 3. People/Body proposals analysis
        if "is_people_body" in metrics_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Processing time for people/body vs others
            pb_data = [
                metrics_df[metrics_df["is_people_body"] == True][
                    "processing_days"
                ].dropna(),
                metrics_df[metrics_df["is_people_body"] == False][
                    "processing_days"
                ].dropna(),
            ]
            axes[0].boxplot(pb_data, labels=["People/Body", "Others"])
            axes[0].set_title("Processing Time: People/Body vs Other Proposals")
            axes[0].set_ylabel("Processing Days")

            # People/body proposals by era
            pb_by_era = (
                metrics_df.groupby(["era", "is_people_body"])
                .size()
                .unstack(fill_value=0)
            )
            pb_by_era.plot(kind="bar", ax=axes[1], alpha=0.8)
            axes[1].set_title("People/Body Proposals by Era")
            axes[1].set_ylabel("Number of Proposals")
            axes[1].set_xlabel("Era")
            axes[1].legend(["Other Proposals", "People/Body Proposals"])
            axes[1].tick_params(axis="x", rotation=0)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "people_body_analysis.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        print(f"Visualizations saved to {output_dir}")

    def generate_report(
        self, metrics_df, comparison_results, people_body_results, output_dir
    ):
        """
        Generate comprehensive Markdown report
        """
        report_path = os.path.join(output_dir, "throughput_analysis_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# UTC Emoji Proposal Throughput Analysis\n\n")
            f.write(
                f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            )

            f.write("## Executive Summary\n\n")
            f.write(
                "This analysis evaluates whether UTC's process standardization changes in 2017 "
            )
            f.write("improved the efficiency of emoji proposal processing.\n\n")

            # Overview statistics
            f.write("## Overview Statistics\n\n")
            f.write(
                f"- **Total Proposals Analyzed**: {comparison_results['overview']['total_proposals']}\n"
            )
            f.write(
                f"- **Pre-2017 Proposals**: {comparison_results['overview']['pre_2017_count']}\n"
            )
            f.write(
                f"- **Post-2017 Proposals**: {comparison_results['overview']['post_2017_count']}\n\n"
            )

            # Key findings
            f.write("## Key Findings\n\n")

            significant_improvements = []
            for metric, data in comparison_results.items():
                if isinstance(data, dict) and data.get("significant", False):
                    improvement_direction = (
                        "improved" if data["improvement"] == "post_2017" else "worsened"
                    )
                    significant_improvements.append(
                        f"- **{metric.replace('_', ' ').title()}**: {improvement_direction} significantly (p < 0.05)"
                    )

            if significant_improvements:
                f.write("### Statistically Significant Changes:\n")
                f.write("\n".join(significant_improvements))
                f.write("\n\n")

            # Detailed metrics comparison
            f.write("## Detailed Metrics Comparison\n\n")
            f.write(
                "| Metric | Pre-2017 Mean | Post-2017 Mean | Change | P-Value | Significant |\n"
            )
            f.write(
                "|--------|---------------|----------------|---------|---------|-------------|\n"
            )

            for metric, data in comparison_results.items():
                if isinstance(data, dict) and "pre_2017_mean" in data:
                    change = data["post_2017_mean"] - data["pre_2017_mean"]
                    change_pct = (
                        (change / data["pre_2017_mean"]) * 100
                        if data["pre_2017_mean"] != 0
                        else 0
                    )

                    f.write(
                        f"| {metric.replace('_', ' ').title()} | {data['pre_2017_mean']:.2f} | "
                    )
                    f.write(f"{data['post_2017_mean']:.2f} | {change_pct:+.1f}% | ")
                    f.write(
                        f"{data['p_value']:.3f} | {'Yes' if data['significant'] else 'No'} |\n"
                    )

            f.write("\n")

            # People/Body proposals section
            if people_body_results:
                f.write("## People & Body Diversity Proposals\n\n")
                f.write(
                    f"- **Total People/Body Proposals**: {people_body_results['overview']['total_people_body']}\n"
                )
                f.write(
                    f"- **Percentage of All Proposals**: {people_body_results['overview']['pb_percentage']:.1f}%\n"
                )
                f.write(
                    f"- **Pre-2017**: {people_body_results['overview']['pb_pre_2017']}\n"
                )
                f.write(
                    f"- **Post-2017**: {people_body_results['overview']['pb_post_2017']}\n\n"
                )

                for metric, data in people_body_results.items():
                    if isinstance(data, dict) and "pre_2017_mean" in data:
                        improvement = "improved" if data["improvement"] else "worsened"
                        f.write(
                            f"- **{metric.replace('_', ' ').title()}**: {improvement} from "
                        )
                        f.write(
                            f"{data['pre_2017_mean']:.2f} to {data['post_2017_mean']:.2f}\n"
                        )

            f.write("\n## Methodology\n\n")
            f.write("This analysis used the following approach:\n")
            f.write(
                "1. Classified proposals by submission date (before/after January 1, 2017)\n"
            )
            f.write("2. Calculated processing metrics for each proposal\n")
            f.write("3. Used Mann-Whitney U tests for statistical significance\n")
            f.write("4. Special analysis for people/body diversity proposals\n\n")

            f.write(
                "*See accompanying visualizations for detailed charts and graphs.*\n"
            )

        print(f"Report generated: {report_path}")
        return report_path

    def run_full_analysis(self):
        """
        Run the complete throughput analysis
        """
        print("Starting UTC Throughput Analysis...")

        # Analyze all proposals
        metrics_df = self.classify_proposals_by_era()

        if metrics_df.empty:
            print("No proposals with valid timeline data found.")
            return

        print(f"Analyzed {len(metrics_df)} proposals with timeline data")

        # Compare eras
        comparison_results = self.compare_eras(metrics_df)

        # Analyze people/body proposals
        people_body_results = self.analyze_people_body_proposals(metrics_df)

        # Create output directory
        output_dir = os.path.join(self.base_path, "throughput_analysis")
        os.makedirs(output_dir, exist_ok=True)

        # Generate visualizations
        self.create_visualizations(metrics_df, comparison_results, output_dir)

        # Generate report
        report_path = self.generate_report(
            metrics_df, comparison_results, people_body_results, output_dir
        )

        # Save processed data
        data_path = os.path.join(output_dir, "proposal_metrics_2017.csv")
        metrics_df.to_csv(data_path, index=False)

        print(f"\n✅ Analysis complete!")
        print(f"📊 Report: {report_path}")
        print(f"📈 Visualizations: {output_dir}")
        print(f"📋 Data: {data_path}")

        return {
            "metrics_df": metrics_df,
            "comparison_results": comparison_results,
            "people_body_results": people_body_results,
            "output_dir": output_dir,
        }


def main():
    """Main execution function"""
    analyzer = UTCThroughputAnalyzer()
    results = analyzer.run_full_analysis()
    return results


if __name__ == "__main__":
    main()
