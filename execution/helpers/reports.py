import os
import logging

def generate_summary_stats(expanded_df):
    """
    Generate summary statistics and a mismatch report after Functional Max assignment.
    """
    total_cases = len(expanded_df)
    total_mismatches = expanded_df["Mismatch"].sum()
    mismatch_rate = (total_mismatches / total_cases) * 100 if total_cases > 0 else 0

    logging.info(f"✅ Total cases processed: {total_cases}")
    logging.info(f"✅ Mismatches found: {total_mismatches} ({mismatch_rate:.2f}%)")

    # Ensure the reports directory exists
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    # Export mismatches to CSV for review
    mismatches_df = expanded_df[expanded_df["Mismatch"]]
    mismatch_report_path = os.path.join(reports_dir, "mismatch_report.csv")
    mismatches_df.to_csv(mismatch_report_path, index=False)

    logging.info(f"✅ Mismatch report saved to {mismatch_report_path}")
