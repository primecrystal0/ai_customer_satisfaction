import pandas as pd
import json

INPUT_FILE = "data/classified_feedback.json"
SUMMARY_CSV = "data/dashboard_summary.csv"
ALERTS_CSV = "data/poor_response_alerts.csv"

CONFIDENCE_ALERT_THRESHOLD = 0.80  # matches your slide's ">0.80 confidence triggers auto-alert"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Drop any rows that errored out during classification
    df = df[df["category"].notna()] if "category" in df.columns else df

    print("=" * 50)
    print("CUSTOMER FEEDBACK DASHBOARD SUMMARY")
    print("=" * 50)

    total = len(df)
    print(f"\nTotal feedback forms processed: {total}")

    # --- Category distribution ---
    print("\n--- Category Distribution ---")
    dist = df["category"].value_counts()
    dist_pct = (dist / total * 100).round(1)
    for category in dist.index:
        print(f"{category:20s}: {dist[category]:3d}  ({dist_pct[category]}%)")

    # --- Average confidence per category ---
    print("\n--- Average Confidence per Category ---")
    avg_conf = df.groupby("category")["confidence"].mean().round(2)
    for category, conf in avg_conf.items():
        print(f"{category:20s}: {conf}")

    # --- High-risk alerts: Poor + high confidence ---
    alerts = df[(df["category"] == "Poor") & (df["confidence"] >= CONFIDENCE_ALERT_THRESHOLD)]
    print(f"\n--- Auto-Alerts Triggered (Poor, confidence >= {CONFIDENCE_ALERT_THRESHOLD}) ---")
    print(f"{len(alerts)} response(s) would trigger an auto-ticket (e.g. ServiceNow/Salesforce)")
    if len(alerts) > 0:
        for _, row in alerts.iterrows():
            print(f"  - {row['filename']}: \"{row['feedback_text'][:60]}...\"")

    # --- Save outputs ---
    summary_df = pd.DataFrame({
        "category": dist.index,
        "count": dist.values,
        "percentage": dist_pct.values,
        "avg_confidence": [avg_conf.get(cat, 0) for cat in dist.index],
    })
    summary_df.to_csv(SUMMARY_CSV, index=False)
    alerts.to_csv(ALERTS_CSV, index=False)

    print(f"\nSummary saved to {SUMMARY_CSV}")
    print(f"Alerts saved to {ALERTS_CSV}")


if __name__ == "__main__":
    main()