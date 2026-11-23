"""
End-to-End Training, Optimal K Selection, and Customer Persona Profiling Pipeline.

Standardizes retail customer behaviors, evaluates Elbow & Silhouette sweeps,
trains K-Means, Hierarchical, and DBSCAN, profiles marketing personas,
generates diagnostic charts, and serializes the production segmentation engine.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import generate_customer_data, load_customer_data
from src.data_preprocessing import (
    preprocess_customer_data,
    scale_customer_features,
    apply_pca_reduction,
)
from src.cluster_models import (
    train_all_clustering_models,
    save_clustering_artifact,
)
from src.cluster_evaluation import (
    compute_elbow_and_silhouette_sweep,
    evaluate_clustering_metrics,
    profile_clusters,
    plot_clustering_diagnostics,
)
from src.segmenter import CustomerSegmenter


def run_pipeline() -> None:
    """Execute complete customer segmentation clustering workflow."""
    print("=" * 85)
    print("      PROJECT 05: CUSTOMER SEGMENTATION USING CLUSTERING - PIPELINE")
    print("=" * 85)

    data_dir = PROJECT_ROOT / "data"
    raw_csv = data_dir / "customer_segmentation.csv"
    proc_csv = data_dir / "processed_customers.csv"

    # Step 1: Ingestion
    print("\n[Step 1/6] Ingesting Customer Behavioral Records...")
    if not raw_csv.exists():
        print("  Generating synthetic retail customer dataset (2,000 customers)...")
        raw_df = generate_customer_data(n_samples=2000, seed=42, output_path=raw_csv)
        print(f"  Saved raw dataset to: {raw_csv}")
    else:
        print(f"  Loading existing dataset from: {raw_csv}")
        raw_df = load_customer_data(raw_csv)

    print(f"  Raw dimensions: {raw_df.shape[0]} records, {raw_df.shape[1]} attributes")

    # Step 2: Preprocessing & Standardization
    print("\n[Step 2/6] Cleaning Data & Applying StandardScaler...")
    clean_df, report = preprocess_customer_data(raw_df)
    clean_df.to_csv(proc_csv, index=False)
    print(f"  Duplicates removed: {report['duplicates_removed']}")
    print(f"  Valid customers after cleaning: {report['final_records']}")
    print(f"  Saved processed dataset to: {proc_csv}")

    X_scaled, scaler = scale_customer_features(clean_df)
    X_pca, pca_model, exp_var = apply_pca_reduction(X_scaled, n_components=2, random_state=42)
    print(f"  PCA 2D projection computed (Explained Variance Ratio: {exp_var:.2%})")

    # Step 3: Optimal K Selection (Elbow & Silhouette Sweep)
    print("\n[Step 3/6] Performing Elbow Method & Silhouette Sweep across k in [2, 10]...")
    sweep_df = compute_elbow_and_silhouette_sweep(X_scaled, k_range=range(2, 11), random_state=42)
    print("\n" + "-" * 45)
    print("      ELBOW & SILHOUETTE SCORE SWEEP")
    print("-" * 45)
    print(f"{'Clusters (k)':<15} | {'Inertia (WCSS)':<15} | {'Silhouette':<10}")
    print("-" * 45)
    for _, row in sweep_df.iterrows():
        print(f"{int(row['k']):<15} | {row['Inertia']:<15.2f} | {row['Silhouette_Score']:<10.4f}")
    print("-" * 45)

    best_k_row = sweep_df.sort_values(by="Silhouette_Score", ascending=False).iloc[0]
    optimal_k = int(best_k_row["k"])
    print(f"  >>> Optimal Cluster Count by Silhouette: k = {optimal_k} (Score = {best_k_row['Silhouette_Score']:.4f})")

    # Step 4: Multi-Algorithm Training (K-Means, Hierarchical, DBSCAN)
    print("\n[Step 4/6] Training Clustering Algorithms (K-Means, Hierarchical, DBSCAN)...")
    models_dict = train_all_clustering_models(X_scaled, n_clusters=optimal_k, random_state=42)
    labels_dict = {name: res[1] for name, res in models_dict.items()}

    eval_records = []
    for name, (_, labels) in models_dict.items():
        metrics = evaluate_clustering_metrics(X_scaled, labels, name)
        eval_records.append(metrics)

    eval_df = pd.DataFrame(eval_records)
    print("\n" + "-" * 75)
    print("                    CLUSTERING ALGORITHM BENCHMARK")
    print("-" * 75)
    print(f"{'Algorithm':<24} | {'Clusters':<8} | {'Silhouette':<10} | {'Davies-Bouldin':<14} | {'Calinski-Harabasz':<16}")
    print("-" * 75)
    for _, r in eval_df.iterrows():
        print(f"{r['Model']:<24} | {r['Clusters']:<8} | {r['Silhouette']:<10.4f} | {r['Davies_Bouldin']:<14.4f} | {r['Calinski_Harabasz']:<16.1f}")
    print("-" * 75)

    # Step 5: Persona Profiling & Business Playbooks
    print(f"\n[Step 5/6] Profiling {optimal_k} Customer Segments into Retail Personas...")
    km_labels = labels_dict["K-Means"]
    profiles = profile_clusters(clean_df, km_labels)

    for cid, p_info in sorted(profiles.items()):
        print(f"\n  Segment C{cid}: {p_info['persona_name']} ({p_info['percentage']}% of customers, N={p_info['customer_count']})")
        print(f"    - Mean Income: ${p_info['mean_attributes']['Annual_Income_k']:.1f}k | Spending Score: {p_info['mean_attributes']['Spending_Score']:.1f}")
        print(f"    - Avg Basket:  ${p_info['mean_attributes']['Average_Basket_Value']:.2f} | Purchase Freq: {p_info['mean_attributes']['Purchase_Frequency']:.0f} orders/yr")
        print(f"    - Strategy:    {p_info['marketing_strategy']}")

    # Step 6: Artifact Serialization & Verification
    print("\n[Step 6/6] Persisting Top Clustering Model & Verifying Segmenter...")
    km_model = models_dict["K-Means"][0]
    km_metrics = eval_df[eval_df["Model"] == "K-Means"].iloc[0].to_dict()

    saved_path = save_clustering_artifact(
        model=km_model,
        scaler=scaler,
        algorithm_name="K-Means",
        n_clusters=optimal_k,
        metrics={
            "silhouette": km_metrics["Silhouette"],
            "davies_bouldin": km_metrics["Davies_Bouldin"],
            "calinski_harabasz": km_metrics["Calinski_Harabasz"],
        },
        cluster_profiles=profiles,
    )
    print(f"  Clustering artifact serialized to: {saved_path}")

    # Generate diagnostic visualizations
    plots = plot_clustering_diagnostics(
        X_scaled=X_scaled,
        X_pca=X_pca,
        labels_dict=labels_dict,
        sweep_df=sweep_df,
        profiles=profiles,
    )
    print("  Generated visual diagnostics:")
    for plot_name, path in plots.items():
        print(f"    - {plot_name}: {path.name}")

    # Verify inference engine
    print("\n  Testing CustomerSegmenter Inference Engine...")
    segmenter = CustomerSegmenter(model_path=saved_path)

    sample_vip = {
        "Age": 38,
        "Annual_Income_k": 95.0,
        "Spending_Score": 85.0,
        "Purchase_Frequency": 30.0,
        "Average_Basket_Value": 220.0,
    }
    res_vip = segmenter.score_single_customer(sample_vip)
    print(f"\n  [Sample Customer: High Earner & Frequent Shopper]")
    print(f"  Assigned Segment:   C{res_vip['cluster_id']} - {res_vip['persona_name']}")
    print(f"  Customer Share:     {res_vip['customer_share_pct']}% of base")
    print(f"  Centroid Distance:  {res_vip['prototypical_distance']}")
    print(f"  Marketing Playbook: {res_vip['marketing_strategy']}")

    print("\n" + "=" * 85)
    print("               PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    run_pipeline()
