"""
Streamlit app for analyzing PhotoTriage training results.

Visualizes model predictions, errors, and performance breakdown by attributes.

Usage:
    streamlit run analyze_phototriage_results.py -- --results outputs/phototriage_multifeature/detailed_predictions.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import argparse
from PIL import Image

st.set_page_config(page_title="PhotoTriage Results Analysis", layout="wide")


@st.cache_data
def load_results(csv_path: str) -> pd.DataFrame:
    """Load detailed predictions CSV."""
    df = pd.read_csv(csv_path)
    return df


def plot_score_distribution(df: pd.DataFrame):
    """Plot distribution of model scores."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df['score1'],
        name='Image 1 Scores',
        opacity=0.7,
        nbinsx=50
    ))

    fig.add_trace(go.Histogram(
        x=df['score2'],
        name='Image 2 Scores',
        opacity=0.7,
        nbinsx=50
    ))

    fig.update_layout(
        title="Distribution of Model Scores",
        xaxis_title="Score",
        yaxis_title="Count",
        barmode='overlay'
    )

    return fig


def plot_accuracy_by_attribute(df: pd.DataFrame, attribute_cols: list):
    """Plot accuracy breakdown by quality attributes."""
    results = []

    for col in attribute_cols:
        if col in df.columns:
            attr_name = col.replace('label_', '').replace('_', ' ').title()

            # Calculate accuracy for pairs where this attribute is present (value > 0)
            mask = df[col] > 0
            if mask.sum() > 0:
                accuracy = df[mask]['correct'].mean()
                count = mask.sum()
                results.append({
                    'attribute': attr_name,
                    'accuracy': accuracy,
                    'count': count
                })

    if not results:
        return None

    results_df = pd.DataFrame(results).sort_values('accuracy')

    fig = px.bar(
        results_df,
        x='accuracy',
        y='attribute',
        orientation='h',
        text=results_df['count'],
        title="Model Accuracy by Quality Attribute",
        labels={'accuracy': 'Accuracy', 'attribute': 'Attribute'}
    )

    fig.update_traces(texttemplate='%{text} pairs', textposition='outside')
    fig.update_xaxes(range=[0, 1])
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Random Baseline")

    return fig


def plot_confusion_matrix(df: pd.DataFrame):
    """Plot confusion matrix of predictions."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(df['true_winner'], df['predicted_winner'])

    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Winner", y="True Winner", color="Count"),
        x=['Image 1', 'Image 2'],
        y=['Image 1', 'Image 2'],
        text_auto=True,
        title="Confusion Matrix"
    )

    return fig


def display_pair_comparison(df: pd.DataFrame, idx: int, image_dir: Path):
    """Display a single pair comparison with images and scores."""
    row = df.iloc[idx]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image 1")
        img1_path = image_dir / row['image1']
        if img1_path.exists():
            st.image(str(img1_path), use_container_width=True)
        else:
            st.error(f"Image not found: {img1_path}")

        st.metric("Model Score", f"{row['score1']:.4f}")
        if row['true_winner'] == 0:
            st.success("TRUE WINNER")
        if row['predicted_winner'] == 0:
            st.info("Predicted Winner")

    with col2:
        st.subheader("Image 2")
        img2_path = image_dir / row['image2']
        if img2_path.exists():
            st.image(str(img2_path), use_container_width=True)
        else:
            st.error(f"Image not found: {img2_path}")

        st.metric("Model Score", f"{row['score2']:.4f}")
        if row['true_winner'] == 1:
            st.success("TRUE WINNER")
        if row['predicted_winner'] == 1:
            st.info("Predicted Winner")

    # Display attributes
    st.subheader("Quality Attributes")
    attr_cols = [col for col in df.columns if col.startswith('label_')]
    active_attrs = [col.replace('label_', '').replace('_', ' ').title()
                   for col in attr_cols if row[col] > 0]

    if active_attrs:
        st.write(", ".join(active_attrs))
    else:
        st.write("No specific attributes tagged")

    # Display metadata
    col1, col2, col3 = st.columns(3)
    col1.metric("Agreement", f"{row['agreement']:.2f}")
    col2.metric("Reviewers", int(row['num_reviewers']))
    col3.metric("Score Difference", f"{row['score_diff']:.4f}")


def main():
    st.title("PhotoTriage Training Results Analysis")

    # Sidebar for file selection
    st.sidebar.header("Configuration")

    results_path = st.sidebar.text_input(
        "Results CSV Path",
        value="outputs/phototriage_multifeature/detailed_predictions.csv"
    )

    image_dir = st.sidebar.text_input(
        "Image Directory",
        value=r"D:/Similar Images/automatic_triage_photo_series/train_val/train_val_imgs"
    )

    if not Path(results_path).exists():
        st.error(f"Results file not found: {results_path}")
        st.info("Please run training first to generate detailed_predictions.csv")
        return

    # Load data
    df = load_results(results_path)

    st.sidebar.metric("Total Pairs", len(df))
    st.sidebar.metric("Overall Accuracy", f"{100*df['correct'].mean():.2f}%")
    st.sidebar.metric("Correct Predictions", df['correct'].sum())
    st.sidebar.metric("Incorrect Predictions", (~df['correct']).sum())

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Attribute Analysis",
        "Error Analysis",
        "Individual Pairs"
    ])

    # Tab 1: Overview
    with tab1:
        st.header("Performance Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(plot_score_distribution(df), use_container_width=True)

        with col2:
            st.plotly_chart(plot_confusion_matrix(df), use_container_width=True)

        # Score difference analysis
        st.subheader("Score Difference vs Correctness")

        df['abs_score_diff'] = df['score_diff'].abs()

        fig = px.box(
            df,
            x='correct',
            y='abs_score_diff',
            labels={'correct': 'Prediction Correct', 'abs_score_diff': 'Absolute Score Difference'},
            title="Model Confidence (Score Difference) by Correctness"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Attribute Analysis
    with tab2:
        st.header("Performance by Quality Attribute")

        attr_cols = [col for col in df.columns if col.startswith('label_')]

        if attr_cols:
            fig = plot_accuracy_by_attribute(df, attr_cols)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No attribute data available")

            # Detailed attribute table
            st.subheader("Detailed Attribute Statistics")

            attr_stats = []
            for col in attr_cols:
                attr_name = col.replace('label_', '').replace('_', ' ').title()
                mask = df[col] > 0

                if mask.sum() > 0:
                    attr_stats.append({
                        'Attribute': attr_name,
                        'Count': mask.sum(),
                        'Accuracy': f"{100*df[mask]['correct'].mean():.2f}%",
                        'Avg Score Diff': f"{df[mask]['abs_score_diff'].mean():.4f}"
                    })

            if attr_stats:
                st.dataframe(pd.DataFrame(attr_stats), use_container_width=True)
        else:
            st.warning("No attribute columns found in results")

    # Tab 3: Error Analysis
    with tab3:
        st.header("Error Analysis")

        errors_df = df[~df['correct']].copy()

        st.metric("Total Errors", len(errors_df))

        if len(errors_df) > 0:
            # Error distribution by agreement level
            st.subheader("Errors by Agreement Level")

            fig = px.histogram(
                errors_df,
                x='agreement',
                nbins=20,
                title="Error Distribution by Human Agreement",
                labels={'agreement': 'Agreement Level', 'count': 'Number of Errors'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # High confidence errors
            st.subheader("High Confidence Errors (Large Score Difference)")

            errors_df['abs_score_diff'] = errors_df['score_diff'].abs()
            high_conf_errors = errors_df.nlargest(10, 'abs_score_diff')[
                ['image1', 'image2', 'score1', 'score2', 'score_diff', 'agreement', 'num_reviewers']
            ]

            st.dataframe(high_conf_errors, use_container_width=True)
        else:
            st.success("No errors! Perfect accuracy!")

    # Tab 4: Individual Pairs
    with tab4:
        st.header("Browse Individual Pairs")

        # Filter options
        col1, col2, col3 = st.columns(3)

        with col1:
            show_correct = st.checkbox("Show Correct", value=True)
        with col2:
            show_incorrect = st.checkbox("Show Incorrect", value=True)
        with col3:
            min_agreement = st.slider("Min Agreement", 0.0, 1.0, 0.7)

        # Filter dataframe
        filtered_df = df[df['agreement'] >= min_agreement].copy()

        if show_correct and not show_incorrect:
            filtered_df = filtered_df[filtered_df['correct']]
        elif show_incorrect and not show_correct:
            filtered_df = filtered_df[~filtered_df['correct']]
        elif not show_correct and not show_incorrect:
            st.warning("Please select at least one filter option")
            return

        st.write(f"Showing {len(filtered_df)} pairs")

        if len(filtered_df) > 0:
            # Pair selector
            pair_idx = st.slider(
                "Select Pair",
                0,
                len(filtered_df) - 1,
                0
            )

            # Display pair
            display_pair_comparison(filtered_df, pair_idx, Path(image_dir))

            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if st.button("Previous Pair") and pair_idx > 0:
                    st.rerun()

            with col3:
                if st.button("Next Pair") and pair_idx < len(filtered_df) - 1:
                    st.rerun()


if __name__ == "__main__":
    main()
