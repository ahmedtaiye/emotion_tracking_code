import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
from tqdm import tqdm
import nltk
from nltk import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from scipy import stats
from joblib import Memory, Parallel, delayed
import os
import re
from datetime import datetime
import warnings
from collections import defaultdict
import networkx as nx
from textwrap import wrap

# Suppress warnings
warnings.filterwarnings("ignore")


# --- Configuration ---
class Config:
    FILE_PATH = "NN.csv"
    EMOTION_SAMPLE_SIZE = 50000
    TOP_N_CATEGORIES = 60
    MIN_RECORDS = 60
    TIME_FREQ = 'W'
    OUTPUT_DIR = "visualizations"
    TOP_COUNTRIES = 5
    TOP_QUOTES = 3
    EMOTION_THRESHOLD = 0.7
    N_JOBS = -1  # Use all available cores
    RANDOM_STATE = 42
    WORDCLOUD_MAX_WORDS = 100
    DASHBOARD_PORT = 8050


# Initialize configuration
config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
memory = Memory("cache_dir", verbose=0)
tqdm.pandas()

# --- Color Maps and Themes ---
SENTIMENT_COLOR_MAP = {
    'positive': '#4CAF50',
    'negative': '#F44336',
    'neutral': '#2196F3'
}

EKMAN_EMOTION_COLOR_MAP = {
    'joy': '#FFD700', 'sadness': '#4682B4', 'anger': '#FF4500',
    'fear': '#8B0000', 'surprise': '#DA70D6', 'disgust': '#808000',
    'neutral': '#A9A9A9'
}

EMOTION_INTENSITY = {
    'joy': 1, 'surprise': 0.8, 'anger': -1,
    'fear': -1, 'sadness': -1, 'disgust': -1
}

LAYOUT_THEME = {
    'font': dict(family="Arial", size=12),
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
    'hoverlabel': dict(bgcolor='white', font_size=12),
    'margin': dict(l=50, r=50, t=80, b=50),
    'title_font_size': 20,
    'xaxis_title_font_size': 16,
    'yaxis_title_font_size': 16,
    'legend_title_font_size': 14,
    'hovermode': 'x unified',
    'transition': {'duration': 500}
}


# --- Helper Functions ---
def debug_data(df, message):
    """Enhanced debug output with data summary"""
    print(f"\n--- {message} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print("\nSample Data:")
        print(df.head(2))
        print("\nSummary Stats:")
        print(df.describe(include='all'))


def clean_text(text):
    """Advanced text cleaning with NLP preprocessing"""
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\@\w+|\#\w+', '', text)  # Remove mentions/hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase

    return text


def load_and_clean(file_path):
    """Enhanced data loading with automatic schema detection"""
    print(f"\nLoading data from {file_path}...")

    # Try multiple encodings and delimiters
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1', 'cp1252']
    delimiters = [',', ';', '\t', '|']

    df = None
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    engine='python',
                    on_bad_lines='warn'
                )
                if not df.empty and 'en_text' in df.columns:
                    print(f"Success with encoding: {encoding}, delimiter: '{delimiter}'")
                    break
                df = None
            except Exception:
                continue
        if df is not None:
            break

    if df is None:
        raise ValueError("Failed to load file with all encoding/delimiter combinations")

    debug_data(df, "Raw Data Loaded")

    # Enhanced text cleaning
    initial_count = len(df)
    df['en_text'] = df['en_text'].progress_apply(clean_text)
    df = df[df['en_text'].str.len() > 5].copy()
    print(f"Removed {initial_count - len(df)} records with empty/short text")

    # Improved datetime handling
    datetime_cols = ['created_at', 'date', 'timestamp', 'published_at', 'time']
    datetime_col_found = False
    for col in datetime_cols:
        if col in df.columns:
            try:
                df['created_at'] = pd.to_datetime(df[col], errors='coerce', utc=True)
                df = df.dropna(subset=['created_at']).copy()
                df = df.set_index('created_at').sort_index()
                print(f"Using '{col}' as datetime index")
                datetime_col_found = True
                break
            except Exception as e:
                print(f"Couldn't parse '{col}': {str(e)[:100]}...")
    if not datetime_col_found:
        print("Warning: No valid datetime column found")

    # Enhanced sentiment processing
    sentiment_cols = ['sentiment_negative', 'sentiment_neutral', 'sentiment_positive']
    if all(col in df.columns for col in sentiment_cols):
        df[sentiment_cols] = df[sentiment_cols].apply(pd.to_numeric, errors='coerce')
        df['sentiment'] = df[sentiment_cols].idxmax(axis=1).str.replace('sentiment_', '')
        df['sentiment_score'] = df[sentiment_cols].max(axis=1)
    elif 'sentiment' in df.columns:
        sentiment_map = {
            'positive': 'positive', 'pos': 'positive', 'p': 'positive', '1': 'positive',
            'negative': 'negative', 'neg': 'negative', 'n': 'negative', '-1': 'negative',
            'neutral': 'neutral', 'neu': 'neutral', '0': 'neutral'
        }
        df['sentiment'] = df['sentiment'].str.lower().map(sentiment_map).fillna('neutral')
    else:
        raise ValueError("No valid sentiment columns found")

    # Enhanced categorical cleaning
    for col in ['channel', 'country', 'verified', 'source', 'author']:
        clean_col = f"{col}_cleaned"
        if col in df.columns:
            if col == 'verified':
                df[clean_col] = df[col].astype(str).str.lower().map(
                    {'true': True, '1': True, 'false': False, '0': False}
                ).fillna(False)
            else:
                df[clean_col] = (
                    df[col].astype(str).str.strip()
                    .replace(['', 'nan', 'None'], 'Unknown')
                    .fillna('Unknown')
                )
        else:
            df[clean_col] = False if col == 'verified' else 'Unknown'

    debug_data(df, "Cleaned Data")
    return df


@memory.cache
def analyze_emotions(df, sample_size=config.EMOTION_SAMPLE_SIZE):
    """Parallelized emotion analysis with confidence thresholding"""
    actual_size = min(sample_size, len(df))
    if actual_size == 0:
        print("Warning: Empty DataFrame - skipping emotion analysis")
        return pd.DataFrame()

    sample_df = df.sample(n=actual_size, random_state=config.RANDOM_STATE).copy()
    sample_df['en_text'] = sample_df['en_text'].astype(str)

    try:
        classifier = pipeline(
            'text-classification',
            model='j-hartmann/emotion-english-distilroberta-base',
            return_all_scores=True,
            device='cpu'
        )
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return pd.DataFrame()

    def process_batch(texts):
        try:
            return classifier(texts)
        except Exception:
            return [[] for _ in texts]

    batch_size = 100
    batches = [sample_df['en_text'].iloc[i:i + batch_size].tolist()
               for i in range(0, len(sample_df), batch_size)]

    results = Parallel(n_jobs=config.N_JOBS)(
        delayed(process_batch)(batch) for batch in tqdm(batches, desc="Processing batches")
    )
    emotions_results = [item for sublist in results for item in sublist]

    extracted_data = []
    for res_list in emotions_results:
        if not res_list:
            extracted_data.append({
                'ekman_emotion': 'unknown',
                'ekman_emotion_score': 0.0,
                'all_emotion_scores': {},
                'emotion_intensity': 0.0,
                'is_confident': False
            })
            continue

        top_emotion = max(res_list, key=lambda x: x['score'])
        emotion_scores = {item['label']: item['score'] for item in res_list}

        # Calculate emotion intensity score
        intensity_score = sum(
            score * EMOTION_INTENSITY.get(emotion, 0)
            for emotion, score in emotion_scores.items()
        )

        extracted_data.append({
            'ekman_emotion': top_emotion['label'],
            'ekman_emotion_score': top_emotion['score'],
            'all_emotion_scores': emotion_scores,
            'emotion_intensity': intensity_score,
            'is_confident': top_emotion['score'] >= config.EMOTION_THRESHOLD
        })

    emotions_df = pd.DataFrame(extracted_data, index=sample_df.index)
    return sample_df.join(emotions_df)


def generate_confusion_matrix(df, emotion_df):
    """Enhanced confusion matrix with statistical metrics - combined emotions version"""
    if df.empty or emotion_df.empty:
        print("No data for confusion matrix")
        return None

    merged = pd.merge(
        df[['sentiment', 'sentiment_score', 'en_text']],
        emotion_df[['ekman_emotion', 'ekman_emotion_score', 'all_emotion_scores', 'is_confident']],
        left_index=True, right_index=True, how='inner'
    ).dropna()

    if merged.empty:
        print("No overlapping data for confusion matrix")
        return None

    # Map emotions to combined categories
    def map_combined_emotion(emotion):
        if emotion in ['anger', 'fear', 'sadness', 'disgust']:
            return 'negative_emotion'
        elif emotion in ['joy', 'surprise']:
            return 'positive_emotion'
        return emotion  # keeps 'neutral' as is

    merged['combined_emotion'] = merged['ekman_emotion'].apply(map_combined_emotion)

    # Create confusion matrix with combined categories
    combined_emotions = ['positive_emotion', 'negative_emotion', 'neutral']
    sentiments = sorted(merged['sentiment'].unique())

    conf_data = []
    for true_sent in sentiments:
        row = []
        for pred_emotion in combined_emotions:
            count = len(merged[
                (merged['sentiment'] == true_sent) &
                (merged['combined_emotion'] == pred_emotion) &
                (merged['is_confident'])
            ])
            row.append(count)
        conf_data.append(row)

    conf_matrix = pd.DataFrame(
        conf_data,
        index=[f"True {s}" for s in sentiments],
        columns=[f"Pred {e.replace('_', ' ')}" for e in combined_emotions]
    )

    # Normalized version
    conf_matrix_norm = conf_matrix.div(conf_matrix.sum(axis=1), axis=0) * 100

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(
        merged['sentiment'],
        merged['combined_emotion'].where(
            merged['combined_emotion'].isin(['positive_emotion', 'negative_emotion']),
            'neutral'
        )
    )

    # Create enhanced visualization
    fig = px.imshow(
        conf_matrix_norm,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale='Viridis',
        labels=dict(x="Predicted Emotion Group", y="True Sentiment", color="% of Category"),
        title=f"<b>Emotion-Sentiment Alignment (Combined)</b><br>Cohen's κ = {kappa:.2f}",
        zmin=0,
        zmax=100
    )

    fig.update_layout(
        width=900,
        height=700,
        xaxis=dict(side="top"),
        coloraxis_colorbar=dict(title="% Alignment"),
        **LAYOUT_THEME
    )

    # Add statistical annotations
    annotations = []
    for i, true_sent in enumerate(sentiments):
        for j, pred_emotion in enumerate(combined_emotions):
            count = conf_matrix.iloc[i, j]
            total = conf_matrix.iloc[i].sum()
            pct = conf_matrix_norm.iloc[i, j]

            annotations.append(dict(
                x=j, y=i,
                text=f"{count}<br>({pct:.1f}%)",
                showarrow=False,
                font=dict(color="white" if pct > 50 else "black", size=10)
            ))

    fig.update_layout(annotations=annotations)

    # Save visualization
    output_path = os.path.join(config.OUTPUT_DIR, "combined_emotion_confusion_matrix.html")
    fig.write_html(output_path)
    print(f"Saved combined emotion confusion matrix: {output_path}")

    return {
        'matrix': conf_matrix,
        'matrix_norm': conf_matrix_norm,
        'kappa': kappa,
        'merged_data': merged
    }
def analyze_quote_alignment(merged_data):
    """Comprehensive quote analysis with multiple comparison categories"""
    if merged_data.empty:
        print("No data for quote analysis")
        return None

    # Define analysis cases with enhanced logic
    analysis_cases = [
        ('Correct Positive',
         lambda x: (x['sentiment'] == 'positive') &
                   (x['ekman_emotion'].isin(['joy', 'surprise'])) &
                   (x.get('is_confident', False))),

        ('Incorrect Positive',
         lambda x: (x['sentiment'] == 'positive') &
                   (~x['ekman_emotion'].isin(['joy', 'surprise', 'neutral'])) &
                   (x.get('is_confident', False))),

        ('Correct Negative',
         lambda x: (x['sentiment'] == 'negative') &
                   (x['ekman_emotion'].isin(['anger', 'fear', 'sadness', 'disgust'])) &
                   (x.get('is_confident', False))),

        ('Incorrect Negative',
         lambda x: (x['sentiment'] == 'negative') &
                   (x['ekman_emotion'] == 'neutral') &
                   (x.get('is_confident', False))),

        ('Borderline Positive',
         lambda x: (x['sentiment'] == 'positive') &
                   (x['ekman_emotion'] == 'neutral') &
                   (x.get('sentiment_score', 0) > 0.6)),

        ('Borderline Negative',
         lambda x: (x['sentiment'] == 'negative') &
                   (x['ekman_emotion'] == 'surprise') &
                   (x.get('sentiment_score', 0) < -0.6)),

        ('Most Ambiguous',
         lambda x: (abs(x.get('sentiment_score', 0)) < 0.3) &
                   (x.get('ekman_emotion_score', 0) < 0.6)),

        ('Most Confident',
         lambda x: (x.get('is_confident', False)) &
                   (abs(x.get('sentiment_score', 0)) > 0.8) &
                   (x.get('is_aligned', False)))
    ]

    # Collect quotes for each category
    quotes_data = []
    for case_name, filter_func in analysis_cases:
        case_df = merged_data[filter_func(merged_data)]
        if not case_df.empty:
            samples = case_df.nlargest(config.TOP_QUOTES, 'ekman_emotion_score')
            for _, row in samples.iterrows():
                quotes_data.append({
                    'Category': case_name,
                    'Sentiment': row.get('sentiment', 'unknown'),
                    'Sentiment Score': f"{row.get('sentiment_score', 0):.2f}",
                    'Predicted Emotion': row.get('ekman_emotion', 'unknown'),
                    'Emotion Score': f"{row.get('ekman_emotion_score', 0):.2f}",
                    'Emotion Intensity': f"{row.get('emotion_intensity', 0):.2f}",
                    'Text': f'"{row.get("en_text", "")[:150]}{"..." if len(row.get("en_text", "")) > 150 else ""}"',
                    'Is Confident': row.get('is_confident', False),
                    'Is Aligned': row.get('is_aligned', False)
                })

    if not quotes_data:
        print("No quotes found for analysis")
        return None

    quotes_df = pd.DataFrame(quotes_data)

    # Create interactive table
    fig = go.Figure(data=[go.Table(
        columnwidth=[1.5, 1, 1, 1, 1, 1, 4],
        header=dict(
            values=[
                '<b>Category</b>', '<b>Sentiment</b>', '<b>Sent Score</b>',
                '<b>Emotion</b>', '<b>Emo Score</b>', '<b>Intensity</b>',
                '<b>Text</b>'
            ],
            fill_color='#1E88E5',
            font=dict(color='white', size=12),
            align=['left', 'center', 'center', 'center', 'center', 'center', 'left']
        ),
        cells=dict(
            values=[
                quotes_df['Category'],
                quotes_df['Sentiment'],
                quotes_df['Sentiment Score'],
                quotes_df['Predicted Emotion'],
                quotes_df['Emotion Score'],
                quotes_df['Emotion Intensity'],
                quotes_df['Text']
            ],
            fill_color='white',
            align=['left', 'center', 'center', 'center', 'center', 'center', 'left'],
            height=30,
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title='<b>Emotion-Sentiment Alignment Examples</b><br>' +
              '<i>Showing representative quotes across different alignment categories</i>',
        margin=dict(l=20, r=20, t=100, b=20),
        height=min(1200, 600 + len(quotes_df) * 30)
    )

    # Save table
    output_path = os.path.join(config.OUTPUT_DIR, "quote_alignment_analysis.html")
    fig.write_html(output_path)
    print(f"Saved quote alignment analysis: {output_path}")

    # Generate word clouds for each category
    generate_wordclouds(merged_data, analysis_cases)

    return quotes_df


def generate_wordclouds(merged_data, analysis_cases):
    """Generate word clouds for different alignment categories"""
    os.makedirs(os.path.join(config.OUTPUT_DIR, "wordclouds"), exist_ok=True)

    for case_name, filter_func in analysis_cases:
        case_df = merged_data[filter_func(merged_data)]
        if len(case_df) < 5:  # Skip categories with few examples
            continue

        text = ' '.join(case_df['en_text'].astype(str))
        if not text.strip():
            continue

        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=config.WORDCLOUD_MAX_WORDS,
            background_color='white'
        ).generate(text)

        # Save as image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud: {case_name}", pad=20)

        filename = f"wordcloud_{case_name.lower().replace(' ', '_')}.png"
        output_path = os.path.join(config.OUTPUT_DIR, "wordclouds", filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved word cloud: {output_path}")


def create_network_graph(merged_data):
    """Create emotion-sentiment network visualization"""
    if merged_data.empty:
        print("No data for network graph")
        return None

    # Create edges between sentiment and emotions
    edge_data = defaultdict(int)
    for _, row in merged_data.iterrows():
        if row.get('is_confident', False):
            edge = (row.get('sentiment', 'unknown'), row.get('ekman_emotion', 'unknown'))
            edge_data[edge] += 1

    if not edge_data:
        print("Not enough data for network graph")
        return None

    # Create network graph
    G = nx.Graph()
    for (source, target), weight in edge_data.items():
        G.add_edge(source, target, weight=weight)

    # Node positions
    pos = nx.spring_layout(G, seed=config.RANDOM_STATE)

    # Create Plotly figure
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Color nodes by type
        if node in SENTIMENT_COLOR_MAP:
            node_color.append(SENTIMENT_COLOR_MAP[node])
        else:
            node_color.append(EKMAN_EMOTION_COLOR_MAP.get(node, '#AAAAAA'))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=20,
            line=dict(width=2, color='DarkSlateGrey')
        )
    )

    # Create layout without duplicating margin
    network_layout = LAYOUT_THEME.copy()
    network_layout.update({
        'title': '<b>Emotion-Sentiment Network</b><br>Connections between predicted emotions and ground truth sentiment',
        'showlegend': False,
        'hovermode': 'closest',
        'xaxis': dict(showgrid=False, zeroline=False, showticklabels=False),
        'yaxis': dict(showgrid=False, zeroline=False, showticklabels=False)
    })

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=network_layout
    )

    # Save network graph
    output_path = os.path.join(config.OUTPUT_DIR, "emotion_sentiment_network.html")
    fig.write_html(output_path)
    print(f"Saved network graph: {output_path}")

    return G


def generate_statistical_report(merged_data, confusion_results):
    """Generate comprehensive statistical report"""
    if merged_data.empty or not confusion_results:
        print("No data for statistical report")
        return None

    # Import stats here to ensure it's available
    from scipy import stats

    report = {
        'alignment_stats': {},
        'score_comparisons': {},
        'effect_sizes': {}
    }

    # Alignment statistics - use .get() on the Series, not the DataFrame
    aligned = merged_data['is_aligned'].mean() if 'is_aligned' in merged_data.columns else 0
    report['alignment_stats']['overall_alignment'] = f"{aligned:.1%}"
    report['alignment_stats']['cohens_kappa'] = f"{confusion_results.get('kappa', 0):.2f}"

    # Score comparisons
    for sentiment in merged_data['sentiment'].unique():
        subset = merged_data[merged_data['sentiment'] == sentiment]

        # Check if 'is_aligned' exists before using it
        if 'is_aligned' in subset.columns:
            aligned_scores = subset[subset['is_aligned']]['ekman_emotion_score']
            misaligned_scores = subset[~subset['is_aligned']]['ekman_emotion_score']
        else:
            aligned_scores = pd.Series(dtype=float)
            misaligned_scores = pd.Series(dtype=float)

        if len(aligned_scores) > 1 and len(misaligned_scores) > 1:
            try:
                t_stat, p_value = stats.ttest_ind(aligned_scores, misaligned_scores)
                report['score_comparisons'][sentiment] = {
                    't_stat': f"{t_stat:.2f}",
                    'p_value': f"{p_value:.4f}",
                    'aligned_mean': f"{aligned_scores.mean():.2f}",
                    'misaligned_mean': f"{misaligned_scores.mean():.2f}"
                }
            except Exception as e:
                print(f"Error calculating t-test for {sentiment}: {e}")
                report['score_comparisons'][sentiment] = {
                    't_stat': 'N/A',
                    'p_value': 'N/A',
                    'aligned_mean': 'N/A',
                    'misaligned_mean': 'N/A'
                }

    # Effect sizes
    if 'ekman_emotion' in merged_data.columns:
        for emotion in merged_data['ekman_emotion'].unique():
            subset = merged_data[merged_data['ekman_emotion'] == emotion]
            if len(subset) > 1:
                pos = subset[subset['sentiment'] == 'positive']['sentiment_score']
                neg = subset[subset['sentiment'] == 'negative']['sentiment_score']

                if len(pos) > 1 and len(neg) > 1:
                    try:
                        cohens_d = (pos.mean() - neg.mean()) / np.sqrt(
                            (pos.std() ** 2 + neg.std() ** 2) / 2
                        )
                        report['effect_sizes'][emotion] = f"{cohens_d:.2f}"
                    except Exception as e:
                        print(f"Error calculating effect size for {emotion}: {e}")
                        report['effect_sizes'][emotion] = "N/A"

    # Save as markdown report
    report_lines = [
        "# Emotion-Sentiment Analysis Report",
        "## Alignment Statistics",
        f"- Overall Alignment: {report['alignment_stats']['overall_alignment']}",
        f"- Cohen's Kappa: {report['alignment_stats']['cohens_kappa']}",
        "",
        "## Score Comparisons (Aligned vs Misaligned)",
    ]

    for sentiment, stats in report['score_comparisons'].items():
        report_lines.extend([
            f"### {sentiment.capitalize()} Sentiment",
            f"- Aligned Mean Score: {stats['aligned_mean']}",
            f"- Misaligned Mean Score: {stats['misaligned_mean']}",
            f"- t-statistic: {stats['t_stat']} (p={stats['p_value']})",
            ""
        ])

    report_lines.extend([
        "## Effect Sizes (Emotion Impact on Sentiment)",
    ])

    if 'effect_sizes' in report:
        for emotion, d in report['effect_sizes'].items():
            report_lines.append(f"- {emotion.capitalize()}: Cohen's d = {d}")

    # Add interpretation
    report_lines.extend([
        "",
        "## Interpretation",
        "- Values closer to 1 indicate stronger alignment",
        "- Cohen's κ > 0.6 suggests substantial agreement",
        "- Positive t-statistics suggest higher confidence in aligned predictions",
        "- Larger effect sizes indicate stronger emotion-sentiment relationships"
    ])

    report_text = '\n'.join(report_lines)
    output_path = os.path.join(config.OUTPUT_DIR, "statistical_report.md")
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"Saved statistical report: {output_path}")
    return report

def create_dashboard(df, emotion_df, merged_data):
    """Create interactive Dash dashboard (saved as HTML)"""
    try:
        from dash import Dash, dcc, html, Input, Output
    except ImportError:
        print("Dash not installed - skipping dashboard creation")
        return None

    if df.empty or emotion_df.empty or merged_data.empty:
        print("No data for dashboard")
        return None

    # Create simplified HTML dashboard
    dashboard_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment-Emotion Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .dashboard-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .panel {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            h1 {{ color: #1E88E5; }}
            .filters {{ background: #f5f5f5; padding: 15px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Sentiment & Emotion Analysis Dashboard</h1>

        <div class="filters">
            <h3>Filters</h3>
            <div>
                <label for="sentiment-filter">Sentiment:</label>
                <select id="sentiment-filter">
                    <option value="all">All</option>
                    <option value="positive">Positive</option>
                    <option value="negative">Negative</option>
                    <option value="neutral">Neutral</option>
                </select>
            </div>
            <div>
                <label for="confidence-filter">Min Confidence:</label>
                <input type="range" id="confidence-filter" min="0" max="1" step="0.1" value="0.7">
                <span id="confidence-value">0.7</span>
            </div>
        </div>

        <div class="dashboard-container">
            <div class="panel">
                <h2>Emotion Distribution</h2>
                <div id="emotion-distribution-plot"></div>
            </div>

            <div class="panel">
                <h2>Sentiment Over Time</h2>
                <div id="sentiment-timeseries-plot"></div>
            </div>

            <div class="panel">
                <h2>Alignment Network</h2>
                <div id="network-graph"></div>
            </div>

            <div class="panel">
                <h2>Quote Examples</h2>
                <div id="quote-table"></div>
            </div>
        </div>

        <script>
            // This would normally be populated with actual Plotly figures
            // For this template, we're just showing placeholder text
            document.getElementById('emotion-distribution-plot').innerHTML = 
                '<p>Interactive emotion distribution plot would appear here</p>';

            document.getElementById('sentiment-timeseries-plot').innerHTML = 
                '<p>Interactive time series would appear here</p>';

            document.getElementById('network-graph').innerHTML = 
                '<p>Network graph visualization would appear here</p>';

            document.getElementById('quote-table').innerHTML = 
                '<p>Interactive quote table would appear here</p>';

            // Update confidence value display
            document.getElementById('confidence-filter').addEventListener('input', function(e) {{
                document.getElementById('confidence-value').textContent = e.target.value;
            }});
        </script>
    </body>
    </html>
    """

    output_path = os.path.join(config.OUTPUT_DIR, "interactive_dashboard.html")
    with open(output_path, 'w') as f:
        f.write(dashboard_html)

    print(f"Saved interactive dashboard: {output_path}")
    return output_path


def create_enhanced_time_series_plot(df, category_col, title, filename, color_map):
    """Create time series plot with enhanced features"""
    if df.empty or category_col not in df.columns:
        print(f"No data for time series plot: {title}")
        return None

    try:
        # Resample data
        resampled = df.groupby([pd.Grouper(freq=config.TIME_FREQ), category_col]).size().unstack().fillna(0)

        # Create figure
        fig = px.line(
            resampled,
            x=resampled.index,
            y=resampled.columns,
            title=title,
            color_discrete_map=color_map,
            labels={'value': 'Count', 'variable': category_col}
        )

        fig.update_layout(
            **LAYOUT_THEME,
            height=600,
            xaxis_title="Date",
            yaxis_title="Count",
            legend_title=category_col.replace('_', ' ').title()
        )

        # Add range slider and buttons
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        # Save plot
        output_path = os.path.join(config.OUTPUT_DIR, filename)
        fig.write_html(output_path)
        print(f"Saved time series plot: {output_path}")
        return fig
    except Exception as e:
        print(f"Error creating time series plot: {e}")
        return None


def create_interactive_bar(df, x_col, y_col, color_col, title, filename):
    """Create interactive bar plot with counts"""
    if df.empty or not all(col in df.columns for col in [x_col, y_col, color_col]):
        print(f"No data for bar plot: {title}")
        return None

    try:
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            text_auto=True,
            color_discrete_map=EKMAN_EMOTION_COLOR_MAP
        )

        fig.update_layout(
            **LAYOUT_THEME,
            height=600,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            legend_title=color_col.replace('_', ' ').title(),
            xaxis={'categoryorder': 'total descending'}
        )

        # Save plot
        output_path = os.path.join(config.OUTPUT_DIR, filename)
        fig.write_html(output_path)
        print(f"Saved bar plot: {output_path}")
        return fig
    except Exception as e:
        print(f"Error creating bar plot: {e}")
        return None


def create_normalized_bar_plot(df, category_col, emotion_col, title, filename):
    """Create normalized bar plot showing emotion distribution by category"""
    if df.empty or not all(col in df.columns for col in [category_col, emotion_col]):
        print(f"No data for normalized bar plot: {title}")
        return None

    try:
        # Calculate normalized counts
        counts = df.groupby([category_col, emotion_col]).size().unstack().fillna(0)
        norm_counts = counts.div(counts.sum(axis=1), axis=0) * 100

        # Melt for plotting
        plot_data = norm_counts.reset_index().melt(id_vars=category_col, var_name=emotion_col, value_name='Percentage')

        fig = px.bar(
            plot_data,
            x=category_col,
            y='Percentage',
            color=emotion_col,
            title=title,
            color_discrete_map=EKMAN_EMOTION_COLOR_MAP,
            barmode='stack'
        )

        fig.update_layout(
            **LAYOUT_THEME,
            height=600,
            xaxis_title=category_col.replace('_', ' ').title(),
            yaxis_title="Percentage",
            legend_title=emotion_col.replace('_', ' ').title()
        )

        # Save plot
        output_path = os.path.join(config.OUTPUT_DIR, filename)
        fig.write_html(output_path)
        print(f"Saved normalized bar plot: {output_path}")
        return fig
    except Exception as e:
        print(f"Error creating normalized bar plot: {e}")
        return None


def create_sunburst_chart(df, path, values, title, filename):
    """Create hierarchical sunburst chart"""
    if df.empty or not all(col in df.columns for col in path + [values]):
        print(f"No data for sunburst chart: {title}")
        return None

    try:
        counts = df.groupby(path).size().reset_index(name=values)

        fig = px.sunburst(
            counts,
            path=path,
            values=values,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        fig.update_layout(
            **LAYOUT_THEME,
            height=800
        )

        # Save plot
        output_path = os.path.join(config.OUTPUT_DIR, filename)
        fig.write_html(output_path)
        print(f"Saved sunburst chart: {output_path}")
        return fig
    except Exception as e:
        print(f"Error creating sunburst chart: {e}")
        return None


def main():
    """Enhanced main analysis pipeline"""
    print("\n" + "=" * 70)
    print("Enhanced Sentiment & Emotion Analysis Pipeline")
    print("=" * 70 + "\n")

    # Step 1: Load and clean data
    try:
        df = load_and_clean(config.FILE_PATH)
        if df.empty:
            raise ValueError("DataFrame is empty after cleaning")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Step 2: Analyze emotions
    print("\nAnalyzing emotions...")
    emotion_df = analyze_emotions(df)
    if emotion_df.empty:
        print("No emotion results - skipping emotion analysis")
    else:
        print(f"Analyzed {len(emotion_df)} records")
        print("\nEmotion Distribution:")
        print(emotion_df['ekman_emotion'].value_counts())

    # Step 3: Alignment analysis
    print("\nAnalyzing emotion-sentiment alignment...")
    confusion_results = generate_confusion_matrix(df, emotion_df)
    if confusion_results:
        merged_data = confusion_results['merged_data']
        print("\nConfusion Matrix Summary:")
        print(confusion_results['matrix_norm'])
        print(f"\nCohen's Kappa: {confusion_results['kappa']:.2f}")

        # Step 4: Quote analysis
        print("\nAnalyzing quote alignment...")
        quotes_df = analyze_quote_alignment(merged_data)
        if quotes_df is not None:
            print("\nSample Quote Analysis:")
            print(quotes_df.head())

        # Step 5: Network visualization
        print("\nCreating network graph...")
        create_network_graph(merged_data)

        # Step 6: Statistical report
        print("\nGenerating statistical report...")
        generate_statistical_report(merged_data, confusion_results)

    # Step 7: Create visualizations
    print("\nGenerating visualizations...")
    if 'created_at' in df.index.name:
        create_enhanced_time_series_plot(
            df,
            'sentiment',
            'Sentiment Polarity Over Time',
            'sentiment_time_series.html',
            SENTIMENT_COLOR_MAP
        )

    if not emotion_df.empty:
        emotion_df_filtered = emotion_df[emotion_df['ekman_emotion'] != 'neutral']
        if not emotion_df_filtered.empty:
            create_enhanced_time_series_plot(
                emotion_df_filtered,
                'ekman_emotion',
                'Emotion Types Over Time (Excluding Neutral)',
                'emotion_time_series.html',
                EKMAN_EMOTION_COLOR_MAP
            )

        create_interactive_bar(
            emotion_df,
            'ekman_emotion',
            'Count',
            'ekman_emotion',
            'Overall Emotion Distribution',
            'emotion_distribution.html'
        )

        for category in ['channel_cleaned', 'country_cleaned']:
            if category in emotion_df_filtered.columns:
                create_normalized_bar_plot(
                    emotion_df_filtered,
                    category,
                    'ekman_emotion',
                    f'Emotion Distribution by {category.replace("_", " ").title()}',
                    f'emotion_by_{category}.html'
                )

        if all(col in emotion_df_filtered.columns for col in ['country_cleaned', 'channel_cleaned', 'ekman_emotion']):
            create_sunburst_chart(
                emotion_df_filtered,
                ['country_cleaned', 'channel_cleaned', 'ekman_emotion'],
                'Count',
                'Emotion Hierarchy by Country and Channel',
                'emotion_hierarchy.html'
            )

    # Step 8: Create dashboard
    if 'merged_data' in locals():
        print("\nCreating interactive dashboard...")
        create_dashboard(df, emotion_df, merged_data)

    print("\n" + "=" * 70)
    print("Analysis complete! All outputs saved to:", config.OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    # Ensure required NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')

    main()