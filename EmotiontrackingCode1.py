import pandas as pd
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import warnings
import re
import csv
import nltk
from joblib import Memory
import os
from datetime import datetime
import numpy as np

# --- Configuration ---
FILE_PATH = "NN.csv"  # Replace with your dataset path
EMOTION_SAMPLE_SIZE = 50000  # Reduce if dataset is smaller
TOP_N_CATEGORIES_FOR_PLOTS = 60
MIN_RECORDS_FOR_CATEGORY_PLOT = 60
DEFAULT_TIME_FREQUENCY = 'W'  # Weekly aggregation
OUTPUT_DIR = "visualizations"  # Directory to save HTML files
TOP_COUNTRIES = 5  # Number of top countries to analyze for the quotes table

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enable caching for emotion analysis to speed up repeated runs
# The 'cache_dir' will store results of 'analyze_emotions' function
memory = Memory("cache_dir", verbose=0)
tqdm.pandas()  # Enable tqdm for pandas operations (e.g., apply)
warnings.filterwarnings("ignore")  # Temporarily suppress warnings

# --- Color Maps and Themes ---
# Define consistent color maps for sentiment and Ekman emotions
SENTIMENT_COLOR_MAP = {'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#2196F3'}
EKMAN_EMOTION_COLOR_MAP = {
    'joy': '#FFD700',  # Gold
    'sadness': '#4682B4',  # SteelBlue
    'anger': '#FF4500',  # OrangeRed
    'fear': '#8B0000',  # DarkRed
    'surprise': '#DA70D6',  # Orchid
    'disgust': '#808000',  # Olive
    'neutral': '#A9A9A9'  # DarkGray (kept for general mapping, but filtered in specific analyses)
}

# Centralized theme for all Plotly plots for consistent aesthetics
LAYOUT_THEME = {
    'font': dict(family="Arial", size=12),
    'plot_bgcolor': 'white',  # Background color of the plot area
    'paper_bgcolor': 'white',  # Background color of the entire paper/figure
    'hoverlabel': dict(bgcolor='white', font_size=12),  # Style for hover tooltips
    'margin': dict(l=50, r=50, t=80, b=50),  # Margins around the plot
    'title_font_size': 20,
    'xaxis_title_font_size': 16,
    'yaxis_title_font_size': 16,
    'legend_title_font_size': 14,
    'hovermode': 'x unified',  # Unify hover effects across x-axis
    'transition': {'duration': 500}  # Smooth transitions for interactive changes
}


# --- Helper Functions ---
def debug_data(df, message):
    """
    Prints debug information (head, shape, columns) of a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to debug.
        message (str): A descriptive message for the debug output.
    """
    print(f"\n--- {message} ---")
    print(df.head(2))
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


def clean_text(text):
    """
    Performs basic text cleaning: handles NaNs, strips whitespace, and
    replaces multiple spaces with a single space.
    Args:
        text (str): The input text string.
    Returns:
        str: The cleaned text string.
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text


def load_and_clean(file_path):
    """
    Loads data from a CSV file, attempts to infer delimiter and encoding,
    and performs initial cleaning and feature engineering.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The cleaned and processed DataFrame, or None if loading fails.
    Raises:
        ValueError: If the file cannot be loaded or is empty after cleaning.
        KeyError: If essential columns like 'en_text' or sentiment columns are missing.
    """
    print(f"\nLoading data from {file_path}...")

    # Define common encodings and delimiters to try for robust loading
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1', 'cp1252']
    delimiters = [',', ';', '\t', '|']

    df = None
    # Iterate through encodings and delimiters to find a successful combination
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                # Attempt to read the CSV with the current encoding and delimiter
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    engine='python',  # 'python' engine is more flexible for delimiter inference
                    on_bad_lines='warn'  # Warns about bad lines instead of raising an error
                )
                # Check if DataFrame is not empty and 'en_text' column exists
                if not df.empty and 'en_text' in df.columns:
                    print(f"Successfully loaded with encoding: {encoding}, delimiter: '{delimiter}'")
                    break  # Break from inner (delimiter) loop
                else:
                    df = None  # Reset df if conditions not met, try next combination
            except Exception as e:
                # Continue to next combination if an error occurs during loading
                continue
        if df is not None:
            break  # Break from outer (encoding) loop if df is loaded

    if df is None:
        raise ValueError("Could not load file with any tried encoding/delimiter combination")

    debug_data(df, "Raw Data Loaded")

    # Basic text cleaning and filtering for valid text length
    initial_count = len(df)
    df['en_text'] = df['en_text'].apply(clean_text)
    df = df[df['en_text'].str.len() > 5].copy()  # Filter records with very short or empty text
    print(f"Removed {initial_count - len(df)} records with empty/short text")

    # Handle datetime column: find, parse, and set as index
    datetime_cols = ['created_at', 'date', 'timestamp']
    datetime_col_found = False
    for col in datetime_cols:
        if col in df.columns:
            try:
                df['created_at'] = pd.to_datetime(df[col], errors='coerce', utc=True)
                df = df.dropna(subset=['created_at']).copy()  # Drop rows where datetime parsing failed
                df = df.set_index('created_at').sort_index()  # Set as index and sort
                print(f"Using '{col}' as datetime index")
                datetime_col_found = True
                break
            except Exception as e:
                print(f"Could not parse '{col}' as datetime: {e}")
                continue
    if not datetime_col_found:
        print("Warning: No suitable datetime column found. Time-series analysis might be skipped.")

    # Handle sentiment columns: derive a single 'sentiment' category
    sentiment_cols = ['sentiment_negative', 'sentiment_neutral', 'sentiment_positive']
    if all(col in df.columns for col in sentiment_cols):
        # If explicit sentiment score columns exist, use them to determine overall sentiment
        df[sentiment_cols] = df[sentiment_cols].apply(pd.to_numeric, errors='coerce')
        # Assign sentiment based on the column with the maximum score
        df['sentiment'] = df[sentiment_cols].idxmax(axis=1).str.replace('sentiment_', '')
    elif 'sentiment' in df.columns:
        # If a generic 'sentiment' column exists, map its values to standard categories
        sentiment_map = {
            'positive': 'positive', 'pos': 'positive', 'p': 'positive',
            'negative': 'negative', 'neg': 'negative', 'n': 'negative',
            'neutral': 'neutral', 'neu': 'neutral', '0': 'neutral'
        }
        df['sentiment'] = df['sentiment'].str.lower().map(sentiment_map).fillna('neutral')
    else:
        raise ValueError("No valid sentiment columns found (neither explicit scores nor a generic 'sentiment' column).")

    # Clean and standardize categorical columns ('channel', 'country', 'verified')
    for col in ['channel', 'country', 'verified']:
        clean_col = f"{col}_cleaned"
        if col in df.columns:
            df[clean_col] = df[col].astype(str).str.strip().replace('', 'Unknown').fillna('Unknown')
            if col == 'verified':
                # Convert 'verified' to boolean
                df[clean_col] = df[clean_col].str.lower().map(
                    {'true': True, '1': True, 'false': False, '0': False}
                ).fillna(False)  # Default to False if value is not clearly true/1
        else:
            # If original column not found, create a cleaned column with 'Unknown' or default False
            print(f"Warning: Original '{col}' column not found. Setting '{clean_col}' to 'Unknown' or default False.")
            if col == 'verified':
                df[clean_col] = False
            else:
                df[clean_col] = 'Unknown'

    debug_data(df, "Cleaned Data")
    return df


@memory.cache
def analyze_emotions(df, sample_size=EMOTION_SAMPLE_SIZE):
    """
    Analyzes emotions in a sample of the DataFrame using a pre-trained emotion analysis model.
    Caches results to avoid reprocessing. Stores top emotion, its score, and all emotion scores.
    Args:
        df (pd.DataFrame): The input DataFrame containing 'en_text'.
        sample_size (int): The number of records to sample for emotion analysis.
    Returns:
        pd.DataFrame: A DataFrame containing the sampled texts with 'ekman_emotion',
                      'ekman_emotion_score', and 'all_emotion_scores' columns.
    """
    actual_sample_size = min(sample_size, len(df))
    if actual_sample_size == 0:
        print("Warning: DataFrame is empty, cannot analyze emotions.")
        return pd.DataFrame()

    # Sample the DataFrame and ensure 'en_text' is string type
    sample_df = df.sample(n=actual_sample_size, random_state=42).copy()
    sample_df['en_text'] = sample_df['en_text'].astype(str)

    try:
        # Load the emotion classification pipeline
        emotion_classifier = pipeline(
            'text-classification',
            model='j-hartmann/emotion-english-distilroberta-base',
            return_all_scores=True,  # Crucial for getting all emotion scores
            device='cpu'  # Use 'cuda' if GPU is available and configured
        )
    except Exception as e:
        print(f"Error loading emotion classifier: {e}. Skipping emotion analysis.")
        return pd.DataFrame()

    emotions_results = []
    # Process texts in batches to improve performance and manage memory
    batch_size = 100
    for i in tqdm(range(0, len(sample_df), batch_size), desc="Analyzing emotions"):
        batch = sample_df['en_text'].iloc[i:i + batch_size].tolist()
        try:
            results = emotion_classifier(batch)
            emotions_results.extend(results)
        except Exception as e:
            print(f"Error processing batch {i} of texts: {e}. Skipping this batch.")
            # Extend with empty lists to maintain length consistency if a batch fails
            emotions_results.extend([[] for _ in batch])

    extracted_emotions_data = []
    # Iterate through the results to extract the top emotion, its score, and all scores
    for res_list in emotions_results:
        if res_list:
            top_emotion = max(res_list, key=lambda x: x['score'])
            emotion_scores_dict = {item['label']: item['score'] for item in res_list}
            extracted_emotions_data.append({
                'ekman_emotion': top_emotion['label'],
                'ekman_emotion_score': top_emotion['score'],
                'all_emotion_scores': emotion_scores_dict
            })
        else:
            # Handle cases where no emotion could be predicted for a text
            extracted_emotions_data.append({
                'ekman_emotion': 'unknown',
                'ekman_emotion_score': 0.0,
                'all_emotion_scores': {}
            })

    # Create a DataFrame from the extracted emotion data and join it back to the sample_df
    emotions_df_results = pd.DataFrame(extracted_emotions_data, index=sample_df.index)
    sample_df = sample_df.join(emotions_df_results)

    return sample_df


def create_enhanced_time_series_plot(df, column, title_suffix, filename, color_map):
    """
    Creates an enhanced interactive time series plot showing the distribution of categories
    (sentiment or emotion) over time. Includes interactive controls and moving averages.
    Args:
        df (pd.DataFrame): The DataFrame with a 'created_at' index.
        column (str): The column to plot (e.g., 'sentiment', 'ekman_emotion').
        title_suffix (str): Additional text for the plot title.
        filename (str): The name of the HTML file to save the plot.
        color_map (dict): A dictionary mapping categories to colors.
    """
    if df.empty or 'created_at' not in df.index.name:
        print(f"Skipping time series plot for '{filename}': DataFrame is empty or 'created_at' index is missing.")
        return

    # Resample and count occurrences of each category (sentiment/emotion) per time frequency
    df_resampled = df.groupby([pd.Grouper(freq=DEFAULT_TIME_FREQUENCY), column]).size().unstack(fill_value=0)
    # Remove time periods where no records exist
    df_resampled = df_resampled[df_resampled.sum(axis=1) > 0]

    if len(df_resampled) < 2:
        print(f"Not enough data points for {column} time series. Skipping plot for {filename}.")
        return

    # Melt the DataFrame for Plotly Express to create multi-line plot
    df_melted = df_resampled.reset_index().melt(id_vars='created_at', var_name=column, value_name='Count')

    # Create the line plot
    fig = px.line(
        df_melted,
        x='created_at',  # Use the 'created_at' column for the x-axis
        y='Count',
        color=column,  # Color lines by sentiment/emotion category
        title=f"{column.replace('_', ' ').title()} Over Time ({title_suffix})<br><sup>Aggregated {DEFAULT_TIME_FREQUENCY} frequency</sup>",
        labels={'Count': 'Number of Records', 'created_at': 'Date', column: column.replace('_', ' ').title()},
        line_shape='spline',  # Smooth the lines
        color_discrete_map=color_map,  # Apply the defined color map
        hover_data={'created_at': "|%Y-%m-%d", 'Count': True}  # Custom hover info for date and count
    )

    # Add interactive controls (dropdowns for category visibility and time aggregation)
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label="All",
                         method="restyle",
                         args=[{"visible": [True] * len(df_resampled.columns)}]),  # Show all lines
                    *[
                        dict(label=cat,
                             method="restyle",
                             args=[{"visible": [True if x == cat else False for x in df_resampled.columns]}])
                        for cat in df_resampled.columns  # Button for each category to show only that line
                    ]
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                y=1.15
            ),
            dict(
                buttons=list([
                    dict(label="Daily", method="relayout", args=[{"xaxis.dtick": "D1"}]),
                    dict(label="Weekly", method="relayout", args=[{"xaxis.dtick": "W1"}]),
                    dict(label="Monthly", method="relayout", args=[{"xaxis.dtick": "M1"}]),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.3,
                y=1.15
            )
        ],
        # Add a range slider and selector for easy date navigation
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        **LAYOUT_THEME  # Apply the centralized theme
    )

    # Add moving averages for each line to show trends
    for col in df_resampled.columns:
        # Dynamic window size: at least 4, or 10% of data points if more than 40
        window_size = max(4, len(df_resampled) // 10)
        moving_avg = df_resampled[col].rolling(window=window_size).mean()
        fig.add_trace(go.Scatter(
            x=df_resampled.index,
            y=moving_avg,
            name=f'{col} (Avg)',  # Label for the moving average line
            line=dict(dash='dot', color=color_map.get(col, '#000000'))  # Dotted line, same color as category
        ))

    # Save the plot as an HTML file
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(output_path)
    print(f"Saved enhanced time series plot: {output_path}")


def create_interactive_bar(df, x_col, y_col, color_col, title, filename):
    """
    Creates an interactive bar chart with sorting options (by count or alphabetically).
    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): Column for the x-axis (categories).
        y_col (str): Column for the y-axis (counts).
        color_col (str): Column to color the bars by.
        title (str): The plot title.
        filename (str): The name of the HTML file to save the plot.
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print(f"Skipping interactive bar chart for '{filename}': Missing required columns or empty DataFrame.")
        return

    # Aggregate data for plotting
    df_agg = df.groupby([x_col, color_col]).size().reset_index(name=y_col)

    # Create two versions of the aggregated DataFrame for sorting options
    df_sorted_count = df_agg.sort_values(y_col, ascending=False)
    df_sorted_name = df_agg.sort_values(x_col)

    # Create the initial bar chart
    fig = px.bar(
        df_agg,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        color_discrete_map=EKMAN_EMOTION_COLOR_MAP if color_col == 'ekman_emotion' else SENTIMENT_COLOR_MAP,
        text_auto=True  # Automatically show text labels on bars
    )

    # Add sorting buttons to the layout
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label="Sort by Count",
                         method="update",
                         args=[{"x": [df_sorted_count[x_col]],
                                "y": [df_sorted_count[y_col]]}]),  # Update x and y data for sorting
                    dict(label="Sort Alphabetically",
                         method="update",
                         args=[{"x": [df_sorted_name[x_col]],
                                "y": [df_sorted_name[y_col]]}]),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                y=1.1
            )
        ],
        **LAYOUT_THEME  # Apply the centralized theme
    )

    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(output_path)
    print(f"Saved bar chart: {output_path}")


def create_sunburst_chart(df, path, values, title, filename):
    """
    Creates a hierarchical sunburst chart.
    Args:
        df (pd.DataFrame): The input DataFrame.
        path (list): List of column names to define the hierarchy levels.
        values (str): Column name for the values to determine segment size.
        title (str): The plot title.
        filename (str): The name of the HTML file to save the plot.
    """
    if df.empty or any(col not in df.columns for col in path):
        print(f"Skipping sunburst chart for '{filename}': Missing required columns or empty DataFrame.")
        return

    # Aggregate data based on the specified hierarchy path
    df_agg = df.groupby(path).size().reset_index(name=values)

    # Create the sunburst chart
    fig = px.sunburst(
        df_agg,
        path=path,  # Define the hierarchy
        values=values,  # Size of segments
        title=title,
        color_discrete_map=EKMAN_EMOTION_COLOR_MAP if 'ekman_emotion' in path else SENTIMENT_COLOR_MAP,
        # Apply color map
        hover_data={col: True for col in path}  # Show all path columns on hover
    )

    fig.update_layout(**LAYOUT_THEME)  # Apply the centralized theme

    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(output_path)
    print(f"Saved sunburst chart: {output_path}")


def create_normalized_bar_plot(df, category_col, color_col, title, filename):
    """
    Creates a normalized stacked bar plot showing percentage distribution of a color_col
    within categories of category_col. Filters for top N categories and min records.
    Args:
        df (pd.DataFrame): The input DataFrame.
        category_col (str): Column for the main categories on the x-axis.
        color_col (str): Column to stack and color the bars by (e.g., 'ekman_emotion').
        title (str): The plot title.
        filename (str): The name of the HTML file to save the plot.
    """
    if df.empty or category_col not in df.columns or color_col not in df.columns:
        print(f"Skipping normalized bar plot for '{filename}': Missing required columns or empty DataFrame.")
        return

    # Filter categories that meet the minimum record threshold
    df_filtered = df.groupby(category_col).filter(lambda x: len(x) >= MIN_RECORDS_FOR_CATEGORY_PLOT)
    if df_filtered.empty:
        print(
            f"Skipping normalized bar plot for '{filename}': No categories meet the min_records threshold ({MIN_RECORDS_FOR_CATEGORY_PLOT}).")
        return

    # Get the top N categories by count
    top_categories = df_filtered[category_col].value_counts().nlargest(TOP_N_CATEGORIES_FOR_PLOTS).index
    df_top = df_filtered[df_filtered[category_col].isin(top_categories)]

    # Calculate percentages for stacking
    df_agg = df_top.groupby([category_col, color_col]).size().reset_index(name='Count')
    df_totals = df_top.groupby(category_col).size().reset_index(name='Total')
    df_agg = pd.merge(df_agg, df_totals, on=category_col)
    df_agg['Percentage'] = (df_agg['Count'] / df_agg['Total']) * 100

    # Create the stacked bar chart
    fig = px.bar(
        df_agg,
        x=category_col,
        y='Percentage',
        color=color_col,
        title=title,
        labels={
            category_col: category_col.replace('_', ' ').title(),
            'Percentage': 'Percentage (%)',
            color_col: color_col.replace('_', ' ').title()
        },
        barmode='stack',  # Stack the bars
        color_discrete_map=EKMAN_EMOTION_COLOR_MAP if color_col == 'ekman_emotion' else SENTIMENT_COLOR_MAP,
        hover_data={'Count': True, 'Total': True, 'Percentage': ':.2f%'}  # Custom hover info
    )

    fig.update_layout(
        yaxis_range=[0, 100],  # Ensure y-axis always goes from 0 to 100 for percentages
        **LAYOUT_THEME  # Apply the centralized theme
    )

    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(output_path)
    print(f"Saved normalized bar plot: {output_path}")


def generate_top_countries_quotes_table(df, emotion_df, top_n=5, top_k_quotes_per_emotion=3,
                                        filename="top_countries_quotes.html"):
    """
    Generates an interactive HTML table showing representative text quotes from top countries,
    filtered by positive/negative Ekman emotions, and selecting the top K quotes per emotion
    based on their confidence scores.
    Args:
        df (pd.DataFrame): The main DataFrame with cleaned data.
        emotion_df (pd.DataFrame): The DataFrame containing emotion analysis results.
        top_n (int): Number of top countries to include.
        top_k_quotes_per_emotion (int): Number of top quotes to retrieve for each emotion within a country.
        filename (str): The name of the HTML file to save the table.
    Returns:
        pd.DataFrame: A DataFrame containing the selected quotes, or None if no data.
    """
    if df.empty or emotion_df.empty:
        print("No data available for top countries quotes analysis.")
        return None

    # Define required columns for merging and analysis
    required_cols_df = ['country_cleaned', 'created_at', 'en_text', 'sentiment']
    required_cols_emotion_df = ['ekman_emotion', 'ekman_emotion_score', 'all_emotion_scores']

    # Validate that all necessary columns exist in the DataFrames
    if not all(col in df.columns for col in required_cols_df) or not all(
            col in emotion_df.columns for col in required_cols_emotion_df):
        print("Missing required columns for top countries quotes analysis.")
        print(f"DF columns: {df.columns.tolist()}")
        print(f"Emotion DF columns: {emotion_df.columns.tolist()}")
        return None

    try:
        # Merge the main DataFrame with the emotion analysis results
        # Resetting index of df to make 'created_at' a regular column for merging by index
        merged_df = pd.merge(
            df.reset_index()[['created_at', 'country_cleaned', 'sentiment', 'en_text']],
            emotion_df[['ekman_emotion', 'ekman_emotion_score', 'all_emotion_scores']],
            left_index=True,  # Merge based on the original index, which is now the default index after reset_index
            right_index=True,  # Merge emotion_df based on its index
            how='inner'  # Only keep records present in both
        )

        # Filter out 'neutral' and 'unknown' Ekman emotions as per requirement
        merged_df = merged_df[~merged_df['ekman_emotion'].isin(['neutral', 'unknown'])].copy()
        if merged_df.empty:
            print("No non-neutral/unknown emotion data after initial filtering for top countries quotes.")
            return None

        # Get the top N countries based on the volume of filtered (non-neutral/unknown) content
        top_countries = merged_df['country_cleaned'].value_counts().head(top_n).index.tolist()

        results = []
        # Iterate through each of the top countries
        for country in top_countries:
            country_data = merged_df[merged_df['country_cleaned'] == country].copy()  # Ensure a copy to avoid warnings

            # Iterate through each unique emotion (excluding neutral/unknown) present in this country's data
            for emotion_label in country_data['ekman_emotion'].unique():
                if emotion_label in ['neutral', 'unknown']:  # Double-check, though already filtered
                    continue

                # Filter data for the current emotion label
                emotion_specific_quotes = country_data[country_data['ekman_emotion'] == emotion_label].copy()

                if not emotion_specific_quotes.empty:
                    # Extract the score for the *specific* emotion_label from 'all_emotion_scores' dictionary
                    # This is crucial for ranking by the confidence of that particular emotion.
                    emotion_specific_quotes['current_emotion_score'] = emotion_specific_quotes[
                        'all_emotion_scores'].apply(
                        lambda x: x.get(emotion_label, 0.0)
                    )

                    # Sort by this specific emotion's score and get the top K quotes
                    top_k_samples = emotion_specific_quotes.sort_values(
                        by='current_emotion_score', ascending=False
                    ).head(top_k_quotes_per_emotion)

                    # Append the selected quotes to the results list
                    for _, row in top_k_samples.iterrows():
                        results.append({
                            'Country': country,
                            'Emotion': row['ekman_emotion'],  # The top predicted Ekman emotion for this text
                            'Emotion Score': f"{row['ekman_emotion_score']:.4f}",  # Score of the top predicted emotion
                            'Sentiment': row['sentiment'],  # Overall sentiment polarity of the text
                            'Date Tweeted': row['created_at'].strftime('%Y-%m-%d %H:%M'),  # Formatted date
                            'Text Quote': f'"{row["en_text"][:150]}{"..." if len(row["en_text"]) > 150 else ""}"'
                            # Truncated text quote
                        })

        # Create a final DataFrame from the collected results
        if not results:
            print("No quotes found after applying all filters for top countries and emotions.")
            return None
        results_df = pd.DataFrame(results)

        # Create an interactive Plotly table
        fig = go.Figure(data=[go.Table(
            columnwidth=[1, 1, 1, 1, 1, 3],  # Adjust column widths for better presentation
            header=dict(
                values=['<b>Country</b>', '<b>Emotion</b>', '<b>Emotion Score</b>', '<b>Sentiment</b>', '<b>Date</b>',
                        '<b>Text Quote</b>'],
                fill_color='#1E88E5',  # Header background color
                font=dict(color='white', size=12),  # Header font style
                align=['left', 'center', 'center', 'center', 'center', 'left']  # Header text alignment
            ),
            cells=dict(
                values=[
                    results_df['Country'],
                    results_df['Emotion'],
                    results_df['Emotion Score'],
                    results_df['Sentiment'],
                    results_df['Date Tweeted'],
                    results_df['Text Quote']
                ],
                fill_color='white',  # Cell background color
                align=['left', 'center', 'center', 'center', 'center', 'left'],  # Cell text alignment
                height=30,  # Row height
                font=dict(size=11)  # Cell font style
            )
        )])

        # Update layout for the table
        fig.update_layout(
            title=f'<b>Top {top_n} Countries - Top {top_k_quotes_per_emotion} Quotes per Emotion (Excluding Neutral)</b><br>'
                  '<i>Showing emotion categorization, sentiment polarity, and actual text quotes based on confidence scores</i>',
            title_font_size=18,
            margin=dict(l=20, r=20, t=100, b=20),
            height=min(1200, 600 + len(results_df) * 30)  # Dynamic height with a maximum limit
        )

        # Save the table as an HTML file
        output_path = os.path.join(OUTPUT_DIR, filename)
        fig.write_html(output_path)
        print(f"Saved top countries quotes table: {output_path}")

        return results_df

    except Exception as e:
        print(f"Error generating top countries quotes table: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return None


def main():
    """Main analysis pipeline to load, clean, analyze, and visualize data."""
    print("\n" + "=" * 70)
    print("Enhanced Sentiment & Emotion Analysis Pipeline")
    print("=" * 70 + "\n")

    # Step 1: Load and clean data
    try:
        df = load_and_clean(FILE_PATH)
        if df.empty:
            raise ValueError("DataFrame is empty after cleaning")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Step 2: Analyze emotions (with caching)
    emotion_df = analyze_emotions(df)
    if emotion_df.empty:
        print("No emotion analysis results - skipping emotion visualizations.")
    else:
        print(f"\nEmotion analysis completed on {len(emotion_df)} records.")
        print("Emotion distribution:")
        print(emotion_df['ekman_emotion'].value_counts())

    # Step 3: Generate visualizations and tables
    print("\nGenerating visualizations and tables...")

    # Enhanced time series plots for Sentiment and Emotion
    if 'created_at' in df.index.name:
        # Sentiment Time Series (full dataset)
        create_enhanced_time_series_plot(
            df,
            'sentiment',
            'Sentiment Polarity Over Time',
            'sentiment_time_series_enhanced.html',
            SENTIMENT_COLOR_MAP
        )

        # Emotion Time Series (filtered to exclude 'neutral' for this plot)
        emotion_df_filtered_for_plots = emotion_df[emotion_df['ekman_emotion'] != 'neutral'].copy()
        if not emotion_df_filtered_for_plots.empty:
            create_enhanced_time_series_plot(
                emotion_df_filtered_for_plots,  # Use filtered df for plotting
                'ekman_emotion',
                'Emotion Types Over Time (Excluding Neutral)',
                'emotion_time_series_enhanced_no_neutral.html',  # New filename for clarity
                EKMAN_EMOTION_COLOR_MAP
            )
        else:
            print("No non-neutral emotion data for time series plot.")

    # Overall Sentiment Distribution Bar Chart
    create_interactive_bar(
        df,
        x_col='sentiment',
        y_col='Count',
        color_col='sentiment',
        title='Overall Sentiment Distribution',
        filename='sentiment_distribution.html'
    )

    # Overall Emotion Distribution Bar Chart (shows all Ekman emotions)
    if not emotion_df.empty:
        create_interactive_bar(
            emotion_df,
            x_col='ekman_emotion',
            y_col='Count',
            color_col='ekman_emotion',
            title='Overall Emotion Distribution (All Ekman Emotions)',
            filename='emotion_distribution_all.html'
        )

        # Normalized emotion plots by category (channel and country, excluding 'neutral')
        if not emotion_df_filtered_for_plots.empty:
            for category in ['channel_cleaned', 'country_cleaned']:
                if category in emotion_df_filtered_for_plots.columns:
                    create_normalized_bar_plot(
                        emotion_df_filtered_for_plots,
                        category_col=category,
                        color_col='ekman_emotion',
                        title=f'Emotion Distribution by {category.replace("_", " ").title()} (Excluding Neutral)',
                        filename=f'emotion_by_{category}_no_neutral.html'  # New filename
                    )
        else:
            print("No non-neutral emotion data for normalized bar plots.")

        # Sunburst charts for hierarchy (excluding 'neutral')
        if not emotion_df_filtered_for_plots.empty and all(col in emotion_df_filtered_for_plots.columns for col in
                                                           ['country_cleaned', 'channel_cleaned', 'ekman_emotion']):
            create_sunburst_chart(
                emotion_df_filtered_for_plots,
                path=['country_cleaned', 'channel_cleaned', 'ekman_emotion'],
                values='Count',
                title='Emotion Hierarchy by Country and Channel (Excluding Neutral)',
                filename='emotion_hierarchy_no_neutral.html'  # New filename
            )
        else:
            print("No non-neutral emotion data for sunburst chart.")

    # Generate the top countries quotes table with specific filtering and ranking
    top_quotes_df = generate_top_countries_quotes_table(df, emotion_df, TOP_COUNTRIES, top_k_quotes_per_emotion=3)
    if top_quotes_df is not None:
        print("\nSample quotes from top countries (excluding neutral emotion, ranked by score):")
        print(top_quotes_df.head(10))  # Print more rows to see diverse quotes

    print("\n" + "=" * 70)
    print("Analysis complete! Check the 'visualizations' folder for outputs.")
    print("=" * 70)


if __name__ == "__main__":
    # Ensure NLTK data is downloaded for tokenizers if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')

    main()
