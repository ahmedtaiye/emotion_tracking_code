# emotion_tracking_code
Emotion distribution for tweets relating to the Russo-Ukrainian War
Download link from /muws-workshop.github.io/cfp/
Make sure you have all the necessary libraries installed 

**N.B:** Python should be installed before running (using any of your prefered IDEs)

**Libraries Used**
**Pandas:** Data manipulation and analysis.
**Transformers:** Natural language processing with pre-trained models.
**Plotly Express:** Simplified data visualization.
**Plotly Graph Objects:** Advanced data visualization capabilities.
**TQDM:** Progress bar for loops and iterations.
**Warnings:** Manage warning messages in the code.
**Regular Expressions (re):** String matching and manipulation.
**CSV:** Handling CSV file reading and writing.
**NLTK:** Natural language processing toolkit.
**Joblib:** Efficiently save and load Python objects.
**OS:** Interact with the operating system.
**Datetime:** Manipulate date and time.
**NumPy:** Numerical operations and array handling.


**Implementation Journey Map**

**1. Initialization and Configuration**

Objective: Set up the environment and configure necessary parameters.
Steps:
Import required libraries.
Define configuration parameters such as file paths, sample sizes, and output directories.
Create the output directory if it does not exist.
Enable caching for emotion analysis and suppress warnings.
Define color maps and themes for consistent visualization aesthetics.
**2. Data Loading and Cleaning**

Objective: Load the dataset, clean it, and prepare it for analysis.
Steps:
Attempt to load the dataset from the specified file path, handling various encodings and delimiters.
Clean the text data by removing NaNs, stripping whitespace, and replacing multiple spaces.
Filter out records with very short or empty text.
Parse and set the datetime column as the index.
Derive a single sentiment category from sentiment score columns if available.
Clean and standardize categorical columns such as 'channel', 'country', and 'verified'.
**3. Emotion Analysis**

Objective: Analyze emotions in a sample of the dataset using a pre-trained model.
Steps:
Sample the dataset based on the specified sample size.
Use the pipeline from transformers to load a pre-trained emotion classification model.
Process the sampled texts in batches to improve performance and manage memory.
Extract the top emotion, its score, and all emotion scores for each text.
Cache the results to avoid reprocessing in future runs.
**4. Visualization Generation**

Objective: Generate various visualizations to analyze and present the data.
Steps:
Time Series Plots:
Create enhanced time series plots for sentiment and emotion over time.
Include interactive controls for category visibility and time aggregation.
Add moving averages to show trends.
Bar Charts:
Generate interactive bar charts for overall sentiment and emotion distribution.
Include sorting options by count or alphabetically.
Normalized Bar Plots:
Create normalized stacked bar plots for emotion distribution by category (e.g., channel, country).
Filter categories based on the minimum number of records and select the top N categories.
Sunburst Charts:
Generate hierarchical sunburst charts for emotion distribution by country and channel.
Exclude 'neutral' emotions for clarity.
**5. Top Countries Quotes Table**

Objective: Generate an interactive HTML table showing representative text quotes from top countries, filtered by positive/negative emotions.
Steps:
Merge the main DataFrame with the emotion analysis results.
Filter out 'neutral' and 'unknown' emotions.
Identify the top N countries based on the volume of content.
For each country, select the top K quotes per emotion based on confidence scores.
Create an interactive Plotly table to display the results.
6. Main Execution

Objective: Execute the entire pipeline and generate all visualizations and tables.
Steps:
Load and clean the dataset.
Perform emotion analysis.
Generate all visualizations and the top countries quotes table.
Save all outputs to the specified directory.
7. Final Output

Objective: Ensure all generated visualizations and tables are saved and accessible.
Steps:
Verify that all visualizations and tables are saved in the output directory.
Print a completion message with the location of the output files.
Summary

This implementation journey map outlines the step-by-step process of loading, cleaning, analyzing, and visualizing data using advanced techniques and tools. Each step is designed to ensure robustness, efficiency, and clarity in the analysis pipeline, ultimately leading to insightful visualizations and interactive tables that provide a comprehensive understanding of the data.

