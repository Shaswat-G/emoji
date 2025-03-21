import pandas as pd
import numpy as np
import os
import re
import ast  # for safely parsing doc_type as a dictionary
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns  # for better visualizations
from wordcloud import WordCloud  # for text analysis
from collections import Counter
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from scipy.stats import pearsonr

# Set aesthetic style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

################################################################################
# 1) Load the Excel Data
################################################################################

working_dir = os.getcwd()
file_name = 'utc_register_all_classified.xlsx'
file_path = os.path.join(working_dir, file_name)

df = pd.read_excel(file_path)
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # convert date column
df.dropna(subset=['date'], inplace=True)                  # remove rows with invalid dates
df['year'] = df['date'].dt.year                           # extract year

# The DataFrame has columns like:
#   doc_num | doc_url | subject | source | date | doc_type | emoji_relevance
# Where doc_type is something like:
#   {"Public Review & Feedback": ["General Feedback & Correspondence"],
#    "Proposals": ["Character Encoding Proposals"]}
# where emoji_relevance is a single category like "Emoji Relevant" or "Irrelevant".

# Create datasets filtered by emoji relevance
df_emoji = df[df['emoji_relevance'] == 'Emoji Relevant']
df_non_emoji = df[df['emoji_relevance'] == 'Irrelevant']

# Print summary statistics about emoji relevance
total_docs = len(df)
emoji_docs = len(df_emoji)
non_emoji_docs = len(df_non_emoji)
emoji_percentage = (emoji_docs / total_docs) * 100 if total_docs > 0 else 0

print(f"Total documents: {total_docs}")
print(f"Emoji-relevant documents: {emoji_docs} ({emoji_percentage:.2f}%)")
print(f"Non-emoji documents: {non_emoji_docs} ({100-emoji_percentage:.2f}%)")

################################################################################
# Function to create plots directory and sanitize filenames
################################################################################

def sanitize_filename(name):
    """Remove or replace characters that are not allowed in filenames."""
    # Replace characters not allowed in filenames with underscores
    return re.sub(r'[\\/*?:"<>|]', '_', name)

# Create plots and data directory structure
plots_dir = os.path.join(working_dir, 'plots')
doc_counts_dir = os.path.join(plots_dir, 'document_counts')
sources_dir = os.path.join(plots_dir, 'sources')
data_dir = os.path.join(working_dir, 'data')

# Create additional directories for emoji analysis
emoji_dir = os.path.join(plots_dir, 'emoji_analysis')
emoji_counts_dir = os.path.join(emoji_dir, 'document_counts')
emoji_sources_dir = os.path.join(emoji_dir, 'sources')
emoji_trends_dir = os.path.join(emoji_dir, 'trends')
emoji_comparative_dir = os.path.join(emoji_dir, 'comparative')

# Create directories if they don't exist
os.makedirs(doc_counts_dir, exist_ok=True)
os.makedirs(sources_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(emoji_dir, exist_ok=True)
os.makedirs(emoji_counts_dir, exist_ok=True)
os.makedirs(emoji_sources_dir, exist_ok=True)
os.makedirs(emoji_trends_dir, exist_ok=True)
os.makedirs(emoji_comparative_dir, exist_ok=True)

# Helper function to process dataframe based on emoji relevance
def process_dataframe_by_relevance(df_input, relevance=None):
    """
    Process a dataframe to expand doc_type into category and subcategory.
    If relevance is specified, filter by emoji_relevance.
    """
    if relevance:
        df_filtered = df_input[df_input['emoji_relevance'] == relevance]
    else:
        df_filtered = df_input.copy()
    
    expanded_rows = []
    
    for idx, row in df_filtered.iterrows():
        doc_num = row['doc_num']
        doc_url = row['doc_url']
        subject = row['subject']
        source = row['source']
        date_  = row['date']
        year   = row['year']
        emoji_relevance = row['emoji_relevance']
        
        try:
            doc_type_dict = row['doc_type']
            if isinstance(doc_type_dict, str):
                doc_type_dict = ast.literal_eval(doc_type_dict)
        except:
            doc_type_dict = {"Others/Miscellaneous": []}

        if not doc_type_dict:
            expanded_rows.append({
                'doc_num': doc_num,
                'doc_url': doc_url,
                'subject': subject,
                'source': source,
                'date': date_,
                'year': year,
                'category': 'Others/Miscellaneous',
                'subcategory': '',
                'emoji_relevance': emoji_relevance
            })
            continue

        for cat, subcats in doc_type_dict.items():
            if not subcats:
                expanded_rows.append({
                    'doc_num': doc_num,
                    'doc_url': doc_url,
                    'subject': subject,
                    'source': source,
                    'date': date_,
                    'year': year,
                    'category': cat,
                    'subcategory': '',
                    'emoji_relevance': emoji_relevance
                })
            else:
                for subcat in subcats:
                    expanded_rows.append({
                        'doc_num': doc_num,
                        'doc_url': doc_url,
                        'subject': subject,
                        'source': source,
                        'date': date_,
                        'year': year,
                        'category': cat,
                        'subcategory': subcat,
                        'emoji_relevance': emoji_relevance
                    })
    
    return pd.DataFrame(expanded_rows)

# Process all dataframes
df_expanded = process_dataframe_by_relevance(df)
df_emoji_expanded = process_dataframe_by_relevance(df, 'Emoji Relevant')
df_non_emoji_expanded = process_dataframe_by_relevance(df, 'Irrelevant')

################################################################################
# 2) Expand doc_type to Handle Multiple Categories/Subcategories
################################################################################

# Some rows may have multiple categories or multiple subcategories in the doc_type dictionary.
# We'll convert each row into one or more rows in a new DataFrame, each with a single category-subcategory.

expanded_rows = []

for idx, row in df.iterrows():
    doc_num = row['doc_num']
    doc_url = row['doc_url']
    subject = row['subject']
    source = row['source']
    date_  = row['date']
    year   = row['year']
    
    # doc_type is a dictionary in string form; parse it safely
    # e.g. "{'Proposals': ['Character Encoding Proposals']}"
    # If doc_type is truly a dictionary already, you can skip ast.literal_eval
    # But usually, from Excel, it may be a string representation.
    try:
        doc_type_dict = row['doc_type']
        # If doc_type is a string, parse it
        if isinstance(doc_type_dict, str):
            doc_type_dict = ast.literal_eval(doc_type_dict)
    except:
        # If something goes wrong, treat it as Others/Misc
        doc_type_dict = {"Others/Miscellaneous": []}

    # If doc_type_dict is empty, treat it as Others/Misc
    if not doc_type_dict:
        expanded_rows.append({
            'doc_num': doc_num,
            'doc_url': doc_url,
            'subject': subject,
            'source': source,
            'date': date_,
            'year': year,
            'category': 'Others/Miscellaneous',
            'subcategory': '',
            'emoji_relevance': row['emoji_relevance']  # Add emoji_relevance column
        })
        continue

    # Otherwise expand all (category -> subcategories) mappings
    for cat, subcats in doc_type_dict.items():
        # If subcats is empty, treat it as a single subcategory of ''
        if not subcats:
            expanded_rows.append({
                'doc_num': doc_num,
                'doc_url': doc_url,
                'subject': subject,
                'source': source,
                'date': date_,
                'year': year,
                'category': cat,
                'subcategory': '',
                'emoji_relevance': row['emoji_relevance']  # Add emoji_relevance column
            })
        else:
            for subcat in subcats:
                expanded_rows.append({
                    'doc_num': doc_num,
                    'doc_url': doc_url,
                    'subject': subject,
                    'source': source,
                    'date': date_,
                    'year': year,
                    'category': cat,
                    'subcategory': subcat,
                    'emoji_relevance': row['emoji_relevance']  # Add emoji_relevance column
                })

# Create the expanded DataFrame
df_expanded = pd.DataFrame(expanded_rows)

################################################################################
# 3) Summary: Counts of Documents per Year by Category & Subcategory
################################################################################

# Create a pivot table: index=year, columns=(category, subcategory), values = count
pivot_counts = df_expanded.pivot_table(
    index='year',
    columns=['category', 'subcategory'],
    aggfunc='size',
    fill_value=0
)

# Sort columns by total count (descending)
column_totals = pivot_counts.sum()
pivot_counts = pivot_counts.loc[:, column_totals.sort_values(ascending=False).index]

print("Document Counts per Year by Category & Subcategory:")
print(pivot_counts)

# Save the pivot table to Excel and CSV
counts_excel_path = os.path.join(data_dir, 'document_counts_by_category.xlsx')
counts_csv_path = os.path.join(data_dir, 'document_counts_by_category.csv')

# Save to Excel (preserves MultiIndex structure better)
pivot_counts.to_excel(counts_excel_path)
# Also save to CSV for easier programmatic use
pivot_counts.to_csv(counts_csv_path)
print(f"Document counts table saved to:\n- {counts_excel_path}\n- {counts_csv_path}")

# Plotting a heatmap with seaborn for better aesthetics
plt.figure(figsize=(16, 10))
ax = sns.heatmap(pivot_counts, cmap='YlOrRd', annot=True, fmt="d", linewidths=.5, 
                cbar_kws={'label': 'Document Count'})
plt.title("Document Counts per Year by Category & Subcategory", fontweight='bold', pad=20)

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, ha='center')
plt.yticks(rotation=0)
plt.xlabel("Category / Subcategory", fontweight='bold')
plt.ylabel("Year", fontweight='bold')
plt.tight_layout()

# Save the plot instead of showing it
doc_counts_filename = os.path.join(doc_counts_dir, "document_counts_heatmap.png")
plt.savefig(doc_counts_filename, dpi=300, bbox_inches='tight')
plt.close()

################################################################################
# 4) Most Important People (Sources) by Year, Category, Subcategory
################################################################################

# The 'source' column can be comma-separated; let's split and explode it
df_expanded['source_list'] = df_expanded['source'].astype(str).str.split(',')
df_sources = df_expanded.explode('source_list')
df_sources['source_list'] = df_sources['source_list'].str.strip()
df_sources.dropna(subset=['source_list'], inplace=True)
df_sources = df_sources[df_sources['source_list'] != '']

# Group by year, category, subcategory, and individual source; count occurrences
grouped_sources = df_sources.groupby(
    ['year', 'category', 'subcategory', 'source_list']
).size().reset_index(name='count')

# Create a summary dataframe for source contributions across categories/subcategories
source_contributions = df_sources.groupby(['source_list', 'category', 'subcategory', 'year']).size().reset_index(name='count')

# Create a pivot table to show source contributions over years by category/subcategory
source_pivot = source_contributions.pivot_table(
    index=['source_list', 'category', 'subcategory'],
    columns='year',
    values='count',
    fill_value=0
)

# Add a total column and sort by it in descending order
source_pivot['Total'] = source_pivot.sum(axis=1)
source_pivot = source_pivot.sort_values(by='Total', ascending=False)
# Drop the Total column after sorting
source_pivot = source_pivot.drop(columns=['Total'])

# Save source contribution data
source_contrib_excel = os.path.join(data_dir, 'source_contributions.xlsx')
source_contrib_csv = os.path.join(data_dir, 'source_contributions.csv')

# Save to Excel (preserves MultiIndex structure better)
source_pivot.to_excel(source_contrib_excel)
# Also save to CSV
source_pivot.to_csv(source_contrib_csv)
print(f"Source contributions data saved to:\n- {source_contrib_excel}\n- {source_contrib_csv}")

# Create a more detailed analysis: each source's contribution by doc type over time
# Group by source and get their category/subcategory distribution
source_category_dist = df_sources.groupby(['source_list', 'category', 'subcategory']).size().reset_index(name='count')
source_category_dist['percentage'] = source_category_dist.groupby('source_list')['count'].transform(lambda x: (x / x.sum()) * 100)

# Save this detailed analysis
source_category_excel = os.path.join(data_dir, 'source_category_distribution.xlsx')
source_category_dist.to_excel(source_category_excel, index=False)
print(f"Source category distribution saved to:\n- {source_category_excel}")

# For each (category, subcategory) pair, produce a heatmap of the top N sources
top_N = 10

unique_pairs = grouped_sources[['category', 'subcategory']].drop_duplicates().values

for cat, subcat in unique_pairs:
    subset = grouped_sources[
        (grouped_sources['category'] == cat) &
        (grouped_sources['subcategory'] == subcat)
    ]
    if subset.empty:
        continue

    # Create pivot: rows=year, columns=source_list, values=count
    pivot_src = subset.pivot_table(
        index='year',
        columns='source_list',
        values='count',
        aggfunc='sum',
        fill_value=0
    )

    # Keep only top N sources by total count
    overall_counts = pivot_src.sum(axis=0).sort_values(ascending=False)
    top_sources = overall_counts.head(top_N).index
    pivot_src = pivot_src[top_sources]

    if pivot_src.empty:
        continue

    # Plot an enhanced heatmap using seaborn
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(pivot_src, annot=True, fmt="d", cmap='YlGnBu', 
                     linewidths=.5, cbar_kws={'label': 'Count'})
    plt.title(f"Top Contributors for: {cat} - {subcat}", fontweight='bold', pad=20)
    plt.xlabel("Source", fontweight='bold')
    plt.ylabel("Year", fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Create a sanitized filename based on category and subcategory
    cat_clean = sanitize_filename(cat)
    subcat_clean = sanitize_filename(subcat) if subcat else "general"
    
    # Save the plot image
    source_filename = os.path.join(
        sources_dir,
        f"sources_{cat_clean}_{subcat_clean}.png"
    )
    plt.savefig(source_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the pivot data for this category/subcategory
    source_cat_data_path = os.path.join(
        data_dir,
        f"sources_data_{cat_clean}_{subcat_clean}.xlsx"
    )
    pivot_src.to_excel(source_cat_data_path)

################################################################################
# 5) NEW: Trend Analysis - Growth/Decline in Document Types Over Time
################################################################################

# Create trend analysis directory
trends_dir = os.path.join(plots_dir, 'trends')
os.makedirs(trends_dir, exist_ok=True)

# Aggregate by year and category
yearly_category_counts = df_expanded.groupby(['year', 'category']).size().unstack(fill_value=0)

# Sort categories by total count (descending)
category_totals = yearly_category_counts.sum().sort_values(ascending=False)
yearly_category_counts = yearly_category_counts[category_totals.index]

# Plot line chart for trends
plt.figure(figsize=(14, 8))
ax = yearly_category_counts.plot(kind='line', marker='o', linewidth=2.5, markersize=8, ax=plt.gca())
plt.title('Document Category Trends Over Time', fontweight='bold', pad=20)
plt.xlabel('Year', fontweight='bold')
plt.ylabel('Number of Documents', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Category', title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

# Add data point labels
for line in ax.lines:
    y_data = line.get_ydata()
    x_data = line.get_xdata()
    for x, y in zip(x_data, y_data):
        if y > 0:  # Only label non-zero values
            ax.annotate(f'{int(y)}', (x, y), textcoords="offset points", 
                        xytext=(0, 5), ha='center', fontsize=9)

plt.tight_layout()
trend_file = os.path.join(trends_dir, 'category_trends_over_time.png')
plt.savefig(trend_file, dpi=300, bbox_inches='tight')
plt.close()

################################################################################
# 6) NEW: Subject Text Analysis
################################################################################

# Create text analysis directory
text_dir = os.path.join(plots_dir, 'text_analysis')
os.makedirs(text_dir, exist_ok=True)

# Word cloud for all subjects
all_subjects = ' '.join(df['subject'].astype(str).dropna())
common_words = ["UTC", "Unicode", "Document", "L2", "WG2", "ISO", "IEC"]

# Generate word cloud
wordcloud = WordCloud(
    width=800, height=400,
    background_color='white',
    colormap='viridis',
    max_words=100,
    contour_width=1,
    contour_color='steelblue'
).generate(all_subjects)

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Document Subjects', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
wordcloud_file = os.path.join(text_dir, 'subject_wordcloud.png')
plt.savefig(wordcloud_file, dpi=300, bbox_inches='tight')
plt.close()

# Word frequency analysis by category
for category in df_expanded['category'].unique():
    category_subjects = ' '.join(df_expanded[df_expanded['category'] == category]['subject'].astype(str).dropna())
    if len(category_subjects) > 50:  # Only process if enough text
        category_wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='plasma',
            max_words=75,
            contour_width=1,
            contour_color='steelblue'
        ).generate(category_subjects)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(category_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Category: {category}', fontweight='bold', fontsize=16, pad=20)
        plt.tight_layout()
        
        cat_clean = sanitize_filename(category)
        wordcloud_cat_file = os.path.join(text_dir, f'wordcloud_{cat_clean}.png')
        plt.savefig(wordcloud_cat_file, dpi=300, bbox_inches='tight')
        plt.close()

################################################################################
# 7) NEW: Time Series Analysis - Monthly/Quarterly Patterns
################################################################################

# Create time series directory
timeseries_dir = os.path.join(plots_dir, 'timeseries')
os.makedirs(timeseries_dir, exist_ok=True)

# Add month and quarter columns
df_expanded['month'] = df_expanded['date'].dt.month
df_expanded['quarter'] = df_expanded['date'].dt.quarter

# Analysis by month
monthly_counts = df_expanded.groupby(['year', 'month']).size().unstack(fill_value=0)

plt.figure(figsize=(15, 8))
sns.heatmap(monthly_counts, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5)
plt.title('Document Submissions by Month and Year', fontweight='bold', pad=20)
plt.xlabel('Month', fontweight='bold')
plt.ylabel('Year', fontweight='bold')
plt.tight_layout()

monthly_file = os.path.join(timeseries_dir, 'monthly_submission_heatmap.png')
plt.savefig(monthly_file, dpi=300, bbox_inches='tight')
plt.close()

# Analysis by quarter - stacked bar chart
quarterly_category_counts = df_expanded.groupby(['year', 'quarter', 'category']).size().unstack(fill_value=0)
quarterly_counts_pivot = quarterly_category_counts.groupby(level=[0, 1]).sum()  # Sum by year-quarter

# Reshape for plotting
quarterly_counts_reset = quarterly_counts_pivot.reset_index()
quarterly_counts_reset['year_quarter'] = quarterly_counts_reset['year'].astype(str) + "-Q" + quarterly_counts_reset['quarter'].astype(str)

# Pivot for plotting
plot_data = quarterly_counts_reset.melt(
    id_vars=['year_quarter', 'year', 'quarter'],
    var_name='category',
    value_name='count'
)

# Sort by year and quarter
plot_data = plot_data.sort_values(['year', 'quarter'])

# Get top 5 categories by total count
top_categories = plot_data.groupby('category')['count'].sum().nlargest(5).index.tolist()
plot_data_top = plot_data[plot_data['category'].isin(top_categories)]

plt.figure(figsize=(16, 8))
chart = sns.barplot(
    x='year_quarter', 
    y='count', 
    hue='category', 
    data=plot_data_top,
    palette='viridis'
)
plt.title('Quarterly Document Submissions by Top 5 Categories', fontweight='bold', pad=20)
plt.xlabel('Year-Quarter', fontweight='bold')
plt.ylabel('Number of Documents', fontweight='bold')
plt.xticks(rotation=90)
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

quarterly_file = os.path.join(timeseries_dir, 'quarterly_category_submissions.png')
plt.savefig(quarterly_file, dpi=300, bbox_inches='tight')
plt.close()

################################################################################
# 8) NEW: Correlation Analysis Between Categories and Sources
################################################################################

# Create correlation directory
corr_dir = os.path.join(plots_dir, 'correlations')
os.makedirs(corr_dir, exist_ok=True)

# Get top 10 sources by contribution count
source_counts = df_sources.groupby('source_list').size().nlargest(10)
top_sources = source_counts.index.tolist()

# Get top 10 categories by document count
category_counts = df_expanded.groupby('category').size().nlargest(10)
top_categories = category_counts.index.tolist()

# Filter the data to top sources and categories
filtered_df = df_sources[
    (df_sources['source_list'].isin(top_sources)) & 
    (df_sources['category'].isin(top_categories))
]

# Create source-category counts using groupby and pivot instead of crosstab
source_category_counts = filtered_df.groupby(['source_list', 'category']).size().reset_index(name='count')
crosstab = source_category_counts.pivot(
    index='source_list',
    columns='category',
    values='count'
)

# Fill any missing values with 0
crosstab = crosstab.fillna(0)

# Create correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(crosstab, cmap='YlGnBu', annot=True, fmt='g', linewidths=.5)
plt.title('Contributions: Top Sources vs Top Categories', fontweight='bold', pad=20)
plt.xlabel('Category', fontweight='bold')
plt.ylabel('Source', fontweight='bold')
plt.tight_layout()

corr_file = os.path.join(corr_dir, 'source_category_correlation.png')
plt.savefig(corr_file, dpi=300, bbox_inches='tight')
plt.close()

# Print information about saved plots and data
print(f"Plots saved to directory: {plots_dir}")
print(f"Data files saved to directory: {data_dir}")
print(f"Document counts heatmap: {doc_counts_filename}")
print(f"Source heatmaps: {len(unique_pairs)} plots saved in {sources_dir}")
print(f"Trend analysis plots saved in: {trends_dir}")
print(f"Text analysis plots saved in: {text_dir}")
print(f"Time series plots saved in: {timeseries_dir}")
print(f"Correlation plots saved in: {corr_dir}")

################################################################################
# NEW: Emoji Relevance Analysis
################################################################################

# 1. Emoji Documents Distribution by Year
emoji_by_year = df_expanded.groupby(['year', 'emoji_relevance']).size().unstack(fill_value=0)
if 'Emoji Relevant' not in emoji_by_year.columns:
    emoji_by_year['Emoji Relevant'] = 0
if 'Irrelevant' not in emoji_by_year.columns:
    emoji_by_year['Irrelevant'] = 0

# Calculate percentages
emoji_by_year['Total'] = emoji_by_year.sum(axis=1)
emoji_by_year['Emoji %'] = (emoji_by_year['Emoji Relevant'] / emoji_by_year['Total'] * 100).round(1)

# Save the data
emoji_year_excel = os.path.join(data_dir, 'emoji_distribution_by_year.xlsx')
emoji_by_year.to_excel(emoji_year_excel)
print(f"Emoji distribution by year saved to: {emoji_year_excel}")

# Plot the emoji distribution by year (stacked bar chart)
plt.figure(figsize=(14, 8))
ax = emoji_by_year[['Emoji Relevant', 'Irrelevant']].plot(
    kind='bar', 
    stacked=True, 
    colormap='viridis',
    ax=plt.gca()
)
plt.title('Document Distribution by Emoji Relevance and Year', fontweight='bold', pad=20)
plt.xlabel('Year', fontweight='bold')
plt.ylabel('Number of Documents', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Emoji Relevance')

# Add percentage labels on top of each bar
for i, year in enumerate(emoji_by_year.index):
    total = emoji_by_year.loc[year, 'Total']
    emoji_count = emoji_by_year.loc[year, 'Emoji Relevant']
    emoji_pct = emoji_by_year.loc[year, 'Emoji %']
    if total > 0:
        plt.annotate(f"{emoji_pct}%",
                    xy=(i, total + 5),
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    color='darkred')

plt.tight_layout()
emoji_year_file = os.path.join(emoji_comparative_dir, 'emoji_distribution_by_year.png')
plt.savefig(emoji_year_file, dpi=300, bbox_inches='tight')
plt.close()

# 2. Document Category Distribution by Emoji Relevance
# Create pivot tables for both emoji and non-emoji documents
emoji_category_counts = df_emoji_expanded.groupby(['category']).size().reset_index(name='count')
non_emoji_category_counts = df_non_emoji_expanded.groupby(['category']).size().reset_index(name='count')

# Sort by count for better visualization
emoji_category_counts = emoji_category_counts.sort_values('count', ascending=False)
non_emoji_category_counts = non_emoji_category_counts.sort_values('count', ascending=False)

# Save the data
emoji_cat_excel = os.path.join(data_dir, 'emoji_category_distribution.xlsx')
emoji_category_counts.to_excel(emoji_cat_excel, index=False)
print(f"Emoji category distribution saved to: {emoji_cat_excel}")

# Plot the top 10 categories for emoji-relevant documents
plt.figure(figsize=(14, 8))
ax = sns.barplot(
    x='count',
    y='category',
    data=emoji_category_counts.head(10),
    hue='category',  # Add hue parameter
    palette='viridis',
    legend=False  # Hide the legend since it would be redundant
)
plt.title('Top 10 Categories for Emoji-Relevant Documents', fontweight='bold', pad=20)
plt.xlabel('Number of Documents', fontweight='bold')
plt.ylabel('Category', fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add count labels
for i, v in enumerate(emoji_category_counts.head(10)['count']):
    ax.text(v + 0.5, i, str(v), va='center')

plt.tight_layout()
emoji_cat_file = os.path.join(emoji_counts_dir, 'emoji_top_categories.png')
plt.savefig(emoji_cat_file, dpi=300, bbox_inches='tight')
plt.close()

# 3. Emoji Category Trends Over Time
# Analyze trends for emoji-relevant documents by category and year
emoji_cat_year = df_emoji_expanded.groupby(['year', 'category']).size().unstack(fill_value=0)

# Get the top 5 categories for emoji documents
top_emoji_cats = emoji_category_counts.head(5)['category'].tolist()
emoji_cat_year_filtered = emoji_cat_year[top_emoji_cats] if set(top_emoji_cats).issubset(set(emoji_cat_year.columns)) else emoji_cat_year

# Plot the trends
plt.figure(figsize=(14, 8))
ax = emoji_cat_year_filtered.plot(
    kind='line',
    marker='o',
    linewidth=2.5,
    markersize=8,
    ax=plt.gca()
)
plt.title('Emoji-Relevant Document Trends by Category', fontweight='bold', pad=20)
plt.xlabel('Year', fontweight='bold')
plt.ylabel('Number of Documents', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Category', title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

# Add data point labels
for line in ax.lines:
    y_data = line.get_ydata()
    x_data = line.get_xdata()
    for x, y in zip(x_data, y_data):
        if y > 0:  # Only label non-zero values
            ax.annotate(f'{int(y)}', (x, y), textcoords="offset points",
                        xytext=(0, 5), ha='center', fontsize=9)

plt.tight_layout()
emoji_trend_file = os.path.join(emoji_trends_dir, 'emoji_category_trends.png')
plt.savefig(emoji_trend_file, dpi=300, bbox_inches='tight')
plt.close()

# 4. Comparative Analysis: Emoji vs. Non-Emoji Sources
# Find top contributors for emoji documents
emoji_sources = df_emoji_expanded['source'].astype(str).str.split(',')
emoji_sources = emoji_sources.explode().str.strip()
emoji_sources = emoji_sources[emoji_sources != '']
emoji_source_counts = emoji_sources.value_counts().reset_index()
emoji_source_counts.columns = ['source', 'count']
emoji_source_counts = emoji_source_counts.sort_values('count', ascending=False)

# Save the data
emoji_source_excel = os.path.join(data_dir, 'emoji_source_distribution.xlsx')
emoji_source_counts.to_excel(emoji_source_excel, index=False)
print(f"Emoji source distribution saved to: {emoji_source_excel}")

# Plot the top contributors for emoji documents
plt.figure(figsize=(14, 8))
ax = sns.barplot(
    x='count',
    y='source',
    data=emoji_source_counts.head(10),
    hue='source',  # Add hue parameter
    palette='plasma',
    legend=False  # Hide the legend since it would be redundant
)
plt.title('Top 10 Contributors to Emoji-Relevant Documents', fontweight='bold', pad=20)
plt.xlabel('Number of Contributions', fontweight='bold')
plt.ylabel('Contributor', fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add count labels
for i, v in enumerate(emoji_source_counts.head(10)['count']):
    ax.text(v + 0.5, i, str(v), va='center')

plt.tight_layout()
emoji_source_file = os.path.join(emoji_sources_dir, 'emoji_top_contributors.png')
plt.savefig(emoji_source_file, dpi=300, bbox_inches='tight')
plt.close()

# 5. Compare emoji vs non-emoji document types using a side-by-side bar chart
# Get the data for the most common categories in both
top_categories = pd.concat([
    emoji_category_counts.head(5).assign(type='Emoji Relevant'),
    non_emoji_category_counts.head(5).assign(type='Irrelevant')
])

# Plot side-by-side comparison
plt.figure(figsize=(16, 10))
ax = sns.barplot(
    x='category',
    y='count',
    hue='type',
    data=top_categories,
    palette=['#1f77b4', '#ff7f0e']
)
plt.title('Top Categories: Emoji vs. Non-Emoji Documents', fontweight='bold', pad=20)
plt.xlabel('Category', fontweight='bold')
plt.ylabel('Number of Documents', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Document Type')

# Add count labels
for container in ax.containers:
    ax.bar_label(container, fmt='%d')

plt.tight_layout()
comparison_file = os.path.join(emoji_comparative_dir, 'emoji_vs_non_emoji_categories.png')
plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
plt.close()

# 6. Word Cloud comparison for emoji vs non-emoji subjects
# Word cloud for emoji subjects
emoji_subjects = ' '.join(df_emoji['subject'].astype(str).dropna())

if len(emoji_subjects) > 50:
    emoji_wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(emoji_subjects)

    plt.figure(figsize=(16, 8))
    plt.imshow(emoji_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Emoji-Relevant Document Subjects', fontweight='bold', fontsize=16, pad=20)
    plt.tight_layout()
    emoji_wordcloud_file = os.path.join(emoji_dir, 'emoji_subject_wordcloud.png')
    plt.savefig(emoji_wordcloud_file, dpi=300, bbox_inches='tight')
    plt.close()

# 7. Emoji relevance by time period - quarterly analysis
emoji_quarterly = df_expanded.groupby(['year', 'quarter', 'emoji_relevance']).size().unstack(fill_value=0)
if 'Emoji Relevant' not in emoji_quarterly.columns:
    emoji_quarterly['Emoji Relevant'] = 0
if 'Irrelevant' not in emoji_quarterly.columns:
    emoji_quarterly['Irrelevant'] = 0

emoji_quarterly['Total'] = emoji_quarterly.sum(axis=1)
emoji_quarterly['Emoji %'] = (emoji_quarterly['Emoji Relevant'] / emoji_quarterly['Total'] * 100).round(1)
emoji_quarterly = emoji_quarterly.reset_index()
emoji_quarterly['year_quarter'] = emoji_quarterly['year'].astype(str) + '-Q' + emoji_quarterly['quarter'].astype(str)

# Create a line chart showing the percentage of emoji-relevant documents over time
plt.figure(figsize=(16, 8))
ax = sns.lineplot(
    x='year_quarter',
    y='Emoji %',
    data=emoji_quarterly,
    marker='o',
    markersize=8,
    linewidth=2
)
plt.title('Percentage of Emoji-Relevant Documents by Quarter', fontweight='bold', pad=20)
plt.xlabel('Year-Quarter', fontweight='bold')
plt.ylabel('Emoji-Relevant Documents (%)', fontweight='bold')
plt.xticks(rotation=90)
plt.grid(True, linestyle='--', alpha=0.7)

# Add percentage labels
for i, row in emoji_quarterly.iterrows():
    if not np.isnan(row['Emoji %']):
        ax.annotate(f"{row['Emoji %']}%",
                   xy=(i, row['Emoji %']),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9)

plt.tight_layout()
emoji_quarterly_file = os.path.join(emoji_trends_dir, 'emoji_percentage_by_quarter.png')
plt.savefig(emoji_quarterly_file, dpi=300, bbox_inches='tight')
plt.close()

################################################################################ 
# 8. NEW: Additional Emoji Analysis - Subcategory Analysis
################################################################################

# Create a directory for subcategory analysis
emoji_subcats_dir = os.path.join(emoji_dir, 'subcategories')
os.makedirs(emoji_subcats_dir, exist_ok=True)

# Analyze emoji-relevant documents by subcategory
emoji_subcat_counts = df_emoji_expanded.groupby(['category', 'subcategory']).size().reset_index(name='count')
emoji_subcat_counts = emoji_subcat_counts.sort_values(['category', 'count'], ascending=[True, False])

# Save subcategory data
emoji_subcat_excel = os.path.join(data_dir, 'emoji_subcategory_distribution.xlsx')
emoji_subcat_counts.to_excel(emoji_subcat_excel, index=False)
print(f"Emoji subcategory distribution saved to: {emoji_subcat_excel}")

# For each major category with emoji documents, create a visualization of its subcategories
for category in emoji_category_counts.head(10)['category']:
    cat_subcats = emoji_subcat_counts[emoji_subcat_counts['category'] == category]
    
    if len(cat_subcats) > 0:
        plt.figure(figsize=(14, max(6, len(cat_subcats) * 0.4)))
        ax = sns.barplot(
            x='count',
            y='subcategory',
            data=cat_subcats,
            hue='subcategory',  # Add hue parameter
            palette='viridis',
            legend=False  # Hide the legend since it would be redundant
        )
        plt.title(f'Emoji-Relevant Documents: Subcategories of {category}', fontweight='bold', pad=20)
        plt.xlabel('Number of Documents', fontweight='bold')
        plt.ylabel('Subcategory', fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add count labels
        for i, v in enumerate(cat_subcats['count']):
            ax.text(v + 0.5, i, str(v), va='center')
        
        plt.tight_layout()
        cat_clean = sanitize_filename(category)
        subcat_file = os.path.join(emoji_subcats_dir, f'emoji_subcats_{cat_clean}.png')
        plt.savefig(subcat_file, dpi=300, bbox_inches='tight')
        plt.close()

################################################################################
# 9. NEW: Emoji Document Timeline Analysis
################################################################################

# Create timeline directory
emoji_timeline_dir = os.path.join(emoji_dir, 'timeline')
os.makedirs(emoji_timeline_dir, exist_ok=True)

# Sort data chronologically
df_emoji_sorted = df_emoji.sort_values('date')
df_emoji_sorted['cumulative_count'] = range(1, len(df_emoji_sorted) + 1)

# Plot cumulative emoji document submissions over time
plt.figure(figsize=(16, 8))
ax = plt.plot(
    df_emoji_sorted['date'],
    df_emoji_sorted['cumulative_count'],
    marker='.',
    linestyle='-',
    color='#1f77b4',
    linewidth=2
)
plt.title('Cumulative Emoji-Relevant Document Submissions Over Time', fontweight='bold', pad=20)
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Cumulative Number of Documents', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)

# Format x-axis date labels
plt.gcf().autofmt_xdate()
date_format = mdates.DateFormatter('%Y-%m')
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
timeline_file = os.path.join(emoji_timeline_dir, 'emoji_document_timeline.png')
plt.savefig(timeline_file, dpi=300, bbox_inches='tight')
plt.close()

################################################################################
# 10. NEW: Emoji vs Non-Emoji Source Overlap Analysis
################################################################################

# Create overlap directory
emoji_overlap_dir = os.path.join(emoji_dir, 'overlap_analysis')
os.makedirs(emoji_overlap_dir, exist_ok=True)

# Get sources for emoji documents
emoji_sources_set = set(emoji_sources.unique())

# Get sources for non-emoji documents
non_emoji_sources = df_non_emoji_expanded['source'].astype(str).str.split(',')
non_emoji_sources = non_emoji_sources.explode().str.strip()
non_emoji_sources = non_emoji_sources[non_emoji_sources != '']
non_emoji_sources_set = set(non_emoji_sources.unique())

# Find overlap
sources_overlap = emoji_sources_set.intersection(non_emoji_sources_set)
emoji_only_sources = emoji_sources_set - non_emoji_sources_set
non_emoji_only_sources = non_emoji_sources_set - emoji_sources_set

# Create and save a summary
source_overlap_data = pd.DataFrame({
    'Category': ['Emoji Only', 'Both Emoji & Non-Emoji', 'Non-Emoji Only'],
    'Count': [len(emoji_only_sources), len(sources_overlap), len(non_emoji_only_sources)]
})

overlap_excel = os.path.join(data_dir, 'source_overlap_analysis.xlsx')
source_overlap_data.to_excel(overlap_excel, index=False)
print(f"Source overlap analysis saved to: {overlap_excel}")

# Visualize source overlap with a Venn-like diagram (pie chart)
plt.figure(figsize=(14, 8))
ax = plt.pie(
    source_overlap_data['Count'],
    labels=source_overlap_data['Category'],
    autopct='%1.1f%%',
    colors=['#ff9999', '#66b3ff', '#99ff99'],
    startangle=90,
    shadow=True,
    explode=(0.1, 0.1, 0.1)
)
plt.title('Distribution of Sources Between Emoji and Non-Emoji Documents', fontweight='bold', pad=20)

plt.tight_layout()
overlap_file = os.path.join(emoji_overlap_dir, 'source_category_overlap.png')
plt.savefig(overlap_file, dpi=300, bbox_inches='tight')
plt.close()

################################################################################
# 11. NEW: Emoji Rate Analysis by Year
################################################################################

# Calculate the rate of emoji document submissions per year
emoji_rates = pd.DataFrame({
    'Year': emoji_by_year.index,
    'Emoji_Count': emoji_by_year['Emoji Relevant'],
    'Total_Count': emoji_by_year['Total'],
    'Percentage': emoji_by_year['Emoji %']
})

# Calculate year-over-year change in emoji document percentage
emoji_rates['YoY_Change'] = emoji_rates['Percentage'].diff()

# Save the analysis
emoji_rates_excel = os.path.join(data_dir, 'emoji_yearly_rates.xlsx')
emoji_rates.to_excel(emoji_rates_excel, index=False)
print(f"Emoji yearly rates saved to: {emoji_rates_excel}")

# Create a combo chart: bars for counts, line for percentage
fig, ax1 = plt.subplots(figsize=(16, 8))

# Plot bars for emoji and non-emoji counts
x = np.arange(len(emoji_rates))
width = 0.35
emoji_bars = ax1.bar(x - width/2, emoji_rates['Emoji_Count'], width, label='Emoji Documents', color='#1f77b4')
non_emoji_bars = ax1.bar(x + width/2, emoji_rates['Total_Count'] - emoji_rates['Emoji_Count'], width, 
                         label='Non-Emoji Documents', color='#ff7f0e')

ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Number of Documents', fontweight='bold')
ax1.set_title('Emoji Document Counts and Percentage by Year', fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(emoji_rates['Year'])
ax1.legend(loc='upper left')

# Create second y-axis for percentage
ax2 = ax1.twinx()
emoji_line = ax2.plot(x, emoji_rates['Percentage'], linestyle='-', marker='o', 
                     linewidth=2, color='red', label='Emoji %')
ax2.set_ylabel('Emoji Documents (%)', color='red', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='red')

# Add a separate legend for the percentage line
lines, labels = ax2.get_legend_handles_labels()
ax2.legend(lines, labels, loc='upper right')

# Add data labels
for i, v in enumerate(emoji_rates['Percentage']):
    if not np.isnan(v):
        ax2.annotate(f"{v:.1f}%",
                    xy=(i, v),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    color='red',
                    fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
rates_file = os.path.join(emoji_trends_dir, 'emoji_rates_by_year.png')
plt.savefig(rates_file, dpi=300, bbox_inches='tight')
plt.close()

################################################################################
# 12. NEW: Emoji Document Submission Patterns by Month
################################################################################

# Add month analysis for emoji documents
df_emoji_expanded['month'] = df_emoji_expanded['date'].dt.month
df_non_emoji_expanded['month'] = df_non_emoji_expanded['date'].dt.month

df_emoji_expanded['quarter'] = df_emoji_expanded['date'].dt.quarter
df_non_emoji_expanded['quarter'] = df_non_emoji_expanded['date'].dt.quarter

# Compare monthly patterns of emoji vs non-emoji documents
emoji_monthly = df_emoji_expanded.groupby('month').size()
non_emoji_monthly = df_non_emoji_expanded.groupby('month').size()

# Calculate percentages to normalize the data
emoji_monthly_pct = (emoji_monthly / emoji_monthly.sum() * 100).round(1)
non_emoji_monthly_pct = (non_emoji_monthly / non_emoji_monthly.sum() * 100).round(1)

# Combine into a dataframe
monthly_pattern = pd.DataFrame({
    'Emoji Count': emoji_monthly,
    'Emoji %': emoji_monthly_pct,
    'Non-Emoji Count': non_emoji_monthly,
    'Non-Emoji %': non_emoji_monthly_pct
})

# Save the data
monthly_excel = os.path.join(data_dir, 'emoji_monthly_patterns.xlsx')
monthly_pattern.to_excel(monthly_excel)
print(f"Emoji monthly patterns saved to: {monthly_excel}")

# Create a bar chart comparing percentages by month
plt.figure(figsize=(14, 8))
monthly_pattern_plot = pd.DataFrame({
    'Emoji-Relevant': emoji_monthly_pct,
    'Irrelevant': non_emoji_monthly_pct
})

ax = monthly_pattern_plot.plot(
    kind='bar',
    width=0.8,
    color=['#1f77b4', '#ff7f0e']
)
plt.title('Monthly Submission Patterns: Emoji vs. Non-Emoji Documents', fontweight='bold', pad=20)
plt.xlabel('Month', fontweight='bold')
plt.ylabel('Percentage of Documents (%)', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Document Type')

# Format x-axis to show month names
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(range(12), month_names)

# Add percentage labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%')

plt.tight_layout()
monthly_pattern_file = os.path.join(emoji_trends_dir, 'emoji_monthly_patterns.png')
plt.savefig(monthly_pattern_file, dpi=300, bbox_inches='tight')
plt.close()

# Print information about the additional emoji analysis
print("\nExpanded Emoji Analysis Summary:")
print(f"Emoji subcategory analysis: {emoji_subcats_dir}")
print(f"Emoji timeline analysis: {timeline_file}")
print(f"Source overlap analysis: {overlap_file}")
print(f"Emoji submission rates: {rates_file}")
print(f"Monthly submission patterns: {monthly_pattern_file}")

################################################################################
# 13. Final Analysis and Comparative Summary
################################################################################

# Create a final summary directory
summary_dir = os.path.join(plots_dir, 'summary')
os.makedirs(summary_dir, exist_ok=True)

# Create a comprehensive summary table of emoji vs non-emoji analysis
summary_data = {
    'Metric': [
        'Total Documents',
        'Documents by Category (Top Category)',
        'Documents by Subcategory (Top Subcategory)',
        'Top Contributor',
        'Documents per Year (Average)',
        'Monthly Peak (Month with most documents)',
        'Quarterly Peak (Quarter with most documents)'
    ]
}

# Calculate metrics for emoji documents
emoji_top_category = emoji_category_counts.iloc[0]['category'] if not emoji_category_counts.empty else 'N/A'
emoji_top_category_count = emoji_category_counts.iloc[0]['count'] if not emoji_category_counts.empty else 0

emoji_top_subcategory = emoji_subcat_counts.iloc[0]['subcategory'] if not emoji_subcat_counts.empty else 'N/A'
emoji_top_subcategory_count = emoji_subcat_counts.iloc[0]['count'] if not emoji_subcat_counts.empty else 0

emoji_top_contributor = emoji_source_counts.iloc[0]['source'] if not emoji_source_counts.empty else 'N/A'
emoji_top_contributor_count = emoji_source_counts.iloc[0]['count'] if not emoji_source_counts.empty else 0

emoji_docs_per_year = len(df_emoji) / len(emoji_by_year) if len(emoji_by_year) > 0 else 0

emoji_monthly_peak = emoji_monthly.idxmax() if not emoji_monthly.empty else 0
emoji_monthly_peak_value = emoji_monthly.max() if not emoji_monthly.empty else 0
emoji_monthly_peak_month = month_names[emoji_monthly_peak-1] if not emoji_monthly.empty else 'N/A'

emoji_quarterly = df_emoji_expanded.groupby('quarter').size()
emoji_quarterly_peak = emoji_quarterly.idxmax() if not emoji_quarterly.empty else 0
emoji_quarterly_peak_value = emoji_quarterly.max() if not emoji_quarterly.empty else 0

# Calculate metrics for non-emoji documents
non_emoji_top_category = non_emoji_category_counts.iloc[0]['category'] if not non_emoji_category_counts.empty else 'N/A'
non_emoji_top_category_count = non_emoji_category_counts.iloc[0]['count'] if not non_emoji_category_counts.empty else 0

non_emoji_subcat_counts = df_non_emoji_expanded.groupby(['category', 'subcategory']).size().reset_index(name='count')
non_emoji_subcat_counts = non_emoji_subcat_counts.sort_values('count', ascending=False)
non_emoji_top_subcategory = non_emoji_subcat_counts.iloc[0]['subcategory'] if not non_emoji_subcat_counts.empty else 'N/A'
non_emoji_top_subcategory_count = non_emoji_subcat_counts.iloc[0]['count'] if not non_emoji_subcat_counts.empty else 0

non_emoji_sources = df_non_emoji_expanded['source'].astype(str).str.split(',')
non_emoji_sources = non_emoji_sources.explode().str.strip()
non_emoji_sources = non_emoji_sources[non_emoji_sources != '']
non_emoji_source_counts = non_emoji_sources.value_counts().reset_index()
non_emoji_source_counts.columns = ['source', 'count']
non_emoji_top_contributor = non_emoji_source_counts.iloc[0]['source'] if not non_emoji_source_counts.empty else 'N/A'
non_emoji_top_contributor_count = non_emoji_source_counts.iloc[0]['count'] if not non_emoji_source_counts.empty else 0

non_emoji_docs_per_year = len(df_non_emoji) / len(emoji_by_year) if len(emoji_by_year) > 0 else 0

non_emoji_monthly_peak = non_emoji_monthly.idxmax() if not non_emoji_monthly.empty else 0
non_emoji_monthly_peak_value = non_emoji_monthly.max() if not non_emoji_monthly.empty else 0
non_emoji_monthly_peak_month = month_names[non_emoji_monthly_peak-1] if not non_emoji_monthly.empty else 'N/A'

non_emoji_quarterly = df_non_emoji_expanded.groupby('quarter').size()
non_emoji_quarterly_peak = non_emoji_quarterly.idxmax() if not non_emoji_quarterly.empty else 0
non_emoji_quarterly_peak_value = non_emoji_quarterly.max() if not non_emoji_quarterly.empty else 0

# Add calculated metrics to summary data
summary_data['Emoji Relevant'] = [
    f"{emoji_docs}",
    f"{emoji_top_category} ({emoji_top_category_count})",
    f"{emoji_top_subcategory} ({emoji_top_subcategory_count})",
    f"{emoji_top_contributor} ({emoji_top_contributor_count})",
    f"{emoji_docs_per_year:.1f}",
    f"{emoji_monthly_peak_month} ({emoji_monthly_peak_value})",
    f"Q{emoji_quarterly_peak} ({emoji_quarterly_peak_value})"
]

summary_data['Irrelevant'] = [
    f"{non_emoji_docs}",
    f"{non_emoji_top_category} ({non_emoji_top_category_count})",
    f"{non_emoji_top_subcategory} ({non_emoji_top_subcategory_count})",
    f"{non_emoji_top_contributor} ({non_emoji_top_contributor_count})",
    f"{non_emoji_docs_per_year:.1f}",
    f"{non_emoji_monthly_peak_month} ({non_emoji_monthly_peak_value})",
    f"Q{non_emoji_quarterly_peak} ({non_emoji_quarterly_peak_value})"
]

# Create summary DataFrame and save to Excel
summary_df = pd.DataFrame(summary_data)
summary_excel = os.path.join(data_dir, 'emoji_vs_non_emoji_summary.xlsx')
summary_df.to_excel(summary_excel, index=False)
print(f"Comparative summary saved to: {summary_excel}")

# Create a concise summary figure
plt.figure(figsize=(12, 10))
# Use a different approach for a table visualization - text-based table
table_data = [[summary_data['Metric'][i], summary_data['Emoji Relevant'][i], summary_data['Irrelevant'][i]] 
              for i in range(len(summary_data['Metric']))]
table = plt.table(cellText=table_data, 
                 colLabels=['Metric', 'Emoji Relevant', 'Irrelevant'],
                 loc='center',
                 cellLoc='center',
                 colColours=['#d9d9d9', '#b3e0ff', '#ffcc99'])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.axis('off')
plt.title('Comparative Analysis: Emoji vs. Non-Emoji Documents', fontweight='bold', pad=20)

plt.tight_layout()
summary_file = os.path.join(summary_dir, 'emoji_vs_non_emoji_summary.png')
plt.savefig(summary_file, dpi=300, bbox_inches='tight')
plt.close()

# Create a final visualization - proportion of emoji documents over time with trend line
plt.figure(figsize=(14, 8))

# Convert emoji_by_year to DataFrame for easier manipulation
emoji_trend_df = emoji_by_year.reset_index()
emoji_trend_df = emoji_trend_df[['year', 'Emoji %', 'Emoji Relevant', 'Total']]

# Create the bar chart
ax1 = plt.subplot()
total_bars = ax1.bar(emoji_trend_df['year'], emoji_trend_df['Total'], color='lightgray', label='Total Documents')
emoji_bars = ax1.bar(emoji_trend_df['year'], emoji_trend_df['Emoji Relevant'], color='#1f77b4', label='Emoji Documents')

# Add a trend line for emoji percentage
ax2 = ax1.twinx()
ax2.plot(emoji_trend_df['year'], emoji_trend_df['Emoji %'], 'ro-', linewidth=2, markersize=8, label='Emoji %')
ax2.set_ylabel('Emoji Document Percentage (%)', color='red', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='red')
ax2.grid(False)

# Add percentage labels above the trend line
for i, row in emoji_trend_df.iterrows():
    if not np.isnan(row['Emoji %']):
        ax2.annotate(f"{row['Emoji %']:.1f}%",
                   xy=(row['year'], row['Emoji %']),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   color='red',
                   fontweight='bold')

# Set up the primary y-axis and other formatting
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Number of Documents', fontweight='bold')
ax1.set_title('UTC Document Evolution: Total vs. Emoji-Related (2004-2023)', fontweight='bold', pad=20)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Create legends for both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
trends_summary_file = os.path.join(summary_dir, 'emoji_document_evolution.png')
plt.savefig(trends_summary_file, dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis Complete!")
print(f"Analysis summary saved to: {summary_file}")
print(f"Document evolution chart saved to: {trends_summary_file}")
print(f"\nAll visualizations and data files have been generated in:")
print(f"- Plots directory: {plots_dir}")
print(f"- Data directory: {data_dir}")
print(f"- Emoji analysis directory: {emoji_dir}")
print("\nKey findings have been organized into the following categories:")
print("1. Document distribution by category and subcategory")
print("2. Source/contributor analysis")
print("3. Time series and trend analysis")
print("4. Emoji relevance analysis")
print("5. Comparative analysis between emoji and non-emoji documents")