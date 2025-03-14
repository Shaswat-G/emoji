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
#   doc_num | doc_url | subject | source | date | doc_type
# Where doc_type is something like:
#   {"Public Review & Feedback": ["General Feedback & Correspondence"],
#    "Proposals": ["Character Encoding Proposals"]}

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

# Create directories if they don't exist
os.makedirs(doc_counts_dir, exist_ok=True)
os.makedirs(sources_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

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
            'subcategory': ''
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
                'subcategory': ''
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
                    'subcategory': subcat
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