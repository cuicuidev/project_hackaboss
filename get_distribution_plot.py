import pandas as pd
import plotly.express as px

def plot_category_distribution(sort_by='Total'):
    # Read data from CSV
    df = pd.read_csv('category_counts.csv')

    # Pivot the DataFrame to get counts for each Parent within the same row for each Category
    pivot_df = df.pivot(index='Category', columns='Parent', values='Count').fillna(0)
    pivot_df['Total'] = pivot_df.sum(axis=1)
    
    # Sort by the column specified
    sorted_df = pivot_df.sort_values(by=sort_by, ascending=False)
    sorted_categories = sorted_df.index.tolist()

    # Create the bar plot
    fig = px.bar(df, x='Count', y='Category', color='Parent',
                 category_orders={"Category": sorted_categories})
    
    fig.update_layout(height = 800, xaxis = dict(side = 'top', title_text = ''), yaxis = dict(title_text = ''))
    
    return fig