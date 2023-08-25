import os
import plotly.express as px
import pandas as pd
import re
import time
import seaborn as sns
import matplotlib.pyplot as plt

def read_hist():
    hist = pd.read_csv('history.csv')
    hist.columns = ['Epoch', 'Loss', 'Accuracy', 'f1_score', 'Val_Loss', 'Val_Accuracy', 'Val_f1_score']
    return hist

def create_seaborn_figure(dataframe, columns, fig_name):
    print(f"Creating figure: {fig_name}")
    start_time = time.time()
    
    plt.figure(figsize=(18, 6))
    
    for col in columns:
        sns.lineplot(data=dataframe, x=dataframe.index, y=col, label=col)
    
    plt.title(fig_name.capitalize())
    plt.legend()
    
    print("Created figure, now writing...")
    plt.savefig(f"./assets/figures/{fig_name}.png")
    
    print(f"Figure {fig_name} created and written. Time taken: {time.time() - start_time}")

def update_readme():
    print("Starting to update README...")
    start_time = time.time()

    # Read existing README.md
    with open("./README.md", "r", encoding='utf-8') as f:
        content = f.read()

    # Load data
    print("Loading data...")
    hist = read_hist()
    # Create and save new Plotly figures
    figures_data = [
        {'data': hist, 'columns': ['Loss', 'Val_Loss'], 'name': 'loss'},
        {'data': hist, 'columns': ['Accuracy', 'Val_Accuracy'], 'name': 'acc'},
        {'data': hist, 'columns': ['f1_score', 'Val_f1_score'], 'name': 'f1'}
    ]
    
    figures_md = "## Métricas actuales del modelo:\n"
    for figure in figures_data:
        create_seaborn_figure(figure['data'], figure['columns'], figure['name'])
        figures_md += f"### {figure['name'].capitalize()} Plot\n"
        figures_md += f"![{figure['name'].capitalize()} Plot](./assets/figures/{figure['name']}.png)\n"

    # Use regex to replace the existing Metrics section with the new figures
    print("Updating README...")
    updated_content = re.sub(r"## Métricas actuales del modelo:.*", figures_md, content, flags=re.DOTALL)

    # Write the modified content back to README.md
    with open("./README.md", "w", encoding='utf-8') as f:
        f.write(updated_content)

    print(f"README updated. Total time taken: {time.time() - start_time}")