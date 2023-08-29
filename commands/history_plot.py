import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

def read_hist():
    hist = pd.read_csv('history.csv')
    hist.columns = ['Epoch', 'Batch_Size', 'Loss', 'Accuracy', 'f1_score', 'Val_Loss', 'Val_Accuracy', 'Val_f1_score', ] # 'avg_vram_usage', 'min_vram_usage', 'max_vram_usage']
    return hist

def create_seaborn_figure(dataframe, columns, fig_name):
    plt.figure(figsize=(18, 6))
    
    for col in columns:
        sns.lineplot(data=dataframe, x=dataframe.index, y=col, label=col)
    
    plt.title(fig_name.capitalize())
    plt.legend()
    plt.savefig(f"./assets/figures/{fig_name}.png")

def update_readme():
    # Read existing README.md
    with open("./README.md", "r", encoding='utf-8') as f:
        content = f.read()

    # Load data
    hist = read_hist()

    # Create and save new Plotly figures
    figures_data = [
        {'data': hist, 'columns': ['Loss', 'Val_Loss'], 'name': 'loss'},
        {'data': hist, 'columns': ['Accuracy', 'Val_Accuracy'], 'name': 'acc'},
        {'data': hist, 'columns': ['f1_score', 'Val_f1_score'], 'name': 'f1'},
        # {'data': hist, 'columns': ['min_vram_usage', 'max_vram_usage', 'avg_vram_usage'], 'name': 'vram'},
        {'data': hist, 'columns': ['Batch_Size'], 'name': 'batch_size'},
    ]
    
    figures_md = "## Métricas actuales del modelo:\n"
    for figure in figures_data:
        create_seaborn_figure(figure['data'], figure['columns'], figure['name'])
        figures_md += f"### {figure['name'].capitalize()} Plot\n"
        figures_md += f"![{figure['name'].capitalize()} Plot](./assets/figures/{figure['name']}.png)\n"

    # Use regex to replace the existing Metrics section with the new figures
    updated_content = re.sub(r"## Métricas actuales del modelo:.*", figures_md, content, flags=re.DOTALL)

    # Write the modified content back to README.md
    with open("./README.md", "w", encoding='utf-8') as f:
        f.write(updated_content)

if __name__ == '__main__':
    update_readme()