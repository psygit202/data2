import nbformat
from nbclient import NotebookClient
import os

def execute_notebook(input_path, output_path):
    print(f"Executing {input_path}...")
    with open(input_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # We need to skip the first cell if it tries to install things
    # or just let it run if the environment has pip.
    # Actually, let's just run it.
    
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    
    try:
        client.execute()
        print("Execution successful.")
    except Exception as e:
        print(f"Execution failed: {e}")
        # We still save the partial results if possible
    
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Saved executed notebook to {output_path}")

if __name__ == "__main__":
    execute_notebook("retail_crm_analytics.ipynb", "retail_crm_analytics.ipynb")
