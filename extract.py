import json
import codecs

with codecs.open('icor_analysis.ipynb', 'r', 'utf-8') as f:
    nb = json.load(f)

with codecs.open('notebook_content.md', 'w', 'utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        source = "".join(cell.get('source', []))
        out.write(f"## Cell {i} ({cell_type})\n")
        out.write(source)
        out.write("\n\n---\n\n")
