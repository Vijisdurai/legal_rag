content = open('evaluation/metrics.py', encoding='utf-8').read()
# Fix tabulate format
content = content.replace("tablefmt='fancy_grid'", "tablefmt='grid'")
open('evaluation/metrics.py', 'w', encoding='utf-8').write(content)
print('Fixed tablefmt in metrics.py')
