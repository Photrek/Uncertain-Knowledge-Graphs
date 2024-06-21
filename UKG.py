import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

#Read of Data
#/content/drive/MyDrive/Knowledge Graphs/Data.xlsx

data = pd.read_csv('/content/diseases and treatments no doubles.csv')
print(data)

# Creating a directed graph for uncertain knowledge
UKG = nx.DiGraph()

# Adding nodes and edges to the graph.
for _, row in data.iterrows():
    UKG.add_node(row['Disease'], node_type='Disease')
    UKG.add_node(row['Medication'], node_type='Medication')
    UKG.add_edge(row['Disease'], row['Medication'], probability=row['Probability'])

# Plot
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(UKG)
nx.draw(UKG, pos, with_labels=True, font_weight="bold", node_size=700, node_color="skyblue", font_color="black", font_size=10)

# Adding edge labels.
edge_labels = nx.get_edge_attributes(UKG, "probability")
nx.draw_networkx_edge_labels(UKG, pos, edge_labels=edge_labels)

plt.show()