import dash
from dash import html, dcc
import plotly.graph_objs as go
import networkx as nx

# Sample data
clues = ["The door was unlocked.", "A note was found on the table.", "No fingerprints on the glass."]
suspects = ["Alice", "Bob"]
events = ["Entry", "Note discovery"]
relationships = [
    ("The door was unlocked.", "Entry"),
    ("A note was found on the table.", "Note discovery"),
    ("No fingerprints on the glass.", "Alice"),
    ("Entry", "Bob"),
    ("Note discovery", "Alice")
]

# Build the graph
G = nx.Graph()
for clue in clues:
    G.add_node(clue, type='clue')
for suspect in suspects:
    G.add_node(suspect, type='suspect')
for event in events:
    G.add_node(event, type='event')
for src, tgt in relationships:
    G.add_edge(src, tgt)

pos = nx.spring_layout(G)

# Edge trace
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Node trace
node_x = []
node_y = []
node_text = []
node_color = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    node_type = G.nodes[node]['type']
    if node_type == 'clue':
        node_color.append('skyblue')
    elif node_type == 'suspect':
        node_color.append('salmon')
    else:
        node_color.append('lightgreen')
node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    text=node_text,
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=node_color,
        size=30,
        line=dict(width=2)
    )
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Case Network Graph',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                ))

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("AI Detective Agent - Interactive Network Graph"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run(debug=True)