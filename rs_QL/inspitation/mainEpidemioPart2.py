import biblioEpidemioSMA_P2 as myBibEpidemio
import pandas as pd
import numpy as np
import bokeh  # interactive visualization library for modern web browsers
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure, output_file, show, save, curdoc  # https://docs.bokeh.org/en/latest/docs/reference/plotting/figure.html?highlight=figure#
from bokeh.models.glyphs import Line  # https://docs.bokeh.org/en/latest/docs/reference/models/glyphs/line.html?highlight=line#bokeh.models.Line.line_alpha
# from bokeh.palettes import Category10
from bokeh.palettes import d3  # https://docs.bokeh.org/en/latest/docs/reference/palettes.html
from bokeh.models.annotations import Legend
from bokeh.models import HoverTool
from bokeh.models.mappers import LinearColorMapper
import panel as pn
import time
from bokeh.models.widgets import Panel, Tabs


def get_column_data(model):
    """pivot the model dataframe to get states count at each step"""
    agent_state = model.datacollector.get_agent_vars_dataframe()
    X = pd.pivot_table(agent_state.reset_index(), index='Step', columns='State', aggfunc=np.size, fill_value=0)
    labels = ['Susceptible', 'Infected', 'Removed']
    X.columns = labels[:len(X.columns)]
    return X


def plot_states_bokeh(model, title=''):
    """Plot cases per country"""
    X = get_column_data(model) # retourne le tableau croisé avec une colonne étape et les autres chaque état : susceptible, infecté...
    X = X.reset_index()
    source = ColumnDataSource(X) # utiliser l'équivalent en Bokeh
    i = 0
    # colors = Category10[3]
    colors = d3['Category10'][3]
    items = []
    p = figure(plot_width=600, plot_height=400, tools=[], title=title, x_range=(0, 100))
    for c in X.columns[1:]:
        line = Line(x='Step', y=c, line_color=colors[i], line_width=3, line_alpha=.8, name=c)
        glyph = p.add_glyph(source, line)
        i += 1
        items.append((c, [glyph]))
    p.xaxis.axis_label = 'Step'
    # Adds an object to the plot in a specified place
    p.add_layout(Legend(location='center_right', items=items))
    p.background_fill_color = "#e1e1ea"
    p.background_fill_alpha = 0.5
    p.legend.label_text_font_size = "10pt"
    p.title.text_font_size = "15pt"
    p.toolbar.logo = None
    p.sizing_mode = 'scale_height'
    return p


def grid_values(model):
    """Get grid cell states"""
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    w = model.grid.width
    df = pd.DataFrame(agent_counts)
    for cell in model.grid.coord_iter():
        agents, x, y = cell
        c = None
        for a in agents:
            c = a.state
        df.iloc[x, y] = c
    return df


# Création du graphique cellules
def plot_cells_bokeh(model):
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    w = model.grid.width
    df = grid_values(model)
    df = pd.DataFrame(df.stack(), columns=['value']).reset_index()
    columns = ['value']
    x = [(i, "@%s" % i) for i in columns]
    # informations curseur : https://docs.bokeh.org/en/latest/docs/reference/models/tools.html?highlight=hovertool#bokeh.models.HoverTool
    hover = HoverTool(tooltips=x, point_policy='follow_mouse')
    # colors = Category10[3]
    colors = d3['Category10'][3]
    mapper = LinearColorMapper(palette=colors, low=df.value.min(), high=df.value.max())
    p = figure(plot_width=500, plot_height=500, tools=[hover], x_range=(-1, w), y_range=(-1, w))
    p.rect(x="level_0", y="level_1", width=1, height=1,
           source=df,
           fill_color={'field': 'value', 'transform': mapper},
           line_color='black')
    p.background_fill_color = "black"
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.toolbar.logo = None
    return p


pop = 100
steps = 100
pn.extension()
plot_pane = pn.pane.Bokeh()
grid_pane = pn.pane.Bokeh()
output_file("epidemio.html")
model = myBibEpidemio.InfectionModel(pop, 20, 20, ptrans=0.5)
for i in range(steps):
    model.step()
    p1 = plot_states_bokeh(model, title='step=%s' % i)
    tab1 = Panel(child=p1, title="Courbes")
    p2 = plot_cells_bokeh(model)
    tab2 = Panel(child=p2, title="Grille")
    tabs = Tabs(tabs=[tab1, tab2])
    save(tabs)
    time.sleep(0.01)
