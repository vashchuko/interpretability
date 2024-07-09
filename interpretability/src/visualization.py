import plotly.express as px

def imshow(tensor, save_location, xaxis="", yaxis="", **kwargs):
    px.imshow(tensor, color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).write_html(save_location)