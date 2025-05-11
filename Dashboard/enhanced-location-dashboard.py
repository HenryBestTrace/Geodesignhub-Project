# Import necessary libraries
from dash import Dash, dcc, html, dash_table, Input, Output, State, ALL, MATCH, callback_context
from dash.dependencies import Input, Output, State
import pandas as pd
from urllib.parse import parse_qs, unquote, quote
import plotly.graph_objects as go
from shapely import wkt
import numpy as np
import warnings
import colorsys
import json
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Read the data with correct encoding
try:
    df = pd.read_csv("output_location_differences.csv", encoding='cp1252')
except:
    # Fallback to UTF-8 if cp1252 fails
    df = pd.read_csv("output_location_differences.csv", encoding='utf-8')

# Ensure numeric columns are properly typed
df['area'] = pd.to_numeric(df['area'], errors='coerce')
df['shape_index'] = pd.to_numeric(df['shape_index'], errors='coerce')

# Add a unique identifier for each row if not already present
if 'id' not in df.columns:
    df['id'] = range(len(df))


# 函数来生成格式化的位置代码
def format_location_code(category, locations):
    """
    基于类别名称格式化位置代码
    将位置代码格式化为：类别前三个字母 + "-Loc" + 两位数序号（01, 02, ...）
    """
    # 获取类别名称的前三个字母
    prefix = ''.join([c for c in category[:3] if c.isalpha()]).capitalize()

    # 为每个位置分配新的格式化代码
    formatted_codes = {}

    # 从1开始为每个位置分配序号
    for i, loc in enumerate(locations, 1):
        formatted_codes[loc] = f"{prefix}-Loc{i:02d}"

    return formatted_codes


# 处理主表：去重 Category + sub
df_main = df[['category', 'sub']].drop_duplicates().reset_index(drop=True)

# 重命名 'sub' 为 'location'
df_main = df_main.rename(columns={'sub': 'location'})

# 计算每个位置的响应数量
location_counts = df.groupby(['category', 'sub']).size().reset_index(name='count')
location_counts = location_counts.rename(columns={'sub': 'location'})

# 合并计数到主表
df_main = pd.merge(df_main, location_counts[['category', 'location', 'count']], on=['category', 'location'], how='left')

# 计算每个类别的位置数（用于表格渲染）
df_main["RowSpan"] = df_main.groupby("category")["location"].transform("count")

# 为每个类别创建格式化的位置代码
# 首先获取每个类别的所有位置
categories = df_main['category'].unique()
format_mapping = {}

for category in categories:
    # 获取该类别下的所有位置
    locations = df_main[df_main['category'] == category]['location'].unique()

    # 创建格式化的位置代码
    formatted_codes = format_location_code(category, locations)

    # 添加到全局映射
    for old_code, new_code in formatted_codes.items():
        format_mapping[(category, old_code)] = new_code

# 添加格式化位置代码列到主表
df_main['display_location'] = df_main.apply(
    lambda row: format_mapping.get((row['category'], row['location']), row['location']),
    axis=1
)

# Initialize Dash app
app = Dash(__name__, requests_pathname_prefix='/location/')
app.config.suppress_callback_exceptions = True

# Style definitions
header_style = {
    'backgroundColor': 'lightgrey',
    'fontWeight': 'bold',
    'padding': '10px',
    'textAlign': 'center',
    'border': '1px solid #ddd'
}

cell_style = {
    'border': '1px solid #ddd',
    'padding': '10px',
    'textAlign': 'center',
    'verticalAlign': 'middle'
}

link_style = {
    'textDecoration': 'none',
    'color': '#0066cc'
}

map_style = {
    'height': '500px',
    'margin': '20px',
    'border': '1px solid #ddd',
    'borderRadius': '5px'
}


# Function to generate pastel colors for responses
def generate_pastel_colors(num_colors):
    """Generate a set of distinct pastel colors with different hues"""
    pastel_colors = []

    # Use evenly spaced hues around the color wheel
    for i in range(num_colors):
        hue = i / num_colors  # Value between 0 and 1
        # Lower saturation (0.4-0.6) and high value (0.9) for pastel look
        saturation = 0.4 + 0.2 * (i % 3) / 2  # Slight variations in saturation
        value = 0.9
        # Convert HSV to RGB
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, saturation, value)]
        pastel_colors.append((r, g, b))

    return pastel_colors


# Function to get color string with transparency
def get_color_with_alpha(rgb_color, alpha=0.5):
    """Convert RGB color tuple to rgba string with given alpha value"""
    r, g, b = rgb_color
    return f'rgba({r}, {g}, {b}, {alpha})'


# For main table styling - neutral colors
main_table_bg_color = '#f8f9fa'  # Light gray background
main_table_text_color = '#333333'  # Dark gray text

# Pre-compute up to 50 pastel colors (should be enough for most responses)
pastel_color_palette = generate_pastel_colors(50)


# Function to convert from Web Mercator to WGS84
def mercator_to_wgs84(x, y):
    """Convert Web Mercator (EPSG:3857) coordinates to WGS84 (EPSG:4326)"""
    # Earth radius in meters
    R = 6378137
    # Convert x-coordinate
    lon = (x * 180) / (R * np.pi)
    # Convert y-coordinate
    lat_rad = np.arcsin(np.tanh(y / R))
    lat = lat_rad * 180 / np.pi
    return lon, lat


# Function to create map from multiple geometries with pastel colors
# Function to create map from multiple geometries with pastel colors
def create_map(filtered_df, highlighted_id=None):
    """
    Create a map with multiple geometries using different pastel colors for each response

    Parameters:
    filtered_df -- DataFrame containing the data to display
    highlighted_id -- ID of the row to highlight (if any)
    """
    fig = go.Figure()

    # If there's no data, return empty figure
    if filtered_df.empty:
        return fig

    # Get unique responses for color assignment
    unique_responses = filtered_df['response'].unique()

    # Assign a unique pastel color to each response
    response_colors = {}
    for i, response in enumerate(unique_responses):
        # Use modulo to cycle through the palette if there are more responses than colors
        color_idx = i % len(pastel_color_palette)
        response_colors[response] = pastel_color_palette[color_idx]

    # Add each geometry to the map
    all_lats = []
    all_lons = []

    # Store coordinates of highlighted geometry for focusing
    highlighted_lats = []
    highlighted_lons = []

    for i, (idx, row) in enumerate(filtered_df.iterrows()):
        try:
            if pd.isna(row['geometry']) or not row['geometry']:
                continue

            geom = wkt.loads(row['geometry'])

            # Get the response for this row
            response = row['response']

            # Get the color for this response
            rgb_color = response_colors.get(response, pastel_color_palette[0])

            # Check if this row is highlighted
            is_highlighted = highlighted_id is not None and row['id'] == highlighted_id

            # Format hover text with proper handling of NaN values
            response_text = f"Response: {row['response']}" if not pd.isna(row['response']) else "Response: N/A"
            olc_text = f"OLC: {row['OLCs']}" if not pd.isna(row['OLCs']) else "OLC: N/A"

            # Add row ID and index to hover text for easier identification
            id_text = f"ID: {row['id']}"
            index_text = f"Index: {i + 1}"

            hover_text = f"{response_text}<br>{olc_text}<br>{id_text}<br>{index_text}"

            # Set customdata for callbacks (row id)
            customdata = [int(row['id'])]

            # Extract coordinates based on geometry type
            if geom.geom_type == 'Polygon':
                # Get coordinates from polygon exterior
                coords = list(geom.exterior.coords)
                lons = []
                lats = []

                # Convert each coordinate from Web Mercator to WGS84
                for x, y in coords:
                    lon, lat = mercator_to_wgs84(x, y)
                    lons.append(lon)
                    lats.append(lat)

                all_lats.extend(lats)
                all_lons.extend(lons)

                # Store coordinates if this is the highlighted geometry
                if is_highlighted:
                    highlighted_lats = lats.copy()
                    highlighted_lons = lons.copy()

                # Apply highlighting: add a wider black outline for highlighted geometries
                if is_highlighted:
                    # First add a wider black outer outline
                    fig.add_trace(go.Scattermapbox(
                        fill=None,
                        lon=lons,
                        lat=lats,
                        mode='lines',  # Only use lines, no markers
                        line={'color': 'black', 'width': 6},  # Wide black outline
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Use highlighting style
                    line_color = 'yellow'  # Yellow highlight
                    line_width = 3
                    # Use higher opacity fill for highlighted elements
                    fill_color = get_color_with_alpha(rgb_color, 0.7)
                else:
                    # Normal style for non-highlighted elements
                    line_color = get_color_with_alpha(rgb_color, 0.8)
                    line_width = 2
                    # Use lower opacity (more transparent) fill for normal elements
                    fill_color = get_color_with_alpha(rgb_color, 0.4)

                # Add the polygon as a filled area
                fig.add_trace(go.Scattermapbox(
                    fill="toself",
                    lon=lons,
                    lat=lats,
                    mode='lines',  # Only use lines, no markers
                    line={'color': line_color, 'width': line_width},
                    fillcolor=fill_color,
                    name=f"Response {i + 1}",
                    hoverinfo="text",
                    text=hover_text,
                    customdata=customdata
                ))
            elif geom.geom_type == 'LineString':
                # Process LineString geometries...
                coords = list(geom.coords)
                lons = []
                lats = []

                for x, y in coords:
                    lon, lat = mercator_to_wgs84(x, y)
                    lons.append(lon)
                    lats.append(lat)

                all_lats.extend(lats)
                all_lons.extend(lons)

                if is_highlighted:
                    highlighted_lats = lats.copy()
                    highlighted_lons = lons.copy()

                # Apply highlighting
                if is_highlighted:
                    # First add a wider black outer line
                    fig.add_trace(go.Scattermapbox(
                        mode="lines",
                        lon=lons,
                        lat=lats,
                        line={'width': 6, 'color': 'black'},
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Use highlighted style
                    line_color = 'yellow'  # Yellow highlight
                    line_width = 3
                else:
                    # Normal style - use pastel color with good opacity
                    line_color = get_color_with_alpha(rgb_color, 0.8)
                    line_width = 2

                # Add the line
                fig.add_trace(go.Scattermapbox(
                    mode="lines",  # Only use lines, no markers
                    lon=lons,
                    lat=lats,
                    line={'width': line_width, 'color': line_color},
                    name=f"Response {i + 1}",
                    hoverinfo="text",
                    text=hover_text,
                    customdata=customdata
                ))
            elif geom.geom_type == 'Point':
                # Process Point geometries...
                lon, lat = mercator_to_wgs84(geom.x, geom.y)

                all_lats.append(lat)
                all_lons.append(lon)

                if is_highlighted:
                    highlighted_lats = [lat]
                    highlighted_lons = [lon]

                # Apply highlighting
                if is_highlighted:
                    # First add a larger black outer circle
                    fig.add_trace(go.Scattermapbox(
                        mode="markers",
                        lon=[lon],
                        lat=[lat],
                        marker={'size': 18, 'color': 'black'},
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Use highlighted style
                    marker_color = 'yellow'  # Yellow highlight
                    marker_size = 12
                else:
                    # Normal style - use pastel color with good opacity
                    marker_color = get_color_with_alpha(rgb_color, 0.8)
                    marker_size = 10

                # Add the point
                fig.add_trace(go.Scattermapbox(
                    mode="markers",
                    lon=[lon],
                    lat=[lat],
                    marker={'size': marker_size, 'color': marker_color},
                    name=f"Response {i + 1}",
                    hoverinfo="text",
                    text=hover_text,
                    customdata=customdata
                ))
            elif geom.geom_type == 'MultiPolygon':
                # Process MultiPolygon geometries...
                for poly in geom.geoms:
                    coords = list(poly.exterior.coords)
                    lons = []
                    lats = []

                    for x, y in coords:
                        lon, lat = mercator_to_wgs84(x, y)
                        lons.append(lon)
                        lats.append(lat)

                    all_lats.extend(lats)
                    all_lons.extend(lons)

                    if is_highlighted:
                        highlighted_lats.extend(lats)
                        highlighted_lons.extend(lons)

                    # Apply highlighting
                    if is_highlighted:
                        # First add a wider black outer outline
                        fig.add_trace(go.Scattermapbox(
                            fill=None,
                            lon=lons,
                            lat=lats,
                            mode='lines',  # Only use lines, no markers
                            line={'color': 'black', 'width': 6},  # Wide black outline
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        # Use highlighted style
                        line_color = 'yellow'  # Yellow highlight
                        line_width = 3
                        # Use higher opacity fill for highlighted elements
                        fill_color = get_color_with_alpha(rgb_color, 0.7)
                    else:
                        # Normal style
                        line_color = get_color_with_alpha(rgb_color, 0.8)
                        line_width = 2
                        # Use lower opacity (more transparent) fill for normal elements
                        fill_color = get_color_with_alpha(rgb_color, 0.4)

                    # Add each polygon as a filled area
                    fig.add_trace(go.Scattermapbox(
                        fill="toself",
                        lon=lons,
                        lat=lats,
                        mode='lines',  # Only use lines, no markers
                        line={'color': line_color, 'width': line_width},
                        fillcolor=fill_color,
                        name=f"Response {i + 1}",
                        hoverinfo="text",
                        text=hover_text,
                        customdata=customdata
                    ))
        except Exception as e:
            print(f"Error processing geometry at row {i}: {e}")

    # Set the map center and zoom
    if all_lats and all_lons:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)

        # Auto-focus on highlighted geometry if available
        if highlighted_id is not None and highlighted_lats and highlighted_lons:
            # Use center of highlighted geometry
            center_lat = sum(highlighted_lats) / len(highlighted_lats)
            center_lon = sum(highlighted_lons) / len(highlighted_lons)
            # Use higher zoom level
            zoom = 16
        else:
            # Default zoom level
            zoom = 15

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

    # Add click event handler
    fig.update_layout(clickmode='event+select')

    return fig


# Main page layout with updated title and description
main_layout = html.Div([
    html.H1("Different Spaces for the Same Ideas", style={'textAlign': 'center'}),
    html.P(
        "This dashboard categorises the different spaces which are proposed for the same category of responses",
        style={'textAlign': 'center', 'marginBottom': '20px', 'fontSize': '16px', 'color': '#555'}
    ),
    html.Div([
        dcc.Input(id="search-input", type="text", placeholder="Enter Category or Location"),
        html.Button("Search", id="search-button")
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div(
        html.Table(
            id='main-table',
            style={'width': '80%', 'borderCollapse': 'collapse', 'margin': 'auto'},
            children=[
                html.Thead(
                    html.Tr([
                        html.Th("Category", style=header_style),
                        html.Th("Category Location", style=header_style),
                        html.Th("Count", style=header_style)
                    ])
                ),
                html.Tbody(id="table-body")
            ]
        )
    )
])


# Detail page layout with custom HTML table for cell merging and interactive features
# Detail page layout with custom HTML table for cell merging and interactive features
def detail_layout(category, location):
    # Filter data using 'sub' since we're using the original dataframe
    filtered_df = df[(df["category"] == category) & (df["sub"] == location)]

    # Create data for the HTML table with merged cells and interactive rows
    table_rows = []

    # Get unique location value and count for merging
    unique_location = location
    location_count = len(filtered_df)

    # Neutral light background for header
    light_bg_color = '#f8f9fa'

    # 获取此类别和位置的格式化显示名称
    display_location = format_mapping.get((category, location), location)

    # Create table rows with interactive features
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        row_id = row['id']

        # Style for interactive row - base style
        interactive_row_style = {
            'cursor': 'pointer',
            'transition': 'all 0.3s'
        }

        if i == 0:
            # First row includes the merged location cell
            table_rows.append(
                html.Tr([
                    # Merged location cell for first row with neutral color
                    html.Td(
                        display_location,
                        rowSpan=location_count,
                        style={
                            'border': '1px solid #ddd',
                            'padding': '10px',
                            'textAlign': 'center',
                            'verticalAlign': 'middle',
                            'backgroundColor': light_bg_color,
                            'fontWeight': 'bold'
                        }
                    ),
                    # Response cell
                    html.Td(row['response'], style=cell_style),
                    # OLC cell
                    html.Td(row['OLCs'] if not pd.isna(row['OLCs']) else "N/A", style=cell_style)
                ],
                    id={'type': 'table-row', 'index': row_id},
                    style=interactive_row_style
                )
            )
        else:
            # Subsequent rows only need the non-merged cells
            table_rows.append(
                html.Tr([
                    html.Td(row['response'], style=cell_style),
                    html.Td(row['OLCs'] if not pd.isna(row['OLCs']) else "N/A", style=cell_style)
                ],
                    id={'type': 'table-row', 'index': row_id},
                    style=interactive_row_style
                )
            )

    return html.Div([
        # 返回按钮已被移除

        html.H2(
            f"Details for {category} - {display_location}",
            style={
                'textAlign': 'center',
                'color': '#333',  # 使用标准深灰色
                'borderBottom': '3px solid #007bff',  # 使用标准蓝色
                'paddingBottom': '8px',
                'display': 'inline-block',
                'margin': '20px auto',
                'width': 'auto'
            }
        ),

        # Instructions for interaction
        html.Div([
            html.P(
                "Click on a table row to highlight the corresponding location on the map. Click on a map element to highlight its row in the table.",
                style={'textAlign': 'center', 'fontStyle': 'italic', 'marginBottom': '20px', 'color': '#666'})
        ]),

        # Hidden div to store the highlighted ID
        dcc.Store(id='highlighted-id', data=None),

        # Custom HTML table with merged cells and interactive rows
        html.Table(
            id='detail-table',
            style={'width': '80%', 'borderCollapse': 'collapse', 'marginBottom': '20px', 'margin': 'auto'},
            children=[
                html.Thead(
                    html.Tr([
                        html.Th(display_location, style=header_style),
                        html.Th("Response", style=header_style),
                        html.Th("OLC", style=header_style)
                    ])
                ),
                html.Tbody(id="detail-table-body", children=table_rows)
            ]
        ),

        # Interactive map
        dcc.Graph(
            id='geometry-map',
            figure=create_map(filtered_df),
            style=map_style,
            clear_on_unhover=True
        )
    ], style={'textAlign': 'center'})


# Callback functions
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('url', 'search')]
)
def display_page(pathname, search):
    print(f"Path requested: {pathname}")  # 调试信息

    # 修复路径匹配问题 - 处理各种可能的路径格式
    if pathname and ('detail' in pathname):
        params = parse_qs(search.lstrip('?'))
        category = unquote(params.get('category', [None])[0])
        location = unquote(params.get('sub', [None])[0])  # Keep as 'sub' in URL parameters for backward compatibility

        print(f"Detail view requested for category: {category}, location: {location}")  # 调试信息

        if category and location:
            return detail_layout(category, location)
        else:
            print(f"Missing parameters - category: {category}, location: {location}")  # 调试信息
            return main_layout
    return main_layout


@app.callback(
    Output("table-body", "children"),
    [Input('url', 'pathname'),
     Input('search-button', 'n_clicks')],
    [State('search-input', 'value')]
)
def update_table_body(pathname, n_clicks, search_term):
    # Filter data if search term is provided
    if search_term:
        filtered_df = df_main[
            (df_main["category"].str.contains(search_term, case=False)) |
            (df_main["location"].str.contains(search_term, case=False)) |
            (df_main["display_location"].str.contains(search_term, case=False))
            ]
    else:
        filtered_df = df_main

    rows = []
    current_category = None

    # 调试信息
    print(f"Generating table rows for {len(filtered_df)} locations")

    for index, row in filtered_df.iterrows():
        category = row["category"]
        location = row["location"]
        display_location = row["display_location"]
        count = row["count"]

        # 构造详细视图的URL，确保路径正确
        # 使用相对路径以适应任何前缀配置
        detail_url = f"detail?category={quote(category)}&sub={quote(location)}"

        if category != current_category:
            current_category = category
            rowspan = int(row["RowSpan"])
            rows.append(html.Tr([
                html.Td(
                    category,
                    rowSpan=rowspan,
                    style={
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'textAlign': 'center',
                        'verticalAlign': 'middle',
                        'fontWeight': 'bold'
                    }
                ),
                html.Td(
                    html.A(
                        display_location,
                        href=detail_url,
                        style={
                            'textDecoration': 'none',
                            'fontWeight': '500',
                            'color': '#0066cc'  # 蓝色以明确表示可点击
                        }
                    ),
                    style=cell_style
                ),
                html.Td(
                    count,
                    style=cell_style
                )
            ]))
        else:
            rows.append(html.Tr([
                html.Td(
                    html.A(
                        display_location,
                        href=detail_url,
                        style={
                            'textDecoration': 'none',
                            'fontWeight': '500',
                            'color': '#0066cc'  # 蓝色以明确表示可点击
                        }
                    ),
                    style=cell_style
                ),
                html.Td(
                    count,
                    style=cell_style
                )
            ]))

    if not rows:
        rows = [html.Tr([html.Td("No data found", colSpan=3, style={'textAlign': 'center', 'padding': '20px'})])]

    return rows


# Callback to update highlighted item when a row is clicked
@app.callback(
    Output('highlighted-id', 'data'),
    [Input({'type': 'table-row', 'index': ALL}, 'n_clicks'),
     Input('geometry-map', 'clickData')],
    [State('highlighted-id', 'data'),
     State('url', 'search')]
)
def update_highlighted_id(row_clicks, map_click_data, current_highlighted, search):
    # Get context of the callback
    ctx = callback_context
    if not ctx.triggered:
        return None

    # Get the ID of the component that triggered the callback
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # If a table row was clicked
    if 'table-row' in trigger_id:
        # Extract the row ID from the trigger ID
        try:
            trigger_dict = json.loads(trigger_id)
            clicked_id = trigger_dict['index']

            # Toggle highlight: if the same row is clicked again, unhighlight it
            if current_highlighted == clicked_id:
                return None
            return clicked_id
        except:
            return current_highlighted

    # If a map element was clicked
    elif 'geometry-map' in trigger_id and map_click_data:
        try:
            # Extract the row ID from the customdata of the clicked point
            clicked_id = map_click_data['points'][0]['customdata'][0]

            # Toggle highlight: if the same element is clicked again, unhighlight it
            if current_highlighted == clicked_id:
                return None
            return clicked_id
        except:
            return current_highlighted

    # If something else triggered the callback
    return current_highlighted


# Callback to update row styles based on highlight state
@app.callback(
    Output({'type': 'table-row', 'index': ALL}, 'style'),
    [Input('highlighted-id', 'data')],
    [State({'type': 'table-row', 'index': ALL}, 'id'),
     State('url', 'search')]
)
def update_row_styles(highlighted_id, row_ids, search):
    # Get category and location from URL
    params = parse_qs(search.lstrip('?'))
    category = unquote(params.get('category', [None])[0])
    location = unquote(params.get('sub', [None])[0])

    # Default style
    styles = []

    # Base style for all rows
    base_style = {
        'cursor': 'pointer',
        'transition': 'all 0.3s'
    }

    # Enhanced highlight style with better contrast and border effect
    highlight_style = {
        'cursor': 'pointer',
        'transition': 'all 0.3s',
        'backgroundColor': 'rgba(255, 255, 0, 0.3)',  # Yellow highlight background
        'boxShadow': 'inset 0 0 0 2px #FFD700',  # Inner gold border
        'fontWeight': 'bold',
        'transform': 'translateX(5px)'  # Slight indent effect
    }

    # Apply appropriate style for each row
    for row_id_dict in row_ids:
        row_id = row_id_dict['index']
        if highlighted_id is not None and row_id == highlighted_id:
            styles.append(highlight_style)
        else:
            styles.append(base_style)

    return styles


# Callback to update map with highlighting
@app.callback(
    Output('geometry-map', 'figure'),
    [Input('highlighted-id', 'data')],
    [State('url', 'search')]
)
def update_map(highlighted_id, search):
    # Get category and location
    params = parse_qs(search.lstrip('?'))
    category = unquote(params.get('category', [None])[0])
    location = unquote(params.get('sub', [None])[0])  # Using 'sub' as the parameter name

    # Filter data
    if category and location:
        filtered_df = df[(df["category"] == category) & (df["sub"] == location)]
        # Create map with double outline and auto-focus effects
        return create_map(filtered_df, highlighted_id)

    # Fallback to empty figure
    return go.Figure()


# App configuration
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8051)