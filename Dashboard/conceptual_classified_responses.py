from dash import Dash, dcc, html, dash_table, ALL, callback_context
from dash.dependencies import Input, Output, State
import pandas as pd
from urllib.parse import parse_qs, unquote, quote
import plotly.graph_objects as go
from shapely import wkt
import numpy as np
import colorsys
import warnings
import json
import re

# 抑制警告
warnings.filterwarnings('ignore')


# 创建一个函数来删除emoji
def remove_emoji(text):
    if not isinstance(text, str):
        return text

    # 定义一个正则表达式模式来匹配emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text).strip()


# 读取数据
df = pd.read_csv("./conceptual_classified_responses.csv")

# 删除Category列中的emoji
df['Category'] = df['Category'].apply(remove_emoji)

# 确保数据有唯一标识符
if 'id' not in df.columns:
    df['id'] = range(len(df))

# 处理主表：去重 Open Location Code + Category
df_main = df[['Open Location Code', 'Category']].drop_duplicates().reset_index(drop=True)

# 计算每个类别在每个位置的响应数量
category_counts = df.groupby(['Open Location Code', 'Category']).size().reset_index(name='Count')

# 合并计数到主表
df_main = pd.merge(df_main, category_counts[['Open Location Code', 'Category', 'Count']],
                   on=['Open Location Code', 'Category'], how='left')

# 计算每个 Open Location Code 的出现次数，用于合并单元格
df_main["RowSpan"] = df_main.groupby("Open Location Code")["Category"].transform("count")


# 生成柔和颜色函数
def generate_pastel_colors(num_colors):
    """生成一组不同色调的柔和颜色"""
    pastel_colors = []

    # 在色轮上使用均匀分布的色调
    for i in range(num_colors):
        hue = i / num_colors  # 0到1之间的值
        # 降低饱和度(0.4-0.6)和高亮度(0.9)以获得柔和外观
        saturation = 0.4 + 0.2 * (i % 3) / 2  # 饱和度轻微变化
        value = 0.9
        # HSV转换为RGB
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, saturation, value)]
        pastel_colors.append((r, g, b))

    return pastel_colors


# 添加透明度函数
def get_color_with_alpha(rgb_color, alpha=0.5):
    """将RGB颜色元组转换为带有指定透明度的rgba字符串"""
    r, g, b = rgb_color
    return f'rgba({r}, {g}, {b}, {alpha})'


# 预计算多达50种柔和颜色（足够应对大多数响应）
pastel_color_palette = generate_pastel_colors(50)


# Web Mercator坐标转换为WGS84函数
def mercator_to_wgs84(x, y):
    """将Web Mercator (EPSG:3857)坐标转换为WGS84 (EPSG:4326)"""
    # 地球半径（米）
    R = 6378137
    # 转换x坐标
    lon = (x * 180) / (R * np.pi)
    # 转换y坐标
    lat_rad = np.arcsin(np.tanh(y / R))
    lat = lat_rad * 180 / np.pi
    return lon, lat


# 地图样式设置
map_style = {
    'height': '500px',
    'margin': '20px',
    'border': '1px solid #ddd',
    'borderRadius': '5px'
}


# 创建地图函数 - 改进版
def create_map(filtered_df, highlighted_id=None):
    """
    创建地图，使用不同的柔和颜色显示每个响应的几何形状

    参数:
    filtered_df -- 包含要显示数据的DataFrame
    highlighted_id -- 要高亮显示的行ID（如果有）
    """
    fig = go.Figure()

    # 如果没有数据，返回空图
    if filtered_df.empty:
        print("Warning: Filtered DataFrame is empty")
        return fig

    # 获取唯一响应进行颜色分配
    unique_responses = filtered_df['Response'].unique()
    print(f"Creating map with {len(unique_responses)} unique responses")

    # 为每个响应分配唯一的柔和颜色
    response_colors = {}
    for i, response in enumerate(unique_responses):
        # 如果响应数量超过颜色数量，使用模运算循环使用颜色
        color_idx = i % len(pastel_color_palette)
        response_colors[response] = pastel_color_palette[color_idx]

    # 用于存储所有有效坐标的列表
    all_lats = []
    all_lons = []

    # 存储高亮坐标
    highlighted_lats = []
    highlighted_lons = []

    # 用于图例的跟踪
    legend_entries = set()

    # 处理每个行
    for i, (idx, row) in enumerate(filtered_df.iterrows()):
        try:
            # 跳过没有几何数据的行
            if pd.isna(row['geometry']) or not row['geometry']:
                continue

            # 获取响应
            response = row['Response']
            legend_name = f"Response {i + 1}"

            # 获取颜色
            rgb_color = response_colors.get(response, pastel_color_palette[0])

            # 检查是否高亮
            is_highlighted = highlighted_id is not None and row['id'] == highlighted_id

            # 创建hover文本
            olc_text = f"OLC: {row['Open Location Code']}" if not pd.isna(row['Open Location Code']) else "OLC: N/A"
            hover_text = f"Response: {response}<br>{olc_text}<br>ID: {row['id']}"

            # 设置回调的自定义数据
            customdata = [int(row['id'])]

            # 尝试解析几何数据
            try:
                geom = wkt.loads(row['geometry'])

                # 根据几何类型处理
                if geom.geom_type == 'Polygon':
                    # 对于多边形，使用外部环
                    coords = list(geom.exterior.coords)

                    # 这些坐标应该已经是WGS84(经度,纬度)格式
                    lons = [x for x, y in coords]
                    lats = [y for x, y in coords]

                    # 验证坐标有效性
                    valid_coords = [(lon, lat) for lon, lat in zip(lons, lats) if
                                    -180 <= lon <= 180 and -90 <= lat <= 90]
                    if not valid_coords:
                        print(f"No valid coordinates for polygon in row {i}")
                        continue

                    # 提取有效坐标
                    valid_lons = [lon for lon, _ in valid_coords]
                    valid_lats = [lat for _, lat in valid_coords]

                    # 保存所有有效坐标用于居中地图
                    all_lats.extend(valid_lats)
                    all_lons.extend(valid_lons)

                    # 应用高亮
                    if is_highlighted:
                        highlighted_lats = valid_lats.copy()
                        highlighted_lons = valid_lons.copy()

                        fig.add_trace(go.Scattermapbox(
                            fill=None,
                            lon=valid_lons,
                            lat=valid_lats,
                            mode='lines',  # 仅使用线条，无标记点
                            line={'color': 'black', 'width': 6},
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        line_color = 'yellow'
                        line_width = 3
                        fill_color = get_color_with_alpha(rgb_color, 0.7)
                    else:
                        line_color = get_color_with_alpha(rgb_color, 0.8)
                        line_width = 2
                        fill_color = get_color_with_alpha(rgb_color, 0.4)

                    # 添加到图例的条件
                    show_legend = legend_name not in legend_entries
                    if show_legend:
                        legend_entries.add(legend_name)

                    # 添加多边形
                    fig.add_trace(go.Scattermapbox(
                        fill='toself',
                        lon=valid_lons,
                        lat=valid_lats,
                        mode='lines',  # 仅使用线条，无标记点
                        line={'color': line_color, 'width': line_width},
                        fillcolor=fill_color,
                        name=legend_name,
                        text=hover_text,
                        hoverinfo='text',
                        customdata=customdata,
                        showlegend=show_legend
                    ))

                elif geom.geom_type == 'LineString':
                    # 对于线串，使用所有顶点
                    coords = list(geom.coords)
                    lons = [x for x, y in coords]
                    lats = [y for x, y in coords]

                    # 验证坐标有效性
                    valid_coords = [(lon, lat) for lon, lat in zip(lons, lats) if
                                    -180 <= lon <= 180 and -90 <= lat <= 90]
                    if not valid_coords:
                        print(f"No valid coordinates for linestring in row {i}")
                        continue

                    # 提取有效坐标
                    valid_lons = [lon for lon, _ in valid_coords]
                    valid_lats = [lat for _, lat in valid_coords]

                    # 保存所有有效坐标用于居中地图
                    all_lats.extend(valid_lats)
                    all_lons.extend(valid_lons)

                    # 应用高亮
                    if is_highlighted:
                        highlighted_lats = valid_lats.copy()
                        highlighted_lons = valid_lons.copy()

                        fig.add_trace(go.Scattermapbox(
                            mode='lines',
                            lon=valid_lons,
                            lat=valid_lats,
                            line={'color': 'black', 'width': 6},
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        line_color = 'yellow'
                        line_width = 3
                    else:
                        line_color = get_color_with_alpha(rgb_color, 0.8)
                        line_width = 2

                    # 添加到图例的条件
                    show_legend = legend_name not in legend_entries
                    if show_legend:
                        legend_entries.add(legend_name)

                    # 添加线条
                    fig.add_trace(go.Scattermapbox(
                        mode='lines',  # 仅使用线条，无标记点
                        lon=valid_lons,
                        lat=valid_lats,
                        line={'color': line_color, 'width': line_width},
                        name=legend_name,
                        text=hover_text,
                        hoverinfo='text',
                        customdata=customdata,
                        showlegend=show_legend
                    ))

                elif geom.geom_type == 'Point':
                    # 对于点，使用x和y坐标
                    lon, lat = geom.x, geom.y

                    # 验证坐标有效性
                    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                        print(f"Invalid coordinates for point in row {i}: ({lon}, {lat})")
                        continue

                    # 保存所有有效坐标用于居中地图
                    all_lats.append(lat)
                    all_lons.append(lon)

                    # 应用高亮
                    if is_highlighted:
                        highlighted_lats = [lat]
                        highlighted_lons = [lon]

                        fig.add_trace(go.Scattermapbox(
                            mode='markers',
                            lon=[lon],
                            lat=[lat],
                            marker={'color': 'black', 'size': 18},
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        marker_color = 'yellow'
                        marker_size = 12
                    else:
                        marker_color = get_color_with_alpha(rgb_color, 0.8)
                        marker_size = 10

                    # 添加到图例的条件
                    show_legend = legend_name not in legend_entries
                    if show_legend:
                        legend_entries.add(legend_name)

                    # 添加点
                    fig.add_trace(go.Scattermapbox(
                        mode='markers',
                        lon=[lon],
                        lat=[lat],
                        marker={'color': marker_color, 'size': marker_size},
                        name=legend_name,
                        text=hover_text,
                        hoverinfo='text',
                        customdata=customdata,
                        showlegend=show_legend
                    ))

                elif geom.geom_type == 'MultiPolygon':
                    # 对于多多边形，处理每个组件
                    for j, poly in enumerate(geom.geoms):
                        coords = list(poly.exterior.coords)
                        lons = [x for x, y in coords]
                        lats = [y for x, y in coords]

                        # 验证坐标有效性
                        valid_coords = [(lon, lat) for lon, lat in zip(lons, lats) if
                                        -180 <= lon <= 180 and -90 <= lat <= 90]
                        if not valid_coords:
                            print(f"No valid coordinates for multipolygon part in row {i}")
                            continue

                        # 提取有效坐标
                        valid_lons = [lon for lon, _ in valid_coords]
                        valid_lats = [lat for _, lat in valid_coords]

                        # 保存所有有效坐标用于居中地图
                        all_lats.extend(valid_lats)
                        all_lons.extend(valid_lons)

                        # 应用高亮
                        if is_highlighted:
                            highlighted_lats.extend(valid_lats)
                            highlighted_lons.extend(valid_lons)

                            fig.add_trace(go.Scattermapbox(
                                fill=None,
                                lon=valid_lons,
                                lat=valid_lats,
                                mode='lines',  # 仅使用线条，无标记点
                                line={'color': 'black', 'width': 6},
                                showlegend=False,
                                hoverinfo='skip'
                            ))

                            line_color = 'yellow'
                            line_width = 3
                            fill_color = get_color_with_alpha(rgb_color, 0.7)
                        else:
                            line_color = get_color_with_alpha(rgb_color, 0.8)
                            line_width = 2
                            fill_color = get_color_with_alpha(rgb_color, 0.4)

                        # 只为第一个组件显示图例
                        show_legend = j == 0 and legend_name not in legend_entries
                        if show_legend:
                            legend_entries.add(legend_name)

                        fig.add_trace(go.Scattermapbox(
                            fill='toself',
                            lon=valid_lons,
                            lat=valid_lats,
                            mode='lines',  # 仅使用线条，无标记点
                            line={'color': line_color, 'width': line_width},
                            fillcolor=fill_color,
                            name=legend_name,
                            text=hover_text,
                            hoverinfo='text',
                            customdata=customdata,
                            showlegend=show_legend
                        ))

            except Exception as e:
                print(f"Error parsing geometry for row {i}: {e}")
                continue

        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

    # 设置地图布局
    # 如果没有有效坐标，使用默认视图
    if not all_lats or not all_lons:
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=0, lon=0),
                zoom=1
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
    else:
        # 计算中心点
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)

        # 设置缩放级别
        if highlighted_id is not None and highlighted_lats and highlighted_lons:
            # 如果有高亮元素，以它为中心
            center_lat = sum(highlighted_lats) / len(highlighted_lats)
            center_lon = sum(highlighted_lons) / len(highlighted_lons)
            zoom = 16
        else:
            zoom = 15

        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            )
        )

    return fig


# 初始化 Dash
app = Dash(__name__, requests_pathname_prefix='/conceptual/')
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# 主页面布局
main_layout = html.Div([
    html.H1("Different Ideas for the Same Spaces", style={'textAlign': 'center'}),
    html.P("This dashboard groups together different ideas which are proposed for the same space",
           style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '30px'}),

    # 添加搜索框（居中放置）
    html.Div([
        html.Label("Search Open Location Code:"),
        dcc.Input(
            id="search-input",
            type="text",
            placeholder="Enter Open Location Code...",
            style={'marginLeft': '10px', 'padding': '5px', 'width': '300px'}
        ),
        html.Button(
            "Search",
            id="search-button",
            style={
                'marginLeft': '10px',
                'padding': '5px 15px',
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer'
            }
        )
    ], style={'margin': '20px 0', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),

    html.Div(id="main_table_div", children=[
        html.Table(
            id="main_table",
            style={'width': '80%', 'margin': 'auto', 'borderCollapse': 'collapse'},
            children=[
                html.Thead(
                    html.Tr([
                        html.Th("Open Location Code", style={
                            'backgroundColor': 'lightgrey',
                            'fontWeight': 'bold',
                            'padding': '10px',
                            'textAlign': 'center',
                            'border': '1px solid #ddd'
                        }),
                        html.Th("Category", style={
                            'backgroundColor': 'lightgrey',
                            'fontWeight': 'bold',
                            'padding': '10px',
                            'textAlign': 'center',
                            'border': '1px solid #ddd'
                        }),
                        html.Th("Count", style={
                            'backgroundColor': 'lightgrey',
                            'fontWeight': 'bold',
                            'padding': '10px',
                            'textAlign': 'center',
                            'border': '1px solid #ddd'
                        })
                    ])
                ),
                html.Tbody(id="table_body")
            ]
        )
    ])
])


# OLC 详情页面布局
# OLC 详情页面布局
def olc_detail_layout(olc):
    # 过滤数据
    filtered_df = df[df["Open Location Code"] == olc]

    if filtered_df.empty:
        return html.Div([
            # 返回按钮已移除
            html.H2("No data found for this location code", style={'textAlign': 'center', 'color': 'red'})
        ])

    # 创建表格行
    rows = []

    # 处理表格行
    for i, (idx, row) in enumerate(filtered_df.iterrows()):
        # 使用交互式样式
        interactive_row_style = {'cursor': 'pointer', 'transition': 'all 0.3s'}

        # 创建行
        rows.append(html.Tr([
            html.Td(row["Open Location Code"],
                    style={'border': '1px solid #ddd', 'padding': '10px', 'textAlign': 'center'}),
            html.Td(row["Category"], style={'border': '1px solid #ddd', 'padding': '10px', 'textAlign': 'center'}),
            html.Td(row["Response"], style={'border': '1px solid #ddd', 'padding': '10px', 'textAlign': 'left'}),
            html.Td(row["Open Location Code"],
                    style={'border': '1px solid #ddd', 'padding': '10px', 'textAlign': 'center'})
        ], id={'type': 'table-row', 'index': row['id']}, style=interactive_row_style))

    # 创建OLC详情页面
    return html.Div([
        # 返回链接已移除

        # 标题
        html.H2(f"Location: {olc}", style={'textAlign': 'center', 'margin': '20px'}),

        # 隐藏存储高亮ID的元素
        dcc.Store(id='highlighted-id', data=None),

        # 表格
        html.Div(
            html.Table(
                style={'width': '80%', 'margin': 'auto', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'},
                children=[
                    html.Thead(html.Tr([
                        html.Th("Open Location Code",
                                style={'backgroundColor': '#e9ecef', 'padding': '12px', 'textAlign': 'center'}),
                        html.Th("Category",
                                style={'backgroundColor': '#e9ecef', 'padding': '12px', 'textAlign': 'center'}),
                        html.Th("Response",
                                style={'backgroundColor': '#e9ecef', 'padding': '12px', 'textAlign': 'center'}),
                        html.Th("OLC", style={'backgroundColor': '#e9ecef', 'padding': '12px', 'textAlign': 'center'})
                    ])),
                    html.Tbody(id="olc-detail-table-body", children=rows)
                ]
            ),
            style={'marginBottom': '30px'}
        ),

        # 地图交互说明
        html.Div([
            html.P(
                "Click on a table row to highlight the corresponding location on the map. Click on a map element to highlight its row in the table.",
                style={'textAlign': 'center', 'fontStyle': 'italic', 'marginBottom': '20px', 'color': '#666'})
        ]),

        # 地图
        dcc.Graph(
            id='geometry-map',
            figure=create_map(filtered_df),
            style=map_style,
            clear_on_unhover=True
        )
    ])


# 类别详情页面布局
def detail_layout(olc, category):
    filtered_df = df[(df["Open Location Code"] == olc) & (df["Category"] == category)]

    if filtered_df.empty:
        return html.Div([
            # 返回按钮已移除
            html.H2("No data found for this location and category", style={'textAlign': 'center', 'color': 'red'})
        ])

    # 创建表格行
    rows = []
    category_span = len(filtered_df)

    # 处理表格行
    for i, (idx, row) in enumerate(filtered_df.iterrows()):
        # 使用交互式样式
        interactive_row_style = {'cursor': 'pointer', 'transition': 'all 0.3s'}

        if i == 0:
            # 第一行需要包含Category并设置rowSpan
            rows.append(html.Tr([
                html.Td(
                    category,
                    rowSpan=category_span,
                    style={
                        'textAlign': 'center',
                        'verticalAlign': 'middle',
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'backgroundColor': '#f9f9f9'
                    }
                ),
                html.Td(row["Idea Number"], style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'textAlign': 'center'
                }),
                html.Td(row["Response"], style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'textAlign': 'left'
                }),
                html.Td(row["Open Location Code"], style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'textAlign': 'center'
                })
            ], id={'type': 'table-row', 'index': row['id']}, style=interactive_row_style))
        else:
            # 后续行不包含Category
            rows.append(html.Tr([
                html.Td(row["Idea Number"], style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'textAlign': 'center'
                }),
                html.Td(row["Response"], style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'textAlign': 'left'
                }),
                html.Td(row["Open Location Code"], style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'textAlign': 'center'
                })
            ], id={'type': 'table-row', 'index': row['id']}, style=interactive_row_style))

    return html.Div([
        # 返回按钮已移除

        html.H2(f"Location: {olc}, Category: {category}", style={'textAlign': 'center'}),

        # 隐藏存储高亮ID的元素
        dcc.Store(id='highlighted-id', data=None),

        # 使用自定义HTML表格来实现合并单元格
        html.Div(
            html.Table(
                style={
                    'width': '80%',
                    'margin': 'auto',
                    'borderCollapse': 'collapse',
                    'marginTop': '20px'
                },
                children=[
                    html.Thead(
                        html.Tr([
                            html.Th("Category", style={
                                'backgroundColor': 'lightgrey',
                                'fontWeight': 'bold',
                                'padding': '10px',
                                'textAlign': 'center',
                                'border': '1px solid #ddd'
                            }),
                            html.Th("Idea Number", style={
                                'backgroundColor': 'lightgrey',
                                'fontWeight': 'bold',
                                'padding': '10px',
                                'textAlign': 'center',
                                'border': '1px solid #ddd'
                            }),
                            html.Th("Response", style={
                                'backgroundColor': 'lightgrey',
                                'fontWeight': 'bold',
                                'padding': '10px',
                                'textAlign': 'center',
                                'border': '1px solid #ddd'
                            }),
                            html.Th("OLC", style={
                                'backgroundColor': 'lightgrey',
                                'fontWeight': 'bold',
                                'padding': '10px',
                                'textAlign': 'center',
                                'border': '1px solid #ddd'
                            })
                        ])
                    ),
                    html.Tbody(id="detail-table-body", children=rows)
                ]
            ),
            style={'marginBottom': '30px'}
        ),

        # 地图交互说明
        html.Div([
            html.P(
                "Click on a table row to highlight the corresponding location on the map. Click on a map element to highlight its row in the table.",
                style={'textAlign': 'center', 'fontStyle': 'italic', 'marginBottom': '20px', 'color': '#666'})
        ]),

        # 地图
        dcc.Graph(
            id='geometry-map',
            figure=create_map(filtered_df),
            style=map_style,
            clear_on_unhover=True
        )
    ])


# 页面路由回调
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('url', 'search')]
)
def display_page(pathname, search):
    print(f"Path requested: {pathname}")  # 调试信息

    if pathname and ('detail' in pathname):
        params = parse_qs(search.lstrip('?'))
        olc = unquote(params.get('olc', [None])[0])
        category = unquote(params.get('category', [None])[0])

        print(f"Detail view requested for OLC: {olc}, category: {category}")  # 调试信息

        if olc and category:
            return detail_layout(olc, category)
        elif olc:
            return olc_detail_layout(olc)
        else:
            print(f"Missing parameters - OLC: {olc}, category: {category}")  # 调试信息
            return html.Div("Invalid Request")
    return main_layout


# 生成主表内容回调
@app.callback(
    Output("table_body", "children"),
    [Input('url', 'pathname'),
     Input("search-button", "n_clicks")],
    [State("search-input", "value")]
)
def update_table_body(_, search_clicks, search_value):
    # 基于搜索值筛选数据
    filtered_df_main = df_main
    if search_value:
        filtered_df_main = df_main[df_main["Open Location Code"].str.contains(search_value, case=False)]

    # 调试信息
    print(f"Generating table rows for {len(filtered_df_main)} groups")

    rows = []
    current_olc = None
    rowspan_count = 0

    for index, row in filtered_df_main.iterrows():
        olc = row["Open Location Code"]
        category = row["Category"]
        count = row["Count"]

        if olc != current_olc:
            current_olc = olc
            # 由于过滤了数据，我们需要重新计算同一OLC的出现次数
            rowspan_count = len(filtered_df_main[filtered_df_main["Open Location Code"] == olc])
            tr = html.Tr([
                html.Td(
                    # 直接显示OLC文本，不使用html.A创建链接，并设置为粗体
                    olc,
                    rowSpan=rowspan_count,
                    style={
                        'textAlign': 'center',
                        'verticalAlign': 'middle',
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'fontWeight': 'bold'  # 添加这一行使文本加粗
                    }
                ),
                html.Td(
                    html.A(
                        category,
                        href=f"detail?olc={quote(olc)}&category={quote(category)}",
                        style={'textDecoration': 'none', 'color': '#0066cc'}
                    ),
                    style={
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'textAlign': 'center'
                    }
                ),
                html.Td(
                    count,
                    style={
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'textAlign': 'center'
                    }
                )
            ])
        else:
            tr = html.Tr([
                html.Td(
                    html.A(
                        category,
                        href=f"detail?olc={quote(olc)}&category={quote(category)}",
                        style={'textDecoration': 'none', 'color': '#0066cc'}
                    ),
                    style={
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'textAlign': 'center'
                    }
                ),
                html.Td(
                    count,
                    style={
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'textAlign': 'center'
                    }
                )
            ])
        rows.append(tr)

    if not rows:
        rows = [
            html.Tr(html.Td("No matching records found", colSpan=3, style={'textAlign': 'center', 'padding': '20px'}))]

    return rows


# 回调函数：更新高亮行
@app.callback(
    Output('highlighted-id', 'data'),
    [Input({'type': 'table-row', 'index': ALL}, 'n_clicks'),
     Input('geometry-map', 'clickData')],
    [State('highlighted-id', 'data'),
     State('url', 'search')]
)
def update_highlighted_id(row_clicks, map_click_data, current_highlighted, search):
    # 获取回调上下文
    ctx = callback_context
    if not ctx.triggered:
        return None

    # 获取触发回调的组件ID
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # 如果点击了表格行
    if 'table-row' in trigger_id:
        # 从触发ID中提取行ID
        try:
            trigger_dict = json.loads(trigger_id)
            clicked_id = trigger_dict['index']

            # 切换高亮：如果再次点击相同行，取消高亮
            if current_highlighted == clicked_id:
                return None
            return clicked_id
        except Exception as e:
            print(f"Error processing table row click: {e}")
            return current_highlighted

    # 如果点击了地图元素
    elif 'geometry-map' in trigger_id and map_click_data:
        try:
            # 从点击点的customdata中提取行ID
            clicked_id = map_click_data['points'][0]['customdata'][0]

            # 切换高亮：如果再次点击相同元素，取消高亮
            if current_highlighted == clicked_id:
                return None
            return clicked_id
        except Exception as e:
            print(f"Error processing map click: {e}")
            return current_highlighted

    # 如果其他内容触发了回调
    return current_highlighted


# 更新行样式的回调函数
@app.callback(
    Output({'type': 'table-row', 'index': ALL}, 'style'),
    [Input('highlighted-id', 'data')],
    [State({'type': 'table-row', 'index': ALL}, 'id')]
)
def update_row_styles(highlighted_id, row_ids):
    # 默认样式
    styles = []

    # 基础样式
    base_style = {
        'cursor': 'pointer',
        'transition': 'all 0.3s'
    }

    # 高亮样式
    highlight_style = {
        'cursor': 'pointer',
        'transition': 'all 0.3s',
        'backgroundColor': 'rgba(255, 255, 0, 0.3)',  # 黄色高亮背景
        'boxShadow': 'inset 0 0 0 2px #FFD700',  # 内部金色边框
        'fontWeight': 'bold',
        'transform': 'translateX(5px)'  # 轻微缩进效果
    }

    # 为每行应用样式
    for row_id_dict in row_ids:
        row_id = row_id_dict['index']
        if highlighted_id is not None and row_id == highlighted_id:
            styles.append(highlight_style)
        else:
            styles.append(base_style)

    return styles


# 更新地图的回调函数
@app.callback(
    Output('geometry-map', 'figure'),
    [Input('highlighted-id', 'data')],
    [State('url', 'search')]
)
def update_map(highlighted_id, search):
    # 获取参数
    params = parse_qs(search.lstrip('?'))
    olc = unquote(params.get('olc', [None])[0])
    category = unquote(params.get('category', [None])[0])

    # 过滤数据
    if olc and category:
        filtered_df = df[(df["Open Location Code"] == olc) & (df["Category"] == category)]
    elif olc:
        filtered_df = df[df["Open Location Code"] == olc]
    else:
        filtered_df = pd.DataFrame()

    # 创建地图并应用高亮效果
    return create_map(filtered_df, highlighted_id)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8053)