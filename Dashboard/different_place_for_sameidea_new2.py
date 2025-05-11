from dash import Dash, dcc, html, dash_table, ALL, callback_context, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from urllib.parse import parse_qs, unquote, quote
import json
import numpy as np
import io
import re
import colorsys  # 导入用于颜色处理的库


# 函数来读取CSV数据
def read_data():
    try:
        # 读取CSV文件
        df = pd.read_csv('output_geography_grouped_2.csv')
        # 重命名列以便显示
        df = df.rename(columns={
            'theme': 'Theme',
            'place_group': 'Groups',
            'comment': 'Comment',
            'wrong': 'Wrong',
            'Open Location Code': 'OLCs'
        })
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        # 如果出错，返回空的DataFrame
        return pd.DataFrame(columns=['Theme', 'Groups', 'Comment', 'Wrong', 'OLCs', 'geometry'])


# 读取数据
df = read_data()

# 创建主表的DataFrame
df_main = df[['Theme', 'Groups']].drop_duplicates().reset_index(drop=True)

# 计算每个主题和分组的响应数量
group_counts = df.groupby(['Theme', 'Groups']).size().reset_index(name='Count')

# 合并计数数据到主表
df_main = pd.merge(df_main, group_counts[['Theme', 'Groups', 'Count']], on=['Theme', 'Groups'], how='left')

# 计算每个主题的行跨度
df_main["RowSpan"] = df_main.groupby("Theme")["Groups"].transform("count")

# 创建主题描述映射 - 使用正确的键名（带空格）
theme_descriptions = {
    "Theme 0": "Theme 0 (Water Service)",
    "Theme 1": "Theme 1 (Parking)",
    "Theme 2": "Theme 2 (Dog Park)",
    "Theme 3": "Theme 3 (Sport Space/Public Utilities)",
    "Theme 4": "Theme 4 (Garden)",
    "Theme 5": "Theme 5 (Play Ground)",
    "Theme 6": "Theme 6 (Shade Structure)",
    "Theme 7": "Theme 7 (Bike Pump Track)",
    "Theme 8": "Theme 8 (Entrance)"
}


# Function to generate a fixed set of base colors for themes
def get_theme_colors(themes):
    """Generate a dictionary mapping themes to distinct base colors"""
    theme_colors = {}
    # Use a fixed set of distinct hues for themes
    distinct_hues = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]  # 12 hues

    for i, theme in enumerate(themes):
        hue = distinct_hues[i % len(distinct_hues)]
        # Convert HSV to RGB (S=1, V=0.8 for a rich but not too bright base color)
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, 1.0, 0.8)]
        theme_colors[theme] = (r, g, b)

    return theme_colors


# Function to adjust the brightness of a color
def adjust_brightness(base_color, brightness_factor):
    """Adjust brightness of a base RGB color"""
    r, g, b = base_color
    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    # Adjust brightness (V value)
    v = min(1.0, max(0.0, v * brightness_factor))
    # Convert back to RGB
    r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)]
    return (r, g, b)


# Generate color mapping for themes
unique_themes = df['Theme'].unique()
theme_colors = get_theme_colors(unique_themes)


# Function to generate group color variations
def get_group_colors(theme, groups):
    """Generate color variations for groups within a theme"""
    base_color = theme_colors.get(theme, (128, 128, 128))  # Default to gray if not found
    group_colors = {}

    # Generate brightness factors that range from 0.7 to 1.3
    # This gives variations from darker to brighter than the base color
    brightness_range = np.linspace(0.7, 1.3, len(groups))

    for i, group in enumerate(groups):
        brightness = brightness_range[i]
        group_colors[group] = adjust_brightness(base_color, brightness)

    return group_colors


# Pre-compute group colors for each theme
theme_group_colors = {}
for theme in unique_themes:
    theme_groups = df[df['Theme'] == theme]['Groups'].unique()
    theme_group_colors[theme] = get_group_colors(theme, theme_groups)


# 函数解析WKT多边形几何并提取坐标
def parse_wkt_polygon(wkt_geometry):
    if pd.isna(wkt_geometry) or not isinstance(wkt_geometry, str):
        return None

    # 从WKT POLYGON格式提取坐标
    try:
        # 从POLYGON字符串提取坐标部分
        coords_match = re.search(r'POLYGON\s*\(\((.*)\)\)', wkt_geometry)
        if not coords_match:
            coords_match = re.search(r'POLYGON\s*\((.*)\)', wkt_geometry)
            if not coords_match:
                return None

        coords_str = coords_match.group(1)

        # 解析坐标
        points = []
        for point_str in coords_str.split(','):
            point_parts = point_str.strip().split()
            if len(point_parts) >= 2:
                x = float(point_parts[0])
                y = float(point_parts[1])
                points.append((x, y))

        # 提取经纬度
        lngs = [point[0] for point in points]
        lats = [point[1] for point in points]

        # 计算中心点
        center_lng = sum(lngs) / len(lngs)
        center_lat = sum(lats) / len(lats)

        return {
            'lngs': lngs,
            'lats': lats,
            'center_lng': center_lng,
            'center_lat': center_lat
        }
    except Exception as e:
        print(f"Error parsing WKT polygon: {e}")
        return None


# 函数将Web Mercator坐标转换为经纬度
def webmercator_to_latlng(x, y):
    # 转换常数
    R = 6378137.0  # 地球半径（米）

    # 转换经度
    lng = (x * 180) / (R * np.pi)

    # 转换纬度
    lat = np.arcsin(np.tanh(y / R)) * 180 / np.pi

    return lat, lng


# 函数创建地图 - 使用enhanced-location-dashboard的颜色方案和高亮逻辑
def create_plotly_map(df_filtered, selected_olc=None):
    fig = go.Figure()
    all_lats = []
    all_lngs = []

    # 保存选中要素的坐标用于视图聚焦
    selected_lats = []
    selected_lngs = []

    # 获取当前主题和分组
    if not df_filtered.empty:
        current_theme = df_filtered['Theme'].iloc[0]
        current_group = df_filtered['Groups'].iloc[0]

        # 获取该分组的基本颜色
        group_color = theme_group_colors.get(current_theme, {}).get(current_group, (128, 128, 128))
    else:
        group_color = (128, 128, 128)  # 默认灰色

    for i, (index, row) in enumerate(df_filtered.iterrows()):
        olc_code = row['OLCs']

        # 解析几何数据
        is_web_mercator = False
        if 'geometry' in row and not pd.isna(row['geometry']):
            polygon = parse_wkt_polygon(row['geometry'])
            # 检查是否可能是Web Mercator坐标
            if polygon and polygon['lngs'] and abs(polygon['lngs'][0]) > 180:
                is_web_mercator = True
        else:
            polygon = None

        if polygon:
            # 如果需要，将Web Mercator坐标转换为经纬度
            if is_web_mercator:
                converted_lats = []
                converted_lngs = []
                for lng, lat in zip(polygon['lngs'], polygon['lats']):
                    conv_lat, conv_lng = webmercator_to_latlng(lng, lat)
                    converted_lats.append(conv_lat)
                    converted_lngs.append(conv_lng)

                # 用转换后的坐标更新多边形
                polygon['lats'] = converted_lats
                polygon['lngs'] = converted_lngs
                polygon['center_lat'] = sum(converted_lats) / len(converted_lats)
                polygon['center_lng'] = sum(converted_lngs) / len(converted_lngs)

            # 确定这是否为选中的OLC
            is_selected = selected_olc == olc_code

            # 记录所有坐标以便边界框计算
            all_lats.extend(polygon['lats'])
            all_lngs.extend(polygon['lngs'])

            # 如果是选中的要素，保存其坐标用于后续聚焦
            if is_selected:
                selected_lats = polygon['lats'].copy()
                selected_lngs = polygon['lngs'].copy()

            # 使用在预计算颜色中稍微调整亮度，为每个OLC创建变体
            brightness_factor = 0.9 + (i * 0.1 % 0.4)  # 在0.9和1.3之间变化
            rgb_color = adjust_brightness(group_color, brightness_factor)

            # 转换为rgba字符串
            r, g, b = rgb_color

            # 应用方案1: 为选中的要素添加双重轮廓效果
            if is_selected:
                # 首先添加一个较宽的黑色外轮廓
                fig.add_trace(go.Scattermapbox(
                    mode='lines',
                    lon=polygon['lngs'],
                    lat=polygon['lats'],
                    fill=None,
                    line=dict(width=6, color='black'),  # 宽黑色外轮廓
                    showlegend=False,
                    hoverinfo='skip'
                ))

                # 内部轮廓使用黄色高亮和较宽的线条
                line_width = 3
                line_color = 'yellow'  # 黄色高亮
                # 选中项使用更不透明的填充色
                fill_color = f"rgba({r}, {g}, {b}, 0.6)"
            else:
                # 非选中项使用正常样式
                line_width = 1.5
                line_color = f"rgba({r}, {g}, {b}, 0.8)"
                fill_color = f"rgba({r}, {g}, {b}, 0.4)"

            # 添加多边形
            fig.add_trace(go.Scattermapbox(
                mode='lines',
                lon=polygon['lngs'],
                lat=polygon['lats'],
                fill='toself',
                fillcolor=fill_color,
                line=dict(width=line_width, color=line_color),
                name=olc_code,
                hoverinfo='text',
                hovertext=f"<b>OLC:</b> {olc_code}<br><b>Group:</b> {row['Groups']}<br><b>Comment:</b> {row['Comment']}<br><b>Wrong:</b> {row['Wrong']}"
            ))

    # 设置地图边界和中心 - 应用方案7: 自动聚焦于选中的要素
    if all_lats and all_lngs:
        # 默认情况下使用所有要素的中心点
        center_lat = sum(all_lats) / len(all_lats)
        center_lng = sum(all_lngs) / len(all_lngs)

        # 基本缩放级别
        zoom = 12

        # 如果有选中的要素，则聚焦于该要素
        if selected_olc is not None and selected_lats and selected_lngs:
            # 使用选中要素的中心点
            center_lat = sum(selected_lats) / len(selected_lats)
            center_lng = sum(selected_lngs) / len(selected_lngs)
            # 选中要素时使用更高的缩放级别
            zoom = 16
        else:
            # 根据数据范围动态确定缩放级别
            lat_range = max(all_lats) - min(all_lats)
            lng_range = max(all_lngs) - min(all_lngs)

            if max(lat_range, lng_range) < 0.01:
                zoom = 16
            elif max(lat_range, lng_range) < 0.1:
                zoom = 14
            elif max(lat_range, lng_range) < 1:
                zoom = 12
            else:
                zoom = 10

        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                zoom=zoom,
                center=dict(lat=center_lat, lon=center_lng)
            ),
            margin=dict(r=0, l=0, t=0, b=0),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.7)"
            )
        )
    else:
        # 如果没有坐标的默认视图
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                zoom=2,
                center=dict(lat=40, lon=-100)
            ),
            margin=dict(r=0, l=0, t=0, b=0)
        )
        fig.add_annotation(text="No valid data to display", showarrow=False)

    return fig


# 定义样式
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
    'borderRadius': '5px',
    'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
}

# 初始化Dash应用
app = Dash(__name__, requests_pathname_prefix='/different/')
app.config.suppress_callback_exceptions = True

# 修改主页面布局 - 新的标题和描述，添加Count列，移除下载按钮
main_layout = html.Div([
    html.H1("Different places for same idea",
            style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '20px'}),
    html.P(
        "This dashboard shows that does a theme repeat in different places",
        style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '30px', 'color': '#555'}
    ),

    # 搜索输入框
    html.Div([
        dcc.Input(id="search-input", type="text", placeholder="Enter Theme or Group",
                  style={'padding': '8px', 'width': '250px', 'border': '1px solid #ddd', 'borderRadius': '4px'}),
        html.Button(
            "Search",
            id="search-button",
            style={
                'marginLeft': '10px',
                'padding': '8px 16px',
                'backgroundColor': '#007bff',
                'color': 'white',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer'
            }
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # 表格 - 添加Count列
    html.Div(
        html.Table(
            style={'width': '80%', 'margin': 'auto', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'},
            children=[
                html.Thead(
                    html.Tr([
                        html.Th("Theme", style=header_style),
                        html.Th("Groups", style=header_style),
                        html.Th("Count", style=header_style)
                    ])
                ),
                html.Tbody(id="table-body")
            ]
        )
    )
    # 下载按钮部分已被移除
])


# 详情页面布局函数
def detail_layout(theme, group):
    filtered_df = df[(df["Theme"] == theme) & (df["Groups"] == group)]

    # 提取唯一值
    group_value = filtered_df['Groups'].iloc[0] if not filtered_df.empty else ""

    # 行数
    num_rows = len(filtered_df)

    # 获取主题颜色
    theme_color = theme_colors.get(theme, (128, 128, 128))
    r, g, b = theme_color
    theme_color_str = f'rgb({r}, {g}, {b})'
    theme_bg_color = f'rgba({r}, {g}, {b}, 0.1)'

    # 获取分组颜色
    group_color = theme_group_colors.get(theme, {}).get(group, (128, 128, 128))
    r, g, b = group_color
    group_color_str = f'rgb({r}, {g}, {b})'
    group_bg_color = f'rgba({r}, {g}, {b}, 0.2)'

    # 创建OLC单元格，带点击事件
    olc_cells = []
    for i in range(num_rows):
        olc_value = filtered_df['OLCs'].iloc[i] if i < len(filtered_df) else ""
        # 为每个OLC单元格创建可点击按钮
        olc_cell = html.Td(
            html.Button(
                olc_value,
                id={'type': 'olc-button', 'index': i},
                style={
                    'width': '100%',
                    'textAlign': 'center',
                    'backgroundColor': 'transparent',
                    'border': 'none',
                    'cursor': 'pointer',
                    'fontFamily': 'inherit',
                    'fontSize': 'inherit',
                    'padding': '0',
                    'color': group_color_str,
                    'fontWeight': 'bold',
                    'transition': 'all 0.3s'
                }
            ),
            style=cell_style
        )
        olc_cells.append(olc_cell)

    # 创建评论单元格
    comment_cells = []
    for i in range(num_rows):
        comment_value = filtered_df['Comment'].iloc[i] if i < len(filtered_df) else ""
        comment_cell = html.Td(comment_value, style=cell_style)
        comment_cells.append(comment_cell)

    # 创建wrong单元格
    wrong_cells = []
    for i in range(num_rows):
        wrong_value = filtered_df['Wrong'].iloc[i] if i < len(filtered_df) else ""
        wrong_cell = html.Td(str(wrong_value), style=cell_style)
        wrong_cells.append(wrong_cell)

    # 创建表格行
    table_rows = []

    # 第一行包含合并单元格
    first_row = html.Tr([
        # 合并单元格 - Groups，使用主题相关颜色
        html.Td(
            group_value,
            rowSpan=num_rows,
            style={
                'border': '1px solid #ddd',
                'padding': '10px',
                'textAlign': 'center',
                'verticalAlign': 'middle',
                'backgroundColor': group_bg_color,
                'color': group_color_str,
                'fontWeight': 'bold'
            }
        ),
        # 第一行的Comment、Wrong和OLCs
        comment_cells[0] if comment_cells else html.Td("", style=cell_style),
        wrong_cells[0] if wrong_cells else html.Td("", style=cell_style),
        olc_cells[0] if olc_cells else html.Td("", style=cell_style)
    ])
    table_rows.append(first_row)

    # 添加剩余行
    for i in range(1, num_rows):
        row = html.Tr([
            comment_cells[i] if i < len(comment_cells) else html.Td("", style=cell_style),
            wrong_cells[i] if i < len(wrong_cells) else html.Td("", style=cell_style),
            olc_cells[i] if i < len(olc_cells) else html.Td("", style=cell_style)
        ])
        table_rows.append(row)

    # 创建Plotly地图
    fig = create_plotly_map(filtered_df)

    return html.Div([
        # 返回按钮已移除

        html.H2(
            f"Details for {theme} - {group}",
            style={
                'textAlign': 'center',
                'color': theme_color_str,
                'borderBottom': f'3px solid {theme_color_str}',
                'paddingBottom': '8px',
                'display': 'inline-block',
                'margin': '20px auto',
                'fontWeight': 'bold',
                'width': 'auto'
            }
        ),

        # 添加交互提示信息
        html.Div(
            html.P(
                "Click on an OLC code to highlight its location on the map. The map will automatically focus on the selected area.",
                style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#666', 'marginBottom': '20px'}
            )
        ),

        # 带合并单元格的表格
        html.Div(
            html.Table(
                style={'width': '100%', 'borderCollapse': 'collapse', 'margin': '20px'},
                children=[
                    # 表头
                    html.Thead(
                        html.Tr([
                            html.Th("Groups", style=header_style),
                            html.Th("Comment", style=header_style),
                            html.Th("Wrong", style=header_style),
                            html.Th("OLCs", style=header_style)
                        ])
                    ),
                    # 表体
                    html.Tbody(table_rows)
                ]
            )
        ),

        # 存储选中的OLC
        dcc.Store(id='selected-row-data', data=""),

        # 地图容器
        html.Div([
            html.H3("Geographic Visualization",
                    style={'textAlign': 'center', 'fontWeight': 'bold', 'margin': '10px 0'}),
            dcc.Graph(
                id='geometry-map',
                figure=fig,
                style={'height': '500px'}
            )
        ], style=map_style)
    ], style={'textAlign': 'center'})


# 回调函数
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('url', 'search')]
)
def display_page(pathname, search):
    if '/detail' in pathname:  # 匹配"/different/detail"
        params = parse_qs(search.lstrip('?'))
        theme = params.get('theme', [None])[0]
        group = params.get('group', [None])[0]

        # 只有当值不是None时才解码
        if theme is not None:
            theme = unquote(theme)
        if group is not None:
            group = unquote(group)

        return detail_layout(theme, group) if theme and group else main_layout
    return main_layout


# 更新表格主体 - 修改以包含Count列并显示描述性主题名称
@app.callback(
    Output("table-body", "children"),
    [Input('url', 'pathname'),
     Input('search-button', 'n_clicks')],
    [State('search-input', 'value')]
)
def update_table_body(_, n_clicks, search_value):
    # 基于搜索值筛选数据
    filtered_df = df_main
    if search_value:
        filtered_df = df_main[
            (df_main["Theme"].str.contains(search_value, case=False)) |
            (df_main["Groups"].str.contains(search_value, case=False))
            ]

    rows = []
    current_theme = None

    # 调试信息
    print(f"Generating table rows for {len(filtered_df)} groups")

    for index, row in filtered_df.iterrows():
        theme = row["Theme"]
        group = row["Groups"]
        count = row["Count"]

        # 构造详细视图的URL
        detail_url = f"detail?theme={quote(theme)}&group={quote(group)}"

        if theme != current_theme:
            current_theme = theme
            rowspan = int(row["RowSpan"])

            # 使用描述性标题（如果有映射）
            display_theme = theme_descriptions.get(theme, theme)

            rows.append(html.Tr([
                html.Td(
                    display_theme,  # 使用带描述的主题名称
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
                        group,
                        href=detail_url,
                        style=link_style
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
                        group,
                        href=detail_url,
                        style=link_style
                    ),
                    style=cell_style
                ),
                html.Td(
                    count,
                    style=cell_style
                )
            ]))

    if not rows:
        rows = [
            html.Tr(html.Td("No matching records found", colSpan=3, style={'textAlign': 'center', 'padding': '20px'}))]

    return rows


# 处理OLC按钮点击 - 使用enhanced-location-dashboard的高亮样式
@app.callback(
    [Output('selected-row-data', 'data'),
     Output({'type': 'olc-button', 'index': ALL}, 'style')],
    [Input({'type': 'olc-button', 'index': ALL}, 'n_clicks')],
    [State('url', 'search'),
     State('selected-row-data', 'data'),
     State({'type': 'olc-button', 'index': ALL}, 'id')]
)
def handle_olc_button_click(n_clicks, search, current_selected, button_ids):
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks):
        # 返回所有按钮的默认样式
        default_styles = []
        for _ in button_ids:
            default_styles.append({
                'width': '100%',
                'textAlign': 'center',
                'backgroundColor': 'transparent',
                'border': 'none',
                'cursor': 'pointer',
                'fontFamily': 'inherit',
                'fontSize': 'inherit',
                'padding': '0',
                'color': '#0066cc',
                'fontWeight': 'bold',
                'transition': 'all 0.3s'
            })
        return "", default_styles

    # 获取被点击按钮的ID
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if not button_id:
        default_styles = []
        for _ in button_ids:
            default_styles.append({
                'width': '100%',
                'textAlign': 'center',
                'backgroundColor': 'transparent',
                'border': 'none',
                'cursor': 'pointer',
                'fontFamily': 'inherit',
                'fontSize': 'inherit',
                'padding': '0',
                'color': '#0066cc',
                'fontWeight': 'bold',
                'transition': 'all 0.3s'
            })
        return "", default_styles

    try:
        # 解析按钮索引
        button_id_dict = json.loads(button_id)
        button_index = button_id_dict.get('index', 0)

        # 获取当前主题和分组
        params = parse_qs(search.lstrip('?'))
        theme = params.get('theme', [None])[0]
        group = params.get('group', [None])[0]

        # 只有当值不是None时才解码
        if theme is not None:
            theme = unquote(theme)
        if group is not None:
            group = unquote(group)

        if theme and group:
            filtered_df = df[(df["Theme"] == theme) & (df["Groups"] == group)]
            if button_index < len(filtered_df):
                selected_row = filtered_df.iloc[button_index]
                selected_olc = selected_row['OLCs']

                # 获取分组颜色
                group_color = theme_group_colors.get(theme, {}).get(group, (128, 128, 128))
                r, g, b = group_color
                group_color_str = f'rgb({r}, {g}, {b})'

                # 实现切换逻辑 - 如果再次点击同一个OLC，则取消选择
                if current_selected == selected_olc:
                    selected_olc = ""

                # 为所有按钮创建样式，突出显示选中的按钮
                styles = []
                for i, btn_id in enumerate(button_ids):
                    index = btn_id.get('index', -1)

                    if index == button_index and selected_olc != "":
                        # 选中按钮的高亮样式 - 使用增强对比度和边框效果
                        styles.append({
                            'width': '100%',
                            'textAlign': 'center',
                            'backgroundColor': 'rgba(255, 255, 0, 0.3)',  # 黄色高亮背景
                            'border': 'none',
                            'cursor': 'pointer',
                            'fontFamily': 'inherit',
                            'fontSize': 'inherit',
                            'padding': '5px',
                            'color': '#000',  # 更深的文本颜色
                            'fontWeight': 'bold',
                            'boxShadow': 'inset 0 0 0 2px #FFD700',  # 内部金色边框
                            'borderRadius': '4px',
                            'transform': 'translateX(5px)',  # 轻微缩进效果
                            'transition': 'all 0.3s'
                        })
                    else:
                        # 非选中按钮的普通样式
                        styles.append({
                            'width': '100%',
                            'textAlign': 'center',
                            'backgroundColor': 'transparent',
                            'border': 'none',
                            'cursor': 'pointer',
                            'fontFamily': 'inherit',
                            'fontSize': 'inherit',
                            'padding': '0',
                            'color': group_color_str,
                            'fontWeight': 'bold',
                            'transition': 'all 0.3s'
                        })

                return selected_olc, styles
    except Exception as e:
        print(f"Error handling button click: {str(e)}")

    # 如果出现错误，返回所有按钮的默认样式
    default_styles = []
    for _ in button_ids:
        default_styles.append({
            'width': '100%',
            'textAlign': 'center',
            'backgroundColor': 'transparent',
            'border': 'none',
            'cursor': 'pointer',
            'fontFamily': 'inherit',
            'fontSize': 'inherit',
            'padding': '0',
            'color': '#0066cc',
            'fontWeight': 'bold',
            'transition': 'all 0.3s'
        })
    return "", default_styles


# 基于选中的OLC更新地图
@app.callback(
    Output('geometry-map', 'figure'),
    [Input('selected-row-data', 'data'),
     Input('url', 'search')]
)
def update_map(selected_olc, search):
    try:
        params = parse_qs(search.lstrip('?'))
        theme = params.get('theme', [None])[0]
        group = params.get('group', [None])[0]

        # 只有当值不是None时才解码
        if theme is not None:
            theme = unquote(theme)
        if group is not None:
            group = unquote(group)

        if theme and group:
            filtered_df = df[(df["Theme"] == theme) & (df["Groups"] == group)]
            return create_plotly_map(filtered_df, selected_olc)
    except Exception as e:
        print(f"Error updating map: {str(e)}")

    return go.Figure()


# 下载数据回调已被移除

# 应用布局
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8054)