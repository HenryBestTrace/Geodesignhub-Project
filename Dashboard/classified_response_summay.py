from dash import Dash, dcc, html, ALL, callback_context
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
try:
    df = pd.read_csv("./classified_response_summaries.csv")
    print("Using new data file: classified_response_summaries.csv")
    print(f"Columns in data: {df.columns.tolist()}")

    # 检查几何数据
    if 'geometry' in df.columns:
        geom_count = df['geometry'].notnull().sum()
        print(f"Found {geom_count} rows with geometry data")

        # 测试几何数据解析
        if geom_count > 0:
            sample = df.loc[df['geometry'].notnull()].iloc[0]
            try:
                geom = wkt.loads(sample['geometry'])
                print(f"Sample geometry type: {geom.geom_type}")
            except Exception as e:
                print(f"Error parsing sample geometry: {e}")

    # 检查OLC列
    if 'Open Location Code' in df.columns:
        olc_count = df['Open Location Code'].notnull().sum()
        print(f"Found {olc_count} rows with Open Location Code")
except Exception as e:
    print(f"Error loading data: {e}")
    # 回退到旧数据文件
    try:
        df = pd.read_csv("./classified_response_summaries2.csv")
        print("Using fallback data file: classified_response_summaries2.csv")
    except Exception as e:
        print(f"Error loading fallback data: {e}")
        # 创建一个空DataFrame以避免错误
        df = pd.DataFrame(
            columns=['Category', 'Groups', 'Summary', 'Response', 'Upvotes', 'Downvotes', 'Open Location Code',
                     'geometry'])

# 删除Category列中的emoji
df['Category'] = df['Category'].apply(remove_emoji)

# 确保数据有唯一标识符
if 'id' not in df.columns:
    df['id'] = range(len(df))

# 处理主表：去重 Category + Groups (子分类)
df_main = df[['Category', 'Groups']].drop_duplicates().reset_index(drop=True)
df_main = df_main.rename(columns={'Groups': 'Sub-category'})

# 计算每个子分类的响应数量
subcategory_counts = df.groupby(['Category', 'Groups']).size().reset_index(name='Count')
subcategory_counts = subcategory_counts.rename(columns={'Groups': 'Sub-category'})

# 合并计数到主表
df_main = pd.merge(df_main, subcategory_counts[['Category', 'Sub-category', 'Count']],
                   on=['Category', 'Sub-category'], how='left')


# 子分类代码格式化函数
def format_subcategory_code(category, subcategories):
    """
    根据类别名称格式化子分类代码
    将子分类代码格式化为：类别前三个字母 + "-" + 两位数序号（01, 02, ...）
    """
    # 取类别名称的前三个字母
    prefix = ''.join([c for c in category[:3] if c.isalpha()]).capitalize()

    # 为每个子分类分配新的格式化代码
    formatted_codes = {}

    # 从1开始为每个子分类分配序号
    for i, sub in enumerate(subcategories, 1):
        formatted_codes[sub] = f"{prefix}-{i:02d}"

    return formatted_codes


# 处理主表数据
# 首先获取每个类别的所有子分类
categories = df['Category'].unique()
for category in categories:
    # 获取该类别下的所有子分类
    subcategories = df[df['Category'] == category]['Groups'].unique()

    # 创建格式化的子分类代码
    formatted_codes = format_subcategory_code(category, subcategories)

    # 在原始DataFrame中更新子分类代码
    for old_code, new_code in formatted_codes.items():
        # 更新主表中的代码
        df_main.loc[
            (df_main['Category'] == category) & (df_main['Sub-category'] == old_code), 'Sub-category'] = new_code

        # 更新原始数据中的代码
        df.loc[(df['Category'] == category) & (df['Groups'] == old_code), 'Groups'] = new_code

# 计算每个类别的行跨度（用于表格渲染）
df_main["RowSpan"] = df_main.groupby("Category")["Sub-category"].transform("count")


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

# 地图样式设置
map_style = {
    'height': '500px',
    'margin': '20px',
    'border': '1px solid #ddd',
    'borderRadius': '5px'
}


# 直接参考enhanced-location-dashboard-modified.py的地图创建函数
# 直接参考enhanced-location-dashboard-modified.py的地图创建函数
def create_map(filtered_df, highlighted_id=None):
    """
    创建一个地图，使用不同的柔和颜色显示每个响应的几何形状
    """
    fig = go.Figure()

    # 如果没有数据，返回空图
    if filtered_df.empty:
        print("Warning: Filtered DataFrame is empty")
        return fig

    # 获取唯一响应
    unique_responses = filtered_df['Response'].unique()
    print(f"Creating map with {len(unique_responses)} unique responses")

    # 为每个响应分配颜色
    response_colors = {}
    for i, response in enumerate(unique_responses):
        color_idx = i % len(pastel_color_palette)
        response_colors[response] = pastel_color_palette[color_idx]

    # 检查是否存在几何数据
    has_geometry = 'geometry' in filtered_df.columns and filtered_df['geometry'].notnull().any()

    # 用于标记图例的集合
    legend_entries = set()

    # 用于存储有效坐标点的列表
    all_lats = []
    all_lons = []

    # 高亮坐标点
    highlighted_lats = []
    highlighted_lons = []

    if has_geometry:
        # 处理几何数据
        for i, (idx, row) in enumerate(filtered_df.iterrows()):
            try:
                # 跳过没有几何数据的行
                if pd.isna(row['geometry']) or not row['geometry']:
                    continue

                # 获取响应
                response = row['Response']
                legend_name = f"Response {i + 1}"

                # 是否高亮
                is_highlighted = highlighted_id is not None and row['id'] == highlighted_id

                # 获取颜色
                rgb_color = response_colors.get(response, pastel_color_palette[0])

                # 获取OLC值
                olc_column = 'Open Location Code' if 'Open Location Code' in row and not pd.isna(
                    row['Open Location Code']) else 'OLCs'
                olc_value = row[olc_column] if olc_column in row and not pd.isna(row[olc_column]) else "N/A"

                # 创建悬停文本
                hover_text = f"Response: {response}<br>OLC: {olc_value}"

                # 自定义数据
                customdata = [int(row['id'])]

                try:
                    # 解析几何数据
                    geom = wkt.loads(row['geometry'])

                    # 添加几何体到地图
                    if geom.geom_type == 'Polygon':
                        # 获取多边形的坐标
                        coords = list(geom.exterior.coords)
                        lons = [x for x, y in coords]
                        lats = [y for x, y in coords]

                        # 检查坐标有效性
                        valid_coords = [(lon, lat) for lon, lat in zip(lons, lats) if
                                        -180 <= lon <= 180 and -90 <= lat <= 90]
                        if not valid_coords:
                            print(f"No valid coordinates for polygon in row {i}")
                            continue

                        # 提取有效坐标
                        valid_lons = [lon for lon, _ in valid_coords]
                        valid_lats = [lat for _, lat in valid_coords]

                        all_lats.extend(valid_lats)
                        all_lons.extend(valid_lons)

                        # 高亮处理
                        if is_highlighted:
                            highlighted_lats = valid_lats.copy()
                            highlighted_lons = valid_lons.copy()

                            # 添加高亮轮廓
                            fig.add_trace(go.Scattermapbox(
                                mode="lines",
                                lon=valid_lons,
                                lat=valid_lats,
                                line=dict(width=6, color="black"),
                                showlegend=False,
                                hoverinfo="skip"
                            ))

                            # 高亮样式
                            line_color = "yellow"
                            line_width = 3
                            fill_color = get_color_with_alpha(rgb_color, 0.7)
                        else:
                            # 正常样式
                            line_color = get_color_with_alpha(rgb_color, 0.8)
                            line_width = 2
                            fill_color = get_color_with_alpha(rgb_color, 0.4)

                        # 添加到图例的条件
                        show_in_legend = legend_name not in legend_entries
                        if show_in_legend:
                            legend_entries.add(legend_name)

                        # 添加多边形
                        fig.add_trace(go.Scattermapbox(
                            fill="toself",
                            lon=valid_lons,
                            lat=valid_lats,
                            mode="lines",  # 只使用线条，不使用标记点
                            line=dict(color=line_color, width=line_width),
                            fillcolor=fill_color,
                            name=legend_name,
                            text=hover_text,
                            hoverinfo="text",
                            customdata=customdata,
                            showlegend=show_in_legend
                        ))

                    elif geom.geom_type == 'LineString':
                        # 获取线的坐标
                        coords = list(geom.coords)
                        lons = [x for x, y in coords]
                        lats = [y for x, y in coords]

                        # 检查坐标有效性
                        valid_coords = [(lon, lat) for lon, lat in zip(lons, lats) if
                                        -180 <= lon <= 180 and -90 <= lat <= 90]
                        if not valid_coords:
                            print(f"No valid coordinates for linestring in row {i}")
                            continue

                        # 提取有效坐标
                        valid_lons = [lon for lon, _ in valid_coords]
                        valid_lats = [lat for _, lat in valid_coords]

                        all_lats.extend(valid_lats)
                        all_lons.extend(valid_lons)

                        # 高亮处理
                        if is_highlighted:
                            highlighted_lats = valid_lats.copy()
                            highlighted_lons = valid_lons.copy()

                            # 添加高亮轮廓
                            fig.add_trace(go.Scattermapbox(
                                mode="lines",
                                lon=valid_lons,
                                lat=valid_lats,
                                line=dict(width=6, color="black"),
                                showlegend=False,
                                hoverinfo="skip"
                            ))

                            # 高亮样式
                            line_color = "yellow"
                            line_width = 3
                        else:
                            # 正常样式
                            line_color = get_color_with_alpha(rgb_color, 0.8)
                            line_width = 2

                        # 添加到图例的条件
                        show_in_legend = legend_name not in legend_entries
                        if show_in_legend:
                            legend_entries.add(legend_name)

                        # 添加线段
                        fig.add_trace(go.Scattermapbox(
                            mode="lines",  # 只使用线条，不使用标记点
                            lon=valid_lons,
                            lat=valid_lats,
                            line=dict(color=line_color, width=line_width),
                            name=legend_name,
                            text=hover_text,
                            hoverinfo="text",
                            customdata=customdata,
                            showlegend=show_in_legend
                        ))

                    elif geom.geom_type == 'Point':
                        # 获取点的坐标
                        lon, lat = geom.x, geom.y

                        # 检查坐标有效性
                        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                            print(f"Invalid coordinates for point in row {i}: ({lon}, {lat})")
                            continue

                        all_lats.append(lat)
                        all_lons.append(lon)

                        # 高亮处理
                        if is_highlighted:
                            highlighted_lats = [lat]
                            highlighted_lons = [lon]

                            # 添加高亮标记
                            fig.add_trace(go.Scattermapbox(
                                mode="markers",
                                lon=[lon],
                                lat=[lat],
                                marker=dict(size=18, color="black"),
                                showlegend=False,
                                hoverinfo="skip"
                            ))

                            # 高亮样式
                            marker_color = "yellow"
                            marker_size = 12
                        else:
                            # 正常样式
                            marker_color = get_color_with_alpha(rgb_color, 0.8)
                            marker_size = 10

                        # 添加到图例的条件
                        show_in_legend = legend_name not in legend_entries
                        if show_in_legend:
                            legend_entries.add(legend_name)

                        # 添加点
                        fig.add_trace(go.Scattermapbox(
                            mode="markers",
                            lon=[lon],
                            lat=[lat],
                            marker=dict(size=marker_size, color=marker_color),
                            name=legend_name,
                            text=hover_text,
                            hoverinfo="text",
                            customdata=customdata,
                            showlegend=show_in_legend
                        ))

                    elif geom.geom_type == 'MultiPolygon':
                        # 处理多多边形
                        for j, poly in enumerate(geom.geoms):
                            coords = list(poly.exterior.coords)
                            lons = [x for x, y in coords]
                            lats = [y for x, y in coords]

                            # 检查坐标有效性
                            valid_coords = [(lon, lat) for lon, lat in zip(lons, lats) if
                                            -180 <= lon <= 180 and -90 <= lat <= 90]
                            if not valid_coords:
                                print(f"No valid coordinates for multipolygon part in row {i}")
                                continue

                            # 提取有效坐标
                            valid_lons = [lon for lon, _ in valid_coords]
                            valid_lats = [lat for _, lat in valid_coords]

                            all_lats.extend(valid_lats)
                            all_lons.extend(valid_lons)

                            # 高亮处理
                            if is_highlighted:
                                highlighted_lats.extend(valid_lats)
                                highlighted_lons.extend(valid_lons)

                                # 添加高亮轮廓
                                fig.add_trace(go.Scattermapbox(
                                    mode="lines",
                                    lon=valid_lons,
                                    lat=valid_lats,
                                    line=dict(width=6, color="black"),
                                    showlegend=False,
                                    hoverinfo="skip"
                                ))

                                # 高亮样式
                                line_color = "yellow"
                                line_width = 3
                                fill_color = get_color_with_alpha(rgb_color, 0.7)
                            else:
                                # 正常样式
                                line_color = get_color_with_alpha(rgb_color, 0.8)
                                line_width = 2
                                fill_color = get_color_with_alpha(rgb_color, 0.4)

                            # 只对第一个多边形部分添加图例
                            show_in_legend = j == 0 and legend_name not in legend_entries
                            if show_in_legend:
                                legend_entries.add(legend_name)

                            # 添加多边形部分
                            fig.add_trace(go.Scattermapbox(
                                fill="toself",
                                lon=valid_lons,
                                lat=valid_lats,
                                mode="lines",  # 只使用线条，不使用标记点
                                line=dict(color=line_color, width=line_width),
                                fillcolor=fill_color,
                                name=legend_name,
                                text=hover_text,
                                hoverinfo="text",
                                customdata=customdata,
                                showlegend=show_in_legend
                            ))

                except Exception as e:
                    print(f"Error processing geometry in row {i}: {e}")
                    continue

            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue

    # 如果没有有效的几何数据，尝试使用OLC值创建点标记
    if not all_lats and not all_lons:
        print("No valid geometry data, trying to use OLC values")
        # 确定OLC列
        olc_column = None
        if 'Open Location Code' in filtered_df.columns and filtered_df['Open Location Code'].notnull().any():
            olc_column = 'Open Location Code'
        elif 'OLCs' in filtered_df.columns and filtered_df['OLCs'].notnull().any():
            olc_column = 'OLCs'

        if olc_column:
            olc_df = filtered_df[filtered_df[olc_column].notnull()]
            print(f"Found {len(olc_df)} rows with OLC values")

            # 使用示例OLC坐标创建点标记
            # 注意：实际应用中应该使用OLC解码库获取真实坐标
            # 这里我们使用简单的循环偏移来创建不同点
            base_lat = 40.0
            base_lon = -74.0

            for i, (idx, row) in enumerate(olc_df.iterrows()):
                try:
                    response = row['Response']
                    legend_name = f"Response {i + 1}"
                    olc_value = row[olc_column]

                    # 创建偏移坐标
                    lat = base_lat + i * 0.01
                    lon = base_lon + i * 0.01

                    # 保存坐标
                    all_lats.append(lat)
                    all_lons.append(lon)

                    # 是否高亮
                    is_highlighted = highlighted_id is not None and row['id'] == highlighted_id
                    if is_highlighted:
                        highlighted_lats = [lat]
                        highlighted_lons = [lon]

                    # 获取颜色
                    rgb_color = response_colors.get(response, pastel_color_palette[0])

                    # 悬停文本
                    hover_text = f"Response: {response}<br>OLC: {olc_value}"

                    # 自定义数据
                    customdata = [int(row['id'])]

                    # 高亮处理
                    if is_highlighted:
                        # 添加高亮标记
                        fig.add_trace(go.Scattermapbox(
                            mode="markers",
                            lon=[lon],
                            lat=[lat],
                            marker=dict(size=18, color="black"),
                            showlegend=False,
                            hoverinfo="skip"
                        ))

                        # 高亮样式
                        marker_color = "yellow"
                        marker_size = 12
                    else:
                        # 正常样式
                        marker_color = get_color_with_alpha(rgb_color, 0.8)
                        marker_size = 10

                    # 添加到图例的条件
                    show_in_legend = legend_name not in legend_entries
                    if show_in_legend:
                        legend_entries.add(legend_name)

                    # 添加点
                    fig.add_trace(go.Scattermapbox(
                        mode="markers",
                        lon=[lon],
                        lat=[lat],
                        marker=dict(size=marker_size, color=marker_color),
                        name=legend_name,
                        text=hover_text,
                        hoverinfo="text",
                        customdata=customdata,
                        showlegend=show_in_legend
                    ))

                except Exception as e:
                    print(f"Error creating OLC marker for row {i}: {e}")
                    continue

    # 设置地图布局
    if all_lats and all_lons:
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
    else:
        print("No valid coordinates found, using default map view")
        # 默认视图
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=0, lon=0),
                zoom=1
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )

    return fig


# 初始化 Dash
app = Dash(__name__, requests_pathname_prefix='/response/')
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# 主页面布局
main_layout = html.Div([
    html.H1("Different Ideas in the Same Category", style={'textAlign': 'center', 'margin': '20px'}),
    html.P("This dashboard groups together similar ideas within each category",
           style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '30px'}),
    html.Div([
        html.Table(
            id="main_table",
            style={'width': '80%', 'margin': 'auto', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'},
            children=[
                html.Thead(html.Tr([
                    html.Th("Category",
                            style={'backgroundColor': '#f8f9fa', 'padding': '12px', 'border': '1px solid #ddd',
                                   'textAlign': 'center'}),
                    html.Th("Sub-category",
                            style={'backgroundColor': '#f8f9fa', 'padding': '12px', 'border': '1px solid #ddd',
                                   'textAlign': 'center'}),
                    html.Th("Count",
                            style={'backgroundColor': '#f8f9fa', 'padding': '12px', 'border': '1px solid #ddd',
                                   'textAlign': 'center'})
                ])),
                html.Tbody(id="table_body")
            ]
        )
    ])
])


# 详情页面布局
# 详情页面布局
def detail_layout(category, group):
    # 过滤数据
    filtered_df = df[(df["Category"] == category) & (df["Groups"] == group)]

    if filtered_df.empty:
        return html.Div([
            # 返回链接已被移除
            html.H2("No data found for this category and group", style={'textAlign': 'center', 'color': 'red'})
        ])

    # 获取该子分类的摘要
    summary = filtered_df["Summary"].iloc[0] if not filtered_df.empty and "Summary" in filtered_df.columns else ""

    # 检查几何数据可用性
    has_geometry = 'geometry' in filtered_df.columns and filtered_df['geometry'].notnull().any()
    geometry_count = filtered_df['geometry'].notnull().sum() if 'geometry' in filtered_df.columns else 0

    # 检查OLC数据可用性
    olc_column = None
    if 'Open Location Code' in filtered_df.columns and filtered_df['Open Location Code'].notnull().any():
        olc_column = 'Open Location Code'
    elif 'OLCs' in filtered_df.columns and filtered_df['OLCs'].notnull().any():
        olc_column = 'OLCs'

    olc_count = filtered_df[olc_column].notnull().sum() if olc_column else 0

    # 创建表格行
    rows = []

    # 处理表格行
    for i, (idx, row) in enumerate(filtered_df.iterrows()):
        # 使用交互式样式
        interactive_row_style = {'cursor': 'pointer', 'transition': 'all 0.3s'}

        # 确定OLC值
        olc_value = row[olc_column] if olc_column and olc_column in row and not pd.isna(row[olc_column]) else "N/A"

        # 创建行
        rows.append(html.Tr([
            html.Td(row["Response"], style={'border': '1px solid #ddd', 'padding': '10px'}),
            html.Td(olc_value, style={'border': '1px solid #ddd', 'padding': '10px', 'textAlign': 'center'})
        ], id={'type': 'table-row', 'index': row['id']}, style=interactive_row_style))

    # 创建详情页面
    return html.Div([
        # 返回链接已被移除

        # 标题
        html.H2(f"{category} - {group}", style={'textAlign': 'center', 'margin': '20px'}),

        # 摘要
        html.Div([
            html.Strong("Summary: ", style={'fontWeight': 'bold'}),
            html.Span(summary)
        ], style={'textAlign': 'center', 'margin': '20px', 'fontSize': '16px', 'padding': '10px',
                  'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

        # 隐藏存储高亮ID的元素
        dcc.Store(id='highlighted-id', data=None),

        # 表格
        html.Div(
            html.Table(
                style={'width': '80%', 'margin': 'auto', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'},
                children=[
                    html.Thead(html.Tr([
                        html.Th("Response",
                                style={'backgroundColor': '#e9ecef', 'padding': '12px', 'textAlign': 'center'}),
                        html.Th("OLC", style={'backgroundColor': '#e9ecef', 'padding': '12px', 'textAlign': 'center'})
                    ])),
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

        # 数据可用性信息
        html.Div([
            html.P(
                f"Data: {len(filtered_df)} rows total, {geometry_count} with geometry, {olc_count} with location codes",
                style={'textAlign': 'center', 'fontSize': '12px', 'color': '#999', 'margin': '5px 0'}
            )
        ]),

        # 地图
        dcc.Graph(
            id='geometry-map',
            figure=create_map(filtered_df),
            style=map_style,
            clear_on_unhover=True
        )
    ])


# 路由回调
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'), Input('url', 'search')]
)
def display_page(pathname, search):
    print(f"Path requested: {pathname}")  # 调试信息

    if pathname and ('detail' in pathname):
        params = parse_qs(search.lstrip('?'))
        category = unquote(params.get('category', [None])[0])
        group = unquote(params.get('group', [None])[0])

        print(f"Detail view requested for category: {category}, group: {group}")  # 调试信息

        if category and group:
            return detail_layout(category, group)
        else:
            print(f"Missing parameters - category: {category}, group: {group}")  # 调试信息
            return html.Div("Invalid Request")
    return main_layout


# 生成主表内容
@app.callback(
    Output("table_body", "children"),
    [Input('url', 'pathname')]
)
def update_table_body(_):
    rows = []
    current_category = None

    # 调试信息
    print(f"Generating table rows for {len(df_main)} groups")

    for index, row in df_main.iterrows():
        category = row["Category"]
        sub_category = row["Sub-category"]
        count = row["Count"]

        # 构造详细视图的URL，确保路径正确
        detail_url = f"detail?category={quote(category)}&group={quote(sub_category)}"

        if category != current_category:
            current_category = category
            rowspan = int(row["RowSpan"])
            rows.append(html.Tr([
                html.Td(category, rowSpan=rowspan,
                        style={'textAlign': 'center', 'verticalAlign': 'middle', 'padding': '12px',
                               'border': '1px solid #ddd', 'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}),
                html.Td(
                    html.A(
                        sub_category,
                        href=detail_url,
                        style={'textDecoration': 'none', 'fontWeight': '500', 'color': '#0066cc'}
                    ),
                    style={'border': '1px solid #ddd', 'padding': '10px', 'textAlign': 'center'}
                ),
                html.Td(
                    count,
                    style={'border': '1px solid #ddd', 'padding': '10px', 'textAlign': 'center'}
                )
            ]))
        else:
            rows.append(html.Tr([
                html.Td(
                    html.A(
                        sub_category,
                        href=detail_url,
                        style={'textDecoration': 'none', 'fontWeight': '500', 'color': '#0066cc'}
                    ),
                    style={'border': '1px solid #ddd', 'padding': '10px', 'textAlign': 'center'}
                ),
                html.Td(
                    count,
                    style={'border': '1px solid #ddd', 'padding': '10px', 'textAlign': 'center'}
                )
            ]))

    if not rows:
        rows = [html.Tr([html.Td("No data found", colSpan=3, style={'textAlign': 'center', 'padding': '20px'})])]

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
    # 获取分类和子分类
    params = parse_qs(search.lstrip('?'))
    category = unquote(params.get('category', [None])[0])
    group = unquote(params.get('group', [None])[0])

    # 过滤数据
    if category and group:
        filtered_df = df[(df["Category"] == category) & (df["Groups"] == group)]
        # 创建地图并应用高亮效果
        return create_map(filtered_df, highlighted_id)

    # 回退到空图表
    return go.Figure()


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8052)
