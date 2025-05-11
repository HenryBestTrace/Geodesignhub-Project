from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import os
import sys
import threading
import importlib.util
import traceback
from PIL import Image, ImageDraw
import random
import time
import webbrowser
import socket

# Global variables
browser_opened = False
EC2_MODE = os.environ.get('EC2_MODE', '0') == '1'  # Environment variable to determine if running on EC2

# Use a light modern theme with BOOTSTRAP + Font Awesome for icons
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    # Add these parameters for EC2/HTTPS deployment
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server

# Define dashboard items with icons
dashboard_items = [
    {
        "id": "geometry-card",
        "title": "Differences Geometries",
        "description": "Analyze geographic location differences with map visualization",
        "icon": "fa-solid fa-map-location-dot",
        "image": "./assets/1.png",
        "path": "/location/",
        "module_path": "enhanced-location-dashboard.py",
        "module_name": "location_differences",
        "port": 8051,
        "color": "#4361ee"  # Custom blue color
    },
    {
        "id": "response-card",
        "title": "Response Summaries",
        "description": "Response data analysis and summaries",
        "icon": "fa-solid fa-chart-column",
        "image": "./assets/2.png",
        "path": "/response/",
        "module_path": "classified_response_summay.py",
        "module_name": "classified_response",
        "port": 8052,
        "color": "#38b000"  # Custom green color
    },
    {
        "id": "conceptual-card",
        "title": "Conceptual Responses",
        "description": "Conceptual classified response exploration",
        "icon": "fa-solid fa-brain",
        "image": "./assets/3.png",
        "path": "/conceptual/",
        "module_path": "conceptual_classified_responses.py",
        "module_name": "conceptual_responses",
        "port": 8053,
        "color": "#8338ec"  # Custom purple color
    },
    {
        "id": "different-card",
        "title": "Different Places",
        "description": "Comparison of same ideas across different locations",
        "icon": "fa-solid fa-location-crosshairs",
        "image": "./assets/4.png",
        "path": "/different/",
        "module_path": "different_place_for_sameidea_new2.py",
        "module_name": "different_place",
        "port": 8054,
        "color": "#ff5400"  # Custom orange color
    }
]

# The rest of the code remains largely unchanged...
# [Keep the existing functions like create_dashboard_cards, create_header, etc.]

# Create enhanced dashboard cards with modern design
def create_dashboard_cards():
    cards = []
    
    for item in dashboard_items:
        # Create a modern card design with subtle effects
        card = dbc.Col(
            html.Div(
                [
                    # Card with image, icon, title and content
                    html.Div(
                        [
                            # Top image with overlay effect
                            html.Div(
                                [
                                    # Background image
                                    html.Div(
                                        style={
                                            "backgroundImage": f"url({item['image']})",
                                            "backgroundSize": "cover",
                                            "backgroundPosition": "center",
                                            "height": "200px",
                                            "borderRadius": "12px 12px 0 0",
                                            "position": "relative"
                                        }
                                    ),
                                    # Icon in circle overlay
                                    html.Div(
                                        html.I(className=item["icon"]),
                                        style={
                                            "position": "absolute",
                                            "bottom": "-25px",
                                            "right": "20px",
                                            "backgroundColor": item["color"],
                                            "color": "white",
                                            "width": "50px",
                                            "height": "50px",
                                            "borderRadius": "50%",
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "center",
                                            "fontSize": "22px",
                                            "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
                                        }
                                    )
                                ],
                                style={"position": "relative"}
                            ),
                            
                            # Card content
                            html.Div(
                                [
                                    # Title
                                    html.H4(
                                        item["title"], 
                                        style={
                                            "color": "#333", 
                                            "fontWeight": "600",
                                            "marginBottom": "12px",
                                            "borderBottom": f"3px solid {item['color']}",
                                            "paddingBottom": "8px",
                                            "display": "inline-block"
                                        }
                                    ),
                                    
                                    # Description
                                    html.P(
                                        item["description"],
                                        style={
                                            "color": "#666",
                                            "fontSize": "16px",
                                            "marginBottom": "25px"
                                        }
                                    ),
                                    
                                    # Button
                                    html.A(
                                        [
                                            "Explore ",
                                            html.I(className="fas fa-arrow-right ms-1")
                                        ],
                                        href=item["path"],
                                        className="dashboard-button",
                                        id=f"btn-{item['id']}",
                                        style={
                                            "backgroundColor": "white",
                                            "color": item["color"],
                                            "border": f"2px solid {item['color']}",
                                            "padding": "8px 20px",
                                            "borderRadius": "30px",
                                            "textDecoration": "none",
                                            "fontWeight": "500",
                                            "display": "inline-block",
                                            "transition": "all 0.3s ease"
                                        }
                                    )
                                ],
                                style={
                                    "padding": "30px 25px 25px",
                                    "backgroundColor": "white"
                                }
                            )
                        ],
                        className="dashboard-card",
                        style={
                            "borderRadius": "12px",
                            "overflow": "hidden",
                            "boxShadow": "0 10px 30px rgba(0,0,0,0.05)",
                            "transition": "all 0.3s ease",
                            "height": "100%"
                        }
                    )
                ],
                style={"padding": "15px"}
            ),
            md=6,
            lg=3,
            className="mb-4"
        )
        cards.append(card)
    
    return dbc.Row(cards)

# Create modern header for homepage
def create_header():
    return html.Div(
        [
            dbc.Container(
                [
                    html.Div(
                        [
                            html.H1(
                                "Geodesignhub Project Dashboard",
                                className="display-4 fw-bold mb-4",
                                style={
                                    "color": "#1e1e1e",
                                    "letterSpacing": "-0.5px",
                                }
                            ),
                            html.P(
                                " ",
                                className="lead fs-4 mb-5",
                                style={"color": "#555", "maxWidth": "800px", "margin": "0 auto"}
                            ),
                            html.Div(
                                html.Div(className="header-line"),
                                style={"width": "100px", "margin": "0 auto 40px"}
                            )
                        ],
                        className="text-center py-5"
                    )
                ],
                fluid=True,
                className="py-4"
            )
        ],
        style={
            "backgroundImage": "linear-gradient(180deg, #f8f9fa, #ffffff)",
            "borderBottom": "1px solid #eaeaea",
            "marginBottom": "30px"
        }
    )

# Homepage layout with clean, modern design
home_layout = html.Div(
    [
        # Modern header
        create_header(),
        
        # Main content container
        dbc.Container(
            [
                # Section title
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Choose a ", style={"color": "#666"}),
                                html.Span("Dashboard", style={"color": "#4361ee"})
                            ],
                            className="h2 mb-4 d-flex justify-content-center gap-2"
                        ),
                        html.P(
                            "Select one of the dashboards below to begin your exploration",
                            className="text-muted text-center mb-5",
                            style={"fontSize": "18px"}
                        )
                    ],
                    className="mb-4 text-center"
                ),
                
                # Dashboard cards
                create_dashboard_cards(),
                
                # Bottom section with more information
                html.Div(
                    [
                        html.Div(className="separator my-5"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.I(
                                            className="fas fa-chart-line me-3",
                                            style={"fontSize": "32px", "color": "#4361ee"}
                                        ),
                                        html.Div(
                                            [
                                                html.H5("Advanced Analytics", className="mb-2"),
                                                html.P(
                                                    "Gain deeper insights with our comprehensive visualization tools",
                                                    className="text-muted mb-0"
                                                )
                                            ]
                                        )
                                    ],
                                    className="d-flex align-items-center mb-4"
                                ),
                                html.Div(
                                    [
                                        html.I(
                                            className="fas fa-map-marked-alt me-3",
                                            style={"fontSize": "32px", "color": "#38b000"}
                                        ),
                                        html.Div(
                                            [
                                                html.H5("Geospatial Mapping", className="mb-2"),
                                                html.P(
                                                    "Explore geographical patterns and spatial relationships in your data",
                                                    className="text-muted mb-0"
                                                )
                                            ]
                                        )
                                    ],
                                    className="d-flex align-items-center"
                                )
                            ],
                            className="col-md-6"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.I(
                                            className="fas fa-comments me-3",
                                            style={"fontSize": "32px", "color": "#8338ec"}
                                        ),
                                        html.Div(
                                            [
                                                html.H5("Response Analysis", className="mb-2"),
                                                html.P(
                                                    "Analyze and categorize responses for better decision making",
                                                    className="text-muted mb-0"
                                                )
                                            ]
                                        )
                                    ],
                                    className="d-flex align-items-center mb-4"
                                ),
                                html.Div(
                                    [
                                        html.I(
                                            className="fas fa-project-diagram me-3",
                                            style={"fontSize": "32px", "color": "#ff5400"}
                                        ),
                                        html.Div(
                                            [
                                                html.H5("Comparative Insights", className="mb-2"),
                                                html.P(
                                                    "Compare ideas across different locations and contexts",
                                                    className="text-muted mb-0"
                                                )
                                            ]
                                        )
                                    ],
                                    className="d-flex align-items-center"
                                )
                            ],
                            className="col-md-6"
                        )
                    ],
                    className="row mt-5 pt-3 feature-section"
                ),
                
                # Footer section
                html.Footer(
                    dbc.Container(
                        [
                            html.Hr(className="my-5"),
                            html.Div(
                                [
                                    html.P(
                                        "Geodata Visualization Dashboard (P10)© 2025",
                                        className="mb-0 text-center text-muted"
                                    ),
                                    html.P(
                                        "Created for advanced data analysis",
                                        className="mb-0 text-center text-muted small"
                                    )
                                ]
                            )
                        ]
                    ),
                    className="mt-5 pb-4"
                )
            ],
            className="py-4"
        )
    ],
    style={"backgroundColor": "#fbfbfd"}
)

# Add a Loading component
loading_layout = dbc.Spinner(
    html.Div(
        "Loading dashboard...",
        id="loading-message",
        style={
            "textAlign": "center",
            "marginTop": "200px",
            "fontSize": "24px",
            "color": "#666"
        }
    ),
    size="lg",
    color="#4361ee",
    type="grow",
    fullscreen=True,
)

# Main application layout
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="loading-container"),
        html.Div(id="debug-info", style={"padding": "10px", "background": "#f8d7da", "display": "none"}),
        html.Div(id="page-content"),
        html.Div(
            dbc.Button(
                [
                    html.I(className="fas fa-home me-2"),
                    "Back to Home"
                ],
                id="home-button",
                href="/",
                color="primary",
                className="mt-3 px-4 py-2",
                style={"display": "none", "borderRadius": "30px", "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"}
            ),
            id="home-button-container",
            className="text-center mb-3"
        ),
        # Store running sub-apps info
        dcc.Store(id="running-subapps", data={}),
        # Status notification component
        dbc.Toast(
            id="status-toast",
            header="Status",
            is_open=False,
            dismissable=True,
            duration=5000,
            style={"position": "fixed", "top": 10, "right": 10, "zIndex": 1000}
        ),
    ]
)

# For status notification callback
@app.callback(
    [Output("status-toast", "is_open"),
     Output("status-toast", "header"),
     Output("status-toast", "children"),
     Output("status-toast", "color")],
    [Input("url", "pathname")],
    [State("running-subapps", "data")]
)
def show_status(pathname, running_subapps_data):
    if pathname == "/" or not pathname:
        return False, "", "", "primary"
    
    # Find matching dashboard
    selected_dashboard = None
    for item in dashboard_items:
        if item["path"] == pathname:
            selected_dashboard = item
            break
    
    if not selected_dashboard:
        return False, "", "", "primary"
    
    # Check if sub-app is loading or loaded
    if selected_dashboard["path"] in running_subapps_data:
        return True, "Dashboard Ready", f"{selected_dashboard['title']} is ready.", "success"
    else:
        return True, "Loading Dashboard", f"Loading {selected_dashboard['title']}...", "primary"

# Loading display callback
@app.callback(
    Output("loading-container", "children"),
    [Input("url", "pathname")],
    [State("running-subapps", "data")]
)
def show_loading(pathname, running_subapps_data):
    if pathname == "/" or not pathname:
        return ""
    
    # Find matching dashboard
    selected_dashboard = None
    for item in dashboard_items:
        if item["path"] == pathname:
            selected_dashboard = item
            break
    
    if not selected_dashboard:
        return ""
    
    # Check if sub-app is loaded
    if selected_dashboard["path"] in running_subapps_data:
        return ""
    else:
        return loading_layout

# Global variable to store sub-app processes
running_subapps = {}

# Modified for EC2 deployment
def load_module(module_path, module_name):
    """Safely load a module and return detailed error information if needed"""
    try:
        # Ensure we use absolute paths
        abs_module_path = os.path.abspath(module_path)
        
        # Temporarily add module directory to sys.path
        module_dir = os.path.dirname(abs_module_path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
            
        # Import the module
        if os.path.exists(abs_module_path):
            spec = importlib.util.spec_from_file_location(module_name, abs_module_path)
            if spec is None:
                print(f"Module not found: {abs_module_path}")
                return None, f"Module not found: {abs_module_path}"
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            try:
                spec.loader.exec_module(module)
                
                # Modify module for EC2 deployment - override host and port settings
                if EC2_MODE and hasattr(module, 'app'):
                    print(f"Configuring {module_name} for EC2 deployment")
                    # We'll be running all apps on the same server in EC2 mode
                    # The sub-app server will not actually be used in EC2 mode
                
                print(f"Successfully loaded module: {module_name}")
                return module, None
            except Exception as e:
                error_msg = f"Error loading module: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                return None, error_msg
        else:
            return None, f"File not found: {abs_module_path}"
    except Exception as e:
        error_msg = f"Error importing module: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

# Check if port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Find available port
def find_available_port(start_port):
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

# Modified run_subapp function for EC2 mode
def run_subapp(module, port, module_name):
    try:
        # Save current working directory
        original_cwd = os.getcwd()
        
        # Change working directory to module location
        module_dir = os.path.dirname(os.path.abspath(sys.modules[module_name].__file__))
        os.chdir(module_dir)
        
        print(f"Starting sub-application {module_name} on port {port}...")
        print(f"Working directory: {os.getcwd()}")
        
        # In EC2 mode, we don't actually run the server for the sub-app
        # We just load the module so we can access its layouts and callbacks
        if not EC2_MODE:
            # Only start the actual server in development mode
            try:
                # Start sub-application with host parameter for network accessibility
                module.app.run_server(debug=False, port=port, use_reloader=False, host='0.0.0.0', 
                               dev_tools_ui=False, dev_tools_props_check=False)
            except Exception as e:
                print(f"Error starting sub-app on port {port}: {str(e)}")
                # Try on a different port
                try:
                    new_port = find_available_port(port + 100)
                    print(f"Retrying on port {new_port}...")
                    module.app.run_server(debug=False, port=new_port, use_reloader=False, host='0.0.0.0', 
                                   dev_tools_ui=False, dev_tools_props_check=False)
                except Exception as retry_e:
                    print(f"Failed to start sub-app even with new port: {str(retry_e)}")
        else:
            # In EC2 mode, we simulate the server is running
            print(f"EC2 MODE: Not starting actual server for {module_name}")
            # Keep thread alive
            while True:
                time.sleep(60)
        
        # Restore original working directory
        os.chdir(original_cwd)
    except Exception as e:
        print(f"Error running sub-application: {str(e)}\n{traceback.format_exc()}")

# Page routing callback - Modified for EC2 deployment
@app.callback(
    [Output("page-content", "children"),
     Output("home-button", "style"),
     Output("debug-info", "children"),
     Output("debug-info", "style"),
     Output("running-subapps", "data")],
    [Input("url", "pathname")],
    [State("running-subapps", "data")]
)
def display_page(pathname, running_subapps_data):
    global running_subapps
    print(f"URL path changed to: {pathname}")
    
    # Homepage
    if pathname == "/" or not pathname:
        return home_layout, {"display": "none"}, "", {"display": "none"}, running_subapps_data
    
    # Show return button
    button_style = {"display": "block", "borderRadius": "30px", "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"}
    
    # Find matching dashboard
    selected_dashboard = None
    for item in dashboard_items:
        if item["path"] == pathname:
            selected_dashboard = item
            break
            
    if not selected_dashboard:
        return home_layout, {"display": "none"}, "Unknown path", {"display": "block", "padding": "10px", "background": "#f8d7da"}, running_subapps_data
    
    # Copy running sub-apps data to update
    updated_subapps_data = dict(running_subapps_data)
    
    # Check if port is already in use
    base_port = selected_dashboard["port"]
    if is_port_in_use(base_port) and selected_dashboard["path"] not in running_subapps:
        # Port is in use, find new port
        port = find_available_port(base_port + 100)
        print(f"Port {base_port} is in use. Using port {port} instead.")
    else:
        port = base_port
    
    # Check if sub-app is already running
    if selected_dashboard["path"] not in running_subapps:
        # Try to load module
        module, error = load_module(selected_dashboard["module_path"], selected_dashboard["module_name"])
        
        if error:
            debug_msg = f"Error loading {selected_dashboard['title']}: {error}"
            print(debug_msg)
            return html.Div([
                html.H4(f"Error loading {selected_dashboard['title']}"),
                html.Pre(error, style={"whiteSpace": "pre-wrap", "overflow": "auto", "maxHeight": "300px"}),
                html.Div([
                    dbc.Button("Try Again", id="retry-button", color="primary", className="me-2", 
                              href=selected_dashboard["path"]),
                    dbc.Button("Return Home", href="/", color="secondary")
                ], className="mt-4 d-flex justify-content-center gap-2")
            ]), button_style, debug_msg, {"display": "block", "padding": "10px", "background": "#f8d7da"}, running_subapps_data
        
        try:
            # Start sub-app in new thread
            thread = threading.Thread(
                target=run_subapp, 
                args=(module, port, selected_dashboard["module_name"]),
                daemon=True
            )
            thread.start()
            
            # Record running sub-app
            running_subapps[selected_dashboard["path"]] = {
                "thread": thread,
                "port": port,
                "title": selected_dashboard["title"],
                "module": module  # Store reference to the module for EC2 mode
            }
            
            # Update stored data
            updated_subapps_data[selected_dashboard["path"]] = {
                "port": port,
                "title": selected_dashboard["title"]
            }
            
            # Wait for server to start in local mode, or simulate success in EC2 mode
            if not EC2_MODE:
                # Give server time to start
                print(f"Waiting for {selected_dashboard['title']} to start...")
                
                # Add timeout check
                max_attempts = 10
                for attempt in range(max_attempts):
                    time.sleep(0.5)
                    # Check if port is active
                    if is_port_in_use(port):
                        print(f"{selected_dashboard['title']} started successfully on port {port}")
                        break
                    print(f"Waiting... ({attempt+1}/{max_attempts})")
                    if attempt == max_attempts - 1:
                        print(f"Warning: Timeout waiting for {selected_dashboard['title']} to start")
            else:
                # In EC2 mode, don't wait for the server as we're not starting it
                print(f"EC2 MODE: Simulating {selected_dashboard['title']} start success")
                time.sleep(0.5)  # Small delay for UX
        except Exception as e:
            error_msg = f"Error starting sub-application: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return html.Div([
                html.H4(f"Error starting {selected_dashboard['title']}"),
                html.Pre(error_msg, style={"whiteSpace": "pre-wrap", "overflow": "auto", "maxHeight": "300px"}),
                html.Div([
                    dbc.Button("Try Again", id="retry-button", color="primary", className="me-2", 
                              href=selected_dashboard["path"]),
                    dbc.Button("Return Home", href="/", color="secondary")
                ], className="mt-4 d-flex justify-content-center gap-2")
            ]), button_style, error_msg, {"display": "block", "padding": "10px", "background": "#f8d7da"}, running_subapps_data
    
    # For EC2 mode, we create content from the module layout directly
    if EC2_MODE:
        try:
            module = running_subapps[selected_dashboard["path"]]["module"]
            
            # Inject the back button into the module's layout
            back_button = html.Div(
                dcc.Link(
                    "🔙 Back to Main",
                    href="/",
                    style={
                        'margin': '10px',
                        'padding': '8px 16px',
                        'backgroundColor': '#007bff',
                        'color': 'white',
                        'borderRadius': '4px',
                        'textDecoration': 'none',
                        'display': 'inline-block'
                    }
                )
            )
            
            # Create a wrapper with the back button and the module's content
            content_layout = html.Div([
                back_button,
                html.Div(
                    # Access the module's layout
                    module.app.layout,
                    style={'margin': '20px 0'}
                )
            ])
            
            return content_layout, {"display": "none"}, "", {"display": "none"}, updated_subapps_data
            
        except Exception as e:
            error_msg = f"Error getting module layout: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return html.Div([
                html.H4(f"Error loading {selected_dashboard['title']} interface"),
                html.Pre(error_msg, style={"whiteSpace": "pre-wrap", "overflow": "auto", "maxHeight": "300px"}),
                dbc.Button("Return Home", href="/", color="secondary", className="mt-3")
            ]), button_style, error_msg, {"display": "block"}, updated_subapps_data
    
    # For development mode, use iframe approach
    port = running_subapps[selected_dashboard["path"]]["port"]
    
    # Create iframe to load sub-app
    try:
        iframe_layout = html.Div([
            html.Div([
                html.I(
                    className=selected_dashboard["icon"], 
                    style={
                        "marginRight": "10px", 
                        "fontSize": "24px",
                        "color": selected_dashboard["color"]
                    }
                ),
                html.H3(
                    selected_dashboard["title"], 
                    className="mb-3",
                    style={"color": "#333"}
                ),
            ], className="d-flex align-items-center justify-content-center"),
            # Error message container
            html.Div(
                id="iframe-error-message",
                style={"display": "none", "textAlign": "center", "margin": "20px 0", "color": "#dc3545"}
            ),
            html.Iframe(
                id="dashboard-iframe",
                src=f"http://127.0.0.1:{port}/",
                style={
                    "width": "100%", 
                    "height": "800px", 
                    "border": "none",
                    "borderRadius": "12px",
                    "boxShadow": "0 6px 18px rgba(0,0,0,0.08)"
                }
            ),
            # Reload button
            html.Div([
                dbc.Button("Reload Dashboard", id="reload-iframe", color="primary", className="mt-3")
            ], id="reload-container", className="text-center", style={"display": "none"})
        ])
        return iframe_layout, button_style, "", {"display": "none"}, updated_subapps_data
    except Exception as e:
        error_msg = f"Error creating interface: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return html.Div([
            html.H4(f"Error loading {selected_dashboard['title']} interface"),
            html.Pre(error_msg, style={"whiteSpace": "pre-wrap", "overflow": "auto", "maxHeight": "300px"}),
            html.Div([
                dbc.Button("Try Again", id="retry-button", color="primary", className="me-2", 
                          href=selected_dashboard["path"]),
                dbc.Button("Return Home", href="/", color="secondary")
            ], className="mt-4 d-flex justify-content-center gap-2")
        ]), button_style, error_msg, {"display": "block", "padding": "10px", "background": "#f8d7da"}, updated_subapps_data

# Client-side callback to detect iframe loading errors
app.clientside_callback(
    """
    function(n) {
        var iframe = document.getElementById('dashboard-iframe');
        var errorMessage = document.getElementById('iframe-error-message');
        var reloadContainer = document.getElementById('reload-container');
        
        if (iframe) {
            iframe.onerror = function() {
                if (errorMessage) {
                    errorMessage.textContent = "Failed to load dashboard. The server may not be responding.";
                    errorMessage.style.display = "block";
                }
                if (reloadContainer) {
                    reloadContainer.style.display = "block";
                }
            };
            
            iframe.onload = function() {
                try {
                    // Try to access iframe content to check if it loaded properly
                    var iframeContent = iframe.contentWindow.document;
                    // If we can access content, hide error message
                    if (errorMessage) {
                        errorMessage.style.display = "none";
                    }
                    if (reloadContainer) {
                        reloadContainer.style.display = "none";
                    }
                } catch (e) {
                    // Security error or cross-origin issue
                    if (errorMessage) {
                        errorMessage.textContent = "Failed to load dashboard. The server may not be responding.";
                        errorMessage.style.display = "block";
                    }
                    if (reloadContainer) {
                        reloadContainer.style.display = "block";
                    }
                }
            };
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output("debug-info", "data"),
    Input("url", "pathname")
)

# Reload iframe callback
@app.callback(
    Output("dashboard-iframe", "src"),
    Input("reload-iframe", "n_clicks"),
    State("url", "pathname"),
    State("running-subapps", "data"),
    prevent_initial_call=True
)
def reload_iframe(n_clicks, pathname, running_subapps_data):
    if not n_clicks:
        return dash.no_update
    
    # Find matching dashboard and port
    for item in dashboard_items:
        if item["path"] == pathname and pathname in running_subapps_data:
            port = running_subapps_data[pathname]["port"]
            # Add timestamp to avoid caching
            timestamp = int(time.time())
            return f"http://127.0.0.1:{port}/?t={timestamp}"
    
    # If no matching dashboard is found, return current src
    return dash.no_update

if __name__ == "__main__":
    # Create assets directory
    if not os.path.exists("assets"):
        os.makedirs("assets")
    
    # Create custom CSS file with enhanced styles
    css_file = os.path.join("assets", "custom.css")
    with open(css_file, "w") as f:
        f.write("""
        /* Enhanced Light Theme Dashboard Styles */
        
        body { 
            font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif; 
            background-color: #fbfbfd;
            color: #333;
        }
        
        /* Card hover effects */
        .dashboard-card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            background-color: white;
            border: none;
        }
        
        .dashboard-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Button hover effect */
        .dashboard-button:hover {
            color: white !important;
            background-color: var(--hover-color) !important;
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Button color variations */
        .dashboard-card:nth-child(1) .dashboard-button { --hover-color: #4361ee; }
        .dashboard-card:nth-child(2) .dashboard-button { --hover-color: #38b000; }
        .dashboard-card:nth-child(3) .dashboard-button { --hover-color: #8338ec; }
        .dashboard-card:nth-child(4) .dashboard-button { --hover-color: #ff5400; }
        
        /* Header line */
        .header-line {
            height: 4px;
            background: linear-gradient(90deg, #4361ee, #38b000, #8338ec, #ff5400);
            border-radius: 2px;
        }
        
        /* Section separator */
        .separator {
            height: 1px;
            background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
            position: relative;
        }
        
        .separator::before {
            content: '';
            position: absolute;
            width: 100px;
            height: 3px;
            background: linear-gradient(90deg, #4361ee, #38b000);
            top: -1px;
            left: 50%;
            transform: translateX(-50%);
            border-radius: 3px;
        }
        
        /* Feature section styling */
        .feature-section {
            background-color: #f8faff;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 3px 15px rgba(0, 0, 0, 0.03);
        }
        
        /* Animation for page transitions */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        #page-content {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Modern scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #c0c0c0;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #a0a0a0;
        }
        
        /* Home button styling */
        #home-button {
            transition: all 0.3s ease;
        }
        
        #home-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15) !important;
        }
        """)
    
    # Auto-open browser based on environment variable
    def open_browser():
        global browser_opened
        if not browser_opened:
            browser_opened = True
            time.sleep(1)
            webbrowser.open("http://127.0.0.1:8050")
    
    # Only open browser when run directly and not in EC2 mode
    if os.environ.get('OPEN_BROWSER', '0') == '1' and not EC2_MODE:
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Start server with host parameter to allow external connections
    print("Starting dashboard application...")
    if EC2_MODE:
        print("Running in EC2 mode (production)")
        # In EC2 mode, bind to all interfaces but don't show debug info
        app.run(debug=False, host='0.0.0.0', port=8050)
    else:
        print("Running in development mode")
        app.run(debug=False, host='0.0.0.0', port=8050)
