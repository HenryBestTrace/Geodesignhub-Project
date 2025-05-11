# Geodesignhub-Project

A comprehensive system for analyzing and visualizing geodesign data, user responses, and spatial relationships. This dashboard provides powerful tools for urban planning, participatory design, and geospatial analysis.
Features

Interactive Map Visualizations: Display geographic data with highlighting and filtering capabilities
Response Analysis: Analyze and summarize user feedback data using NLP techniques
Spatial Clustering: Identify patterns and relationships in geographical data
Multi-Dashboard System: Four specialized dashboards for different analysis needs
Flexible Deployment: Local and cloud deployment options

Dashboard Components
The system consists of four interconnected dashboards:

Location Differences Dashboard: Visualize geographic differences across locations
Response Summary Dashboard: Analyze and categorize user responses
Conceptual Responses Dashboard: Explore conceptually classified responses
Different Places Dashboard: Compare similar ideas implemented at different locations

File Structure
Dashboard Applications

enhanced-location-dashboard.py: Location differences dashboard with interactive maps showing geographic features and their differences
classified_response_summay.py: Response summary dashboard displaying semantic analysis of user responses with clustering
conceptual_classified_responses.py: Dashboard for exploring responses classified by conceptual categories
different_place_for_sameidea_new2.py: Dashboard comparing similar ideas implemented at different geographic locations

Core Data Processing Modules

topology.py: Manages spatial topology relationships, identifies connected features using buffer analysis and spatial similarity metrics
geography.py: Combines NLP and spatial clustering for geographic theme analysis, implements DBSCAN for spatial grouping
different_ideology_for_same_place_(1).py: Analyzes different conceptual proposals for the same location, categorizes by Open Location Code
different_responses_for_same_idea_v2.py: Advanced NLP module for response clustering, uses sentence embeddings and multiple clustering algorithms

System Components

main_app_ec2.py: Main application launcher with integrated dashboard navigation interface
run_dashboard_ec2.py: Configuration and deployment script for AWS EC2 environments
services/: Directory containing systemd service configuration files

dashboard.service: Main dashboard service configuration
dashboard-location.service: Location dashboard service
dashboard-conceptual.service: Conceptual dashboard service
dashboard-response.service: Response dashboard service
dashboard-different.service: Different places dashboard service



Installation
Local Development Environment
bash# Clone the repository
git clone https://github.com/yourusername/geodata-visualization-dashboard.git
cd geodata-visualization-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main application
python main_app_ec2.py
Production Deployment
bash# Run with production settings
python run_dashboard_ec2.py --prod
Usage

Launch the main application which provides access to all dashboards
Select the specific dashboard you need for your analysis
Upload or select available geodata for visualization
Interact with the maps and visualizations to explore patterns
Export results or summaries as needed

Requirements
dash
dash-bootstrap-components
pandas
plotly
geopandas
shapely
pillow
numpy
scikit-learn
nltk
sentence-transformers
gunicorn
Data Formats
The system supports various geographic data formats:

GeoJSON
GPKG (GeoPackage)
WKT (Well-Known Text)
CSV with spatial coordinates

Citation
If you use this dashboard in your research, please cite:
Author Name. (2025). Geographic Data Visualization Dashboard (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.15383261
