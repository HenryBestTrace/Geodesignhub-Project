# Geodesignhub-Project

## Project Description
A comprehensive system for analyzing and visualizing geodesign data, user responses, and spatial relationships. This dashboard provides powerful tools for urban planning, participatory design, and geospatial analysis.

### Features
- Interactive Map Visualizations: Display geographic data with highlighting and filtering capabilities
- Response Analysis: Analyze and summarize user feedback data using NLP techniques
- Spatial Clustering: Identify patterns and relationships in geographical data
- Multi-Dashboard System: Four specialized dashboards for different analysis needs
- Flexible Deployment: Local and cloud deployment options

### Dashboard Components
- The system consists of four interconnected dashboards:
- Location Differences Dashboard: Visualize geographic differences across locations
- Response Summary Dashboard: Analyze and categorize user responses
- Conceptual Responses Dashboard: Explore conceptually classified responses
- Different Places Dashboard: Compare similar ideas implemented at different locations

## File Structure
### Dashboard Applications
- enhanced-location-dashboard.py: Location differences dashboard with interactive maps showing geographic features and their differences
- classified_response_summay.py: Response summary dashboard displaying semantic analysis of user responses with clustering
- conceptual_classified_responses.py: Dashboard for exploring responses classified by conceptual categories
- different_place_for_sameidea_new2.py: Dashboard comparing similar ideas implemented at different geographic locations
- main_app_ec2.py: Main application launcher with integrated dashboard navigation interface
- run_dashboard_ec2.py: Configuration and deployment script for AWS EC2 environments

### Core Data Processing Modules
- topology.py: Manages spatial topology relationships, identifies connected features using buffer analysis and spatial similarity metrics
- geography.py: Combines NLP and spatial clustering for geographic theme analysis, implements DBSCAN for spatial grouping
- different_ideology_for_same_place_(1).py: Analyzes different conceptual proposals for the same location, categorizes by Open Location Code
- different_responses_for_same_idea_v2.py: Advanced NLP module for response clustering, uses sentence embeddings and multiple clustering algorithms

### Dashboard System Components
- services/: Directory containing systemd service configuration files
- dashboard.service: Main dashboard service configuration
- dashboard-location.service: Location dashboard service
- dashboard-conceptual.service: Conceptual dashboard service
- dashboard-response.service: Response dashboard service
- dashboard-different.service: Different places dashboard service

### Citation
If you use this dashboard in your research, please cite:
Author Name. (2025). Geographic Data Visualization Dashboard (v1.0.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.15383261
