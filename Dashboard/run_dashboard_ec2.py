#!/usr/bin/env python

"""
EC2 Deployment Script for Unified Dashboard
-------------------------------------------
This script prepares and launches the unified dashboard for deployment on AWS EC2 with HTTPS
"""

import os
import sys
import subprocess
import shutil
import argparse

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'dash',
        'dash-bootstrap-components',
        'pandas',
        'plotly',
        'shapely',
        'pillow',
        'numpy',
        'gunicorn'  # Added for production deployment
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
        print("All dependencies installed successfully!")
    else:
        print("All dependencies are already installed.")

def check_files():
    """Check if all required files exist"""
    required_files = [
        'main_app_ec2.py',
        'enhanced-location-dashboard.py',
        'classified_response_summay.py',
        'conceptual_classified_responses.py',
        'different_place_for_sameidea_new2.py'
    ]
    
    missing_files = [file for file in required_files if not os.path.exists(file)]
    
    if missing_files:
        print(f"ERROR: Missing required files: {', '.join(missing_files)}")
        print("Please make sure all application files are in the current directory.")
        return False
    
    return True

def check_data_files():
    """Check if data files exist, warning only"""
    data_files = [
        'classified_geometries.csv',
        'classified_response_summaries2.csv',
        'conceptual_classified_responses.csv',
        'different_place_for_sameidea2.csv'
    ]
    
    missing_data = [file for file in data_files if not os.path.exists(file)]
    
    if missing_data:
        print(f"WARNING: Missing data files: {', '.join(missing_data)}")
        print("The application may not function correctly without these data files.")
        return False
    
    return True

def create_sample_images():
    """Create sample images to ensure assets directory has necessary images"""
    from PIL import Image, ImageDraw
    
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    # Create four colored sample images
    colors = [
        (67, 97, 238),  # Blue
        (56, 176, 0),   # Green
        (131, 56, 236), # Purple
        (255, 84, 0)    # Orange
    ]
    
    for i, color in enumerate(colors, 1):
        img = Image.new('RGB', (400, 200), color=color)
        draw = ImageDraw.Draw(img)
        
        # Add patterns to make the images more interesting
        for j in range(0, 400, 20):
            draw.line([(j, 0), (j, 200)], fill=(255, 255, 255, 50), width=1)
        for j in range(0, 200, 20):
            draw.line([(0, j), (400, j)], fill=(255, 255, 255, 50), width=1)
            
        # Draw center pattern
        draw.ellipse((150, 50, 250, 150), fill=(255, 255, 255, 100))
        
        img.save(f'assets/{i}.png')
    
    print("Created sample images in assets directory.")

def setup_theme_files():
    """Set up theme-related files"""
    assets_dir = 'assets'
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    
    # Create theme switcher JavaScript file
    js_file = os.path.join(assets_dir, "theme_switcher.js")
    if not os.path.exists(js_file):
        with open(js_file, "w") as f:
            f.write("""
if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.clientside = {
    applyTheme: function(themeData) {
        const theme = themeData.current_theme;
        const body = document.body;
        
        // Apply theme class to body
        if (theme === 'dark') {
            body.classList.add('dark-theme');
            body.classList.remove('light-theme');
        } else {
            body.classList.add('light-theme');
            body.classList.remove('dark-theme');
        }
        
        // Set data attribute for CSS targeting
        document.documentElement.setAttribute('data-theme', theme);
        
        // Update color scheme meta tag for browser UI
        const metaTag = document.querySelector('meta[name="color-scheme"]');
        if (metaTag) {
            metaTag.setAttribute('content', theme);
        } else {
            const newMeta = document.createElement('meta');
            newMeta.name = 'color-scheme';
            newMeta.content = theme;
            document.head.appendChild(newMeta);
        }
        
        // Save theme preference in localStorage
        localStorage.setItem('dashboardTheme', theme);
        
        // Try to update iframes with theme preference
        try {
            const iframe = document.getElementById('dashboard-iframe');
            if (iframe && iframe.contentWindow) {
                iframe.contentWindow.postMessage({ type: 'themeChange', theme: theme }, '*');
            }
        } catch (e) {
            console.log('Could not communicate with iframe:', e);
        }
        
        return theme;
    },
    
    // Function to initialize theme from saved preference
    initTheme: function() {
        const savedTheme = localStorage.getItem('dashboardTheme') || 'light';
        
        // Apply saved theme
        document.documentElement.setAttribute('data-theme', savedTheme);
        
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-theme');
            document.body.classList.remove('light-theme');
        }
        
        return savedTheme;
    }
};

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        window.dash_clientside.clientside.initTheme();
        
        // Initialize switch if already saved as dark
        const savedTheme = localStorage.getItem('dashboardTheme');
        if (savedTheme === 'dark') {
            const switches = document.querySelectorAll('#theme-switch input[type="checkbox"]');
            if (switches.length > 0) {
                switches[0].checked = true;
            }
        }
    }, 100);
});
            """)
        print("Created theme switcher JavaScript file.")

def run_development_mode():
    """Run the dashboard in development mode"""
    print("\nLaunching dashboard in development mode...")
    print("The dashboard will be available at http://localhost:8050")
    print("Press Ctrl+C in this terminal to stop the server when done.")
    
    try:
        # Set EC2_MODE to 0 for development
        os.environ['EC2_MODE'] = '0'
        os.environ['OPEN_BROWSER'] = '1'
        subprocess.run([sys.executable, "main_app_ec2.py"], check=True, env=os.environ)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except subprocess.CalledProcessError:
        print("\nServer stopped due to an error.")

def run_production_mode():
    """Run the dashboard in production mode via Gunicorn"""
    print("\nLaunching dashboard in production mode (for EC2)...")
    print("The dashboard will be available at http://0.0.0.0:8050")
    print("Press Ctrl+C in this terminal to stop the server when done.")
    
    try:
        # Set EC2_MODE to 1 for production
        os.environ['EC2_MODE'] = '1'
        # Run with gunicorn for production
        subprocess.run([
            "gunicorn", 
            "--workers", "3", 
            "--bind", "0.0.0.0:8050", 
            "main_app_ec2:server"
        ], check=True, env=os.environ)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except subprocess.CalledProcessError:
        print("\nServer stopped due to an error.")

def main():
    """Main function to run the dashboard"""
    parser = argparse.ArgumentParser(description="Launch the Dashboard in development or production mode")
    parser.add_argument("--prod", action="store_true", help="Run in production mode (for EC2 deployment)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Unified Dashboard Launcher".center(60))
    print("=" * 60)
    
    # Check dependencies
    print("\nChecking dependencies...")
    check_dependencies()
    
    # Check required files
    print("\nChecking required files...")
    if not check_files():
        sys.exit(1)
    
    # Check data files (warning only)
    print("\nChecking data files...")
    check_data_files()
    
    # Create assets directory and sample images if they don't exist
    if not os.path.exists('assets') or not all(os.path.exists(f'assets/{i}.png') for i in range(1, 5)):
        print("\nCreating sample images...")
        create_sample_images()
    
    # Setup theme files
    print("\nSetting up theme files...")
    setup_theme_files()
    
    # Run in appropriate mode
    if args.prod:
        run_production_mode()
    else:
        run_development_mode()

if __name__ == "__main__":
    main()
