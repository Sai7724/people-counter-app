# 🚀 Deployment Guide for Streamlit Community Cloud

## Prerequisites
- GitHub account
- Streamlit Community Cloud account

## Step-by-Step Deployment

### 1. Create GitHub Repository
1. Go to [GitHub.com](https://github.com)
2. Click "New repository"
3. Name it: `people-counter-app`
4. Make it **Public** (required for free Streamlit Cloud)
5. Don't initialize with README (we already have one)
6. Click "Create repository"

### 2. Connect Local Repository to GitHub
```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/people-counter-app.git

# Push your code to GitHub
git branch -M main
git push -u origin main
```

### 3. Deploy to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `people-counter-app`
5. Set the main file path: `people_counter.py`
6. Click "Deploy!"

### 4. Important Notes
- **Model Files**: The YOLO model files (`*.pt`) are excluded from the repository to keep it small
- **First Run**: The app will download the YOLO model automatically on first use
- **Public Repository**: Required for free Streamlit Cloud deployment

### 5. Troubleshooting
- If deployment fails, check the logs in Streamlit Cloud
- Ensure all dependencies are in `requirements.txt`
- Make sure the main file path is correct

## Repository Structure
```
people-counter-app/
├── people_counter.py      # Main application
├── tracker.py            # Tracking utilities
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Git ignore rules
└── DEPLOYMENT.md        # This file
```

## Files Excluded from Repository
- `*.pt` files (YOLO models - downloaded automatically)
- `*.mp4` files (processed videos)
- `__pycache__/` (Python cache)
- `.history/` (VS Code history)
- `.snapshots/` (IDE snapshots)

Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`
