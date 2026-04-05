# 🏏 IPL Match Predictor 2026

> XGBoost + SHAP + Claude AI · Playing XI Powered · Pre & Post Toss

## Setup

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/ipl-predictor
cd ipl-predictor
pip install -r requirements.txt
```

### 2. Add model files
Download from Google Drive and place in `models/`:
- `xgb_ipl_model.pkl`
- `le_team.pkl`
- `le_venue.pkl`
- `features.pkl`
- `shap_explainer.pkl`
- `feature_df.csv`

### 3. Add API key
Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your-key-here"
```

### 4. Run
```bash
streamlit run app.py
```

## Streamlit Cloud Deploy
1. Push to GitHub (model files included)
2. Go to share.streamlit.io → New app
3. Add `ANTHROPIC_API_KEY` in Secrets settings
4. Deploy ✅

## How it works
- **XGBoost** trained on IPL 2008-2024 ball-by-ball data
- **SHAP** explains which features drove the prediction
- **Claude AI** adds cricket domain reasoning on top
- **Playing XI** input makes predictions match-specific
- **Pre-toss / Post-toss** modes for progressive accuracy
