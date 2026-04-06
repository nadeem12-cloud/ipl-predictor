<div align="center">

# 🏏 IPL Match Predictor 2026

### *Predict smarter. Watch better.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ipl-predictor-79.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=flat)](https://xgboost.readthedocs.io)
[![GitHub](https://img.shields.io/badge/GitHub-nadeem12--cloud-181717?style=flat&logo=github)](https://github.com/nadeem12-cloud)

**A machine learning powered IPL match prediction app — select your Playing XII, pick a mode, get a prediction with reasoning.**

[🚀 Live App](https://ipl-predictor-79.streamlit.app/) &nbsp;·&nbsp; [📓 Training Notebook](IPL_Predictor.ipynb) &nbsp;·&nbsp; [👤 LinkedIn](https://linkedin.com/in/nadeem10)

</div>

---

## 📸 Screenshots

| Match Setup | Playing XII Selector |
|:-----------:|:--------------------:|
| ![Setup](assest/ss1.png) | ![Squad](assest/ss2.png) |

| Prediction Result | SHAP Feature Chart |
|:-----------------:|:-----------------:|
| ![Result](assest/ss3.png) | ![SHAP](assest/ss4.png) |

---

## ✨ Features

| | Feature | Description |
|--|---------|-------------|
| 🔵 | **Pre-Toss Prediction** | Predict before toss using team strength, form & H2H |
| 🟡 | **Post-Toss Prediction** | Sharper prediction after toss — batting/fielding context included |
| 🏏 | **Playing XII Selector** | Select 11 + 1 Impact Player per team (IPL 2026 rule) |
| 📊 | **SHAP Reasoning** | Feature importance chart showing exactly *why* the model predicted what it did |
| 🗓️ | **IPL 2026 Squads** | All 10 teams with official post-auction rosters verified from ESPNcricinfo |
| 🏟️ | **Venue Intelligence** | Historical toss-to-win rates per venue baked into the model |

---

## 🤖 How the Model Works

```
Ball-by-ball IPL data (2008–2024)
            ↓
    Feature Engineering
  ┌─────────────────────────────────┐
  │  Team form (last 5 matches)     │
  │  Head-to-head win rate          │
  │  Venue toss conversion rate     │
  │  Toss result + decision         │
  │  Team historical strength       │
  │  Season recency weight          │
  └─────────────────────────────────┘
            ↓
    XGBoost Classifier
  (time-based split — no leakage)
            ↓
    Win Probability + SHAP values
```

### 📈 Model Performance

| Metric | Score |
|--------|-------|
| ✅ Test Accuracy | **~60–63%** |
| ✅ Test ROC-AUC | **0.69** |
| ⚠️ Train-Val Loss Gap | 0.095 (acceptable) |

> Cricket has inherent randomness — even professional analysts rarely exceed 65% pre-match accuracy. A ROC-AUC of **0.69 means genuine predictive signal** well beyond random guessing (0.50).

### 🗂️ Training Split Strategy

```
2008 ───────────── 2020  │  2021──2022  │  2023+
       TRAIN              │     VAL      │   TEST
  (never touch test!)     │  (tune HP)   │  (final score)
```

> ⚠️ Never used random split — sports data must always be split chronologically to prevent future data leaking into training.

---

## 📁 Project Structure

```
ipl-predictor/
│
├── 📄 app.py                    ← Streamlit app (main file)
├── 📄 requirements.txt          ← Python dependencies
├── 📓 IPL_Predictor.ipynb       ← Google Colab training notebook
│
├── 📂 models/
│   ├── xgb_ipl_model.pkl        ← Trained XGBoost model
│   ├── le_team.pkl              ← Team label encoder
│   ├── le_venue.pkl             ← Venue label encoder
│   ├── features.pkl             ← Feature list
│   └── feature_df.csv           ← Historical venue + feature data
│
├── 📂 assest/
│   ├── ss1.png                  ← Screenshots
│   ├── ss2.png
│   ├── ss3.png
│   └── ss4.png
│
└── 📂 .streamlit/
    └── config.toml              ← Dark theme config
```

---

## ⚡ Local Setup

```bash
# Clone
git clone https://github.com/nadeem12-cloud/ipl-predictor.git
cd ipl-predictor

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 🔁 Retrain the Model

Open `IPL_Predictor.ipynb` in **Google Colab**.

You'll need a **Kaggle API key** to auto-download the dataset. The notebook handles everything end-to-end:

```
1. Mount Google Drive
2. Download IPL dataset via Kaggle API
3. Clean data + standardize team names
4. Engineer features (form, H2H, venue stats)
5. Time-based train/val/test split
6. Train XGBoost with early stopping
7. Evaluate with accuracy + ROC-AUC
8. Save all artifacts to Drive
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| 🤖 ML Model | XGBoost |
| 📊 Explainability | XGBoost built-in `pred_contribs` |
| 🗃️ Training Data | IPL 2008–2024 via Kaggle |
| 🌐 Frontend | Streamlit |
| ☁️ Deployment | Streamlit Cloud |
| 🏋️ Training | Google Colab |

---

## 🏟️ IPL 2026 Teams

All 10 franchises — official post-auction squads verified from ESPNcricinfo / BCCI (Nov 2025):

| Team | Short | Key Players |
|------|-------|-------------|
| Mumbai Indians | MI | Rohit Sharma, Hardik Pandya, Jasprit Bumrah |
| Chennai Super Kings | CSK | Ruturaj Gaikwad, MS Dhoni, Sanju Samson |
| Royal Challengers Bengaluru | RCB | Virat Kohli, Rajat Patidar, Krunal Pandya |
| Kolkata Knight Riders | KKR | Sunil Narine, Cameron Green, Varun Chakravarthy |
| Sunrisers Hyderabad | SRH | Travis Head, Pat Cummins, Heinrich Klaasen |
| Rajasthan Royals | RR | Yashasvi Jaiswal, Jofra Archer, Ravindra Jadeja |
| Gujarat Titans | GT | Shubman Gill, Jos Buttler, Kagiso Rabada |
| Punjab Kings | PBKS | Shreyas Iyer, Arshdeep Singh, Marcus Stoinis |
| Delhi Capitals | DC | KL Rahul, Axar Patel, Mitchell Starc |
| Lucknow Super Giants | LSG | Rishabh Pant, Mohammad Shami, Aiden Markram |

---

## ⚠️ Limitations

- Form and H2H sliders need **manual update** before each match based on current standings
- Model does **not** account for pitch reports, weather, or last-minute injuries
- Accuracy is inherently capped by cricket's randomness — one rogue delivery changes everything

---

## 👤 Author

<div align="center">

**Mohamad Nadeem**


[![GitHub](https://img.shields.io/badge/GitHub-nadeem12--cloud-181717?style=flat&logo=github)](https://github.com/nadeem12-cloud)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-nadeem10-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/nadeem10)

</div>

---

<div align="center">

*Built during IPL 2026 season · Trained on 16 years of IPL data*

**🏏 Use as informed guidance, not certainty**

</div>
