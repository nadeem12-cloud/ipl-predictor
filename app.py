import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cricinfo_data import (
    fetch_points_table, fetch_match_results, fetch_season_stats,
    fetch_player_id, fetch_player_stats, compute_team_form, compute_h2h,
)

st.set_page_config(page_title="IPL Predictor 2026", page_icon="🏏",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Playfair Display', serif !important; }
.block-container { padding: 1.2rem 1.8rem !important; max-width: 1200px !important; }
#MainMenu, footer, header { visibility: hidden; }
.divider { height:2px; background:linear-gradient(90deg,transparent,#D4AF37,transparent); margin:1rem 0; }
.reason-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:10px; padding:10px 14px; margin:5px 0; }
.winner-banner { background:linear-gradient(135deg,rgba(212,175,55,0.15),rgba(212,175,55,0.05)); border:1px solid rgba(212,175,55,0.4); border-radius:20px; padding:1.5rem; text-align:center; }
.wl-chip { display:inline-block; width:28px; height:28px; border-radius:50%; text-align:center; line-height:28px; font-size:11px; font-weight:800; margin:2px; }
.stat-box { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:8px; padding:8px 12px; text-align:center; }
section[data-testid="stSidebar"] { background:#080f20 !important; border-right:1px solid rgba(212,175,55,0.15); }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
TEAMS = [
    "Mumbai Indians","Chennai Super Kings","Royal Challengers Bengaluru",
    "Kolkata Knight Riders","Sunrisers Hyderabad","Rajasthan Royals",
    "Gujarat Titans","Punjab Kings","Delhi Capitals","Lucknow Super Giants"
]
TEAM_META = {
    "Mumbai Indians":              {"short":"MI",   "color":"#004BA0"},
    "Chennai Super Kings":         {"short":"CSK",  "color":"#D4AF00"},
    "Royal Challengers Bengaluru": {"short":"RCB",  "color":"#EC1C24"},
    "Kolkata Knight Riders":       {"short":"KKR",  "color":"#7B5EA7"},
    "Sunrisers Hyderabad":         {"short":"SRH",  "color":"#FF6B00"},
    "Rajasthan Royals":            {"short":"RR",   "color":"#EA1A85"},
    "Gujarat Titans":              {"short":"GT",   "color":"#00C2C7"},
    "Punjab Kings":                {"short":"PBKS", "color":"#ED1B24"},
    "Delhi Capitals":              {"short":"DC",   "color":"#0078BC"},
    "Lucknow Super Giants":        {"short":"LSG",  "color":"#A72056"},
}
TEAM_SQUADS = {
    "Mumbai Indians":["Rohit Sharma","Suryakumar Yadav","Hardik Pandya","Tilak Varma","Ryan Rickelton","Quinton de Kock","Naman Dhir","Sherfane Rutherford","Jasprit Bumrah","Trent Boult","Deepak Chahar","Mitchell Santner","Will Jacks","Shardul Thakur","Corbin Bosch","Allah Ghazanfar","Robin Minz","Raj Angad Bawa","Atharva Ankolekar","Mayank Markande","Danish Malewar","Raghu Sharma","Ashwani Kumar","Mohammad Izhar"],
    "Chennai Super Kings":["Ruturaj Gaikwad","MS Dhoni","Sanju Samson","Dewald Brevis","Ayush Mhatre","Kartik Sharma","Sarfaraz Khan","Urvil Patel","Shivam Dube","Jamie Overton","Ramakrishna Ghosh","Matthew Short","Prashant Veer","Aman Khan","Zak Foulkes","Shreyas Gopal","Khaleel Ahmed","Anshul Kamboj","Noor Ahmad","Rahul Chahar","Gurjapneet Singh","Akeal Hosein","Matt Henry","Spencer Johnson","Mukesh Choudhary"],
    "Royal Challengers Bengaluru":["Virat Kohli","Rajat Patidar","Devdutt Padikkal","Phil Salt","Jitesh Sharma","Jordan Cox","Krunal Pandya","Venkatesh Iyer","Tim David","Romario Shepherd","Jacob Bethell","Swapnil Singh","Vicky Ostwal","Mangesh Yadav","Josh Hazlewood","Rasikh Dar","Suyash Sharma","Bhuvneshwar Kumar","Nuwan Thushara","Yash Dayal","Jacob Duffy","Abinandan Singh","Kanishk Chouhan","Satvik Deswal"],
    "Kolkata Knight Riders":["Ajinkya Rahane","Rinku Singh","Angkrish Raghuvanshi","Manish Pandey","Finn Allen","Rovman Powell","Tim Seifert","Rahul Tripathi","Sunil Narine","Cameron Green","Rachin Ravindra","Ramandeep Singh","Anukul Roy","Sarthak Ranjan","Daksh Kamra","Varun Chakravarthy","Matheesha Pathirana","Blessing Muzarabani","Navdeep Saini","Vaibhav Arora","Kartik Tyagi","Prashant Solanki","Saurabh Dubey","Umran Malik","Tejasvi Singh"],
    "Sunrisers Hyderabad":["Travis Head","Ishan Kishan","Heinrich Klaasen","Abhishek Sharma","Nitish Kumar Reddy","Aniket Verma","Liam Livingstone","Kamindu Mendis","Harshal Patel","Brydon Carse","David Payne","Pat Cummins","Jaydev Unadkat","Zeeshan Ansari","Shivam Mavi","Smaran Ravichandran","Salil Arora","Harsh Dubey","Eshan Malinga","Onkar Tarmale","Amit Kumar","Praful Hinge","Sakib Hussain","Shivang Kumar"],
    "Rajasthan Royals":["Yashasvi Jaiswal","Dhruv Jurel","Shimron Hetmyer","Riyan Parag","Shubham Dubey","Vaibhav Suryavanshi","Donovan Ferreira","Himmat Singh","Ravindra Jadeja","Yudhvir Singh Charak","Dasun Shanaka","Jofra Archer","Ravi Bishnoi","Tushar Deshpande","Sandeep Sharma","Nandre Burger","Kuldeep Sen","Kwena Maphaka","Adam Milne","Sushant Mishra","Yash Raj Punja","Vignesh Puthur","Brijesh Sharma","Lhuan-Dre Pretorius","Ravi Singh"],
    "Gujarat Titans":["Shubman Gill","Jos Buttler","B Sai Sudharsan","Kumar Kushagra","Anuj Rawat","Tom Banton","Glenn Phillips","Shahrukh Khan","Washington Sundar","Rahul Tewatia","Nishant Sindhu","Sai Kishore","Jayant Yadav","Jason Holder","Mohd Arshad Khan","Kagiso Rabada","Mohammed Siraj","Rashid Khan","Prasidh Krishna","Ishant Sharma","Luke Wood","Gurnoor Brar","Manav Suthar","Ashok Sharma","Kulwant Khejroliya"],
    "Punjab Kings":["Shreyas Iyer","Prabhsimran Singh","Shashank Singh","Nehal Wadhera","Priyansh Arya","Musheer Khan","Vishnu Vinod","Harnoor Pannu","Marcus Stoinis","Marco Jansen","Azmatullah Omarzai","Harpreet Brar","Cooper Connolly","Mitch Owen","Ben Dwarshuis","Suryansh Shedge","Arshdeep Singh","Yuzvendra Chahal","Vyshak Vijaykumar","Lockie Ferguson","Xavier Bartlett","Yash Thakur","Pravin Dubey","Vishal Nishad","Pyla Avinash"],
    "Delhi Capitals":["KL Rahul","Karun Nair","Prithvi Shaw","Abishek Porel","Tristan Stubbs","Pathum Nissanka","David Miller","Sahil Parakh","Axar Patel","Nitish Rana","Sameer Rizvi","Ashutosh Sharma","Vipraj Nigam","Ajay Mandal","Tripurana Vijay","Madhav Tiwari","Mitchell Starc","Kuldeep Yadav","T Natarajan","Mukesh Kumar","Kyle Jamieson","Dushmantha Chameera","Lungisani Ngidi","Auqib Nabi"],
    "Lucknow Super Giants":["Rishabh Pant","Nicholas Pooran","Aiden Markram","Josh Inglis","Matthew Breetzke","Akshat Raghuwanshi","Abdul Samad","Shahbaz Ahmed","Wanindu Hasaranga","Arshin Kulkarni","Ayush Badoni","Mitchell Marsh","Mohammad Shami","Avesh Khan","M Siddharth","Digvesh Singh","Akash Singh","Prince Yadav","Arjun Tendulkar","Naman Tiwari","Anrich Nortje","Mayank Yadav","Mohsin Khan","Himmat Singh","Mukul Choudhary"],
}
PLAYER_ROLES = {
    "Rohit Sharma":"BAT","Suryakumar Yadav":"BAT","Hardik Pandya":"ALL","Tilak Varma":"BAT","Ryan Rickelton":"WK","Quinton de Kock":"WK","Naman Dhir":"ALL","Sherfane Rutherford":"ALL","Jasprit Bumrah":"BOWL","Trent Boult":"BOWL","Deepak Chahar":"BOWL","Mitchell Santner":"ALL","Will Jacks":"ALL","Shardul Thakur":"ALL","Corbin Bosch":"ALL","Allah Ghazanfar":"BOWL","Robin Minz":"WK","Raj Angad Bawa":"ALL","Atharva Ankolekar":"ALL","Mayank Markande":"BOWL",
    "Ruturaj Gaikwad":"BAT","MS Dhoni":"WK","Sanju Samson":"WK","Dewald Brevis":"ALL","Ayush Mhatre":"BAT","Kartik Sharma":"BAT","Sarfaraz Khan":"BAT","Urvil Patel":"WK","Shivam Dube":"ALL","Jamie Overton":"ALL","Ramakrishna Ghosh":"ALL","Matthew Short":"ALL","Prashant Veer":"ALL","Zak Foulkes":"BOWL","Shreyas Gopal":"ALL","Khaleel Ahmed":"BOWL","Anshul Kamboj":"BOWL","Noor Ahmad":"BOWL","Rahul Chahar":"BOWL","Gurjapneet Singh":"BOWL","Akeal Hosein":"BOWL","Matt Henry":"BOWL","Spencer Johnson":"BOWL","Mukesh Choudhary":"BOWL",
    "Virat Kohli":"BAT","Rajat Patidar":"BAT","Devdutt Padikkal":"BAT","Phil Salt":"WK","Jitesh Sharma":"WK","Jordan Cox":"WK","Krunal Pandya":"ALL","Venkatesh Iyer":"ALL","Tim David":"ALL","Romario Shepherd":"ALL","Jacob Bethell":"ALL","Swapnil Singh":"ALL","Vicky Ostwal":"BOWL","Mangesh Yadav":"BOWL","Josh Hazlewood":"BOWL","Rasikh Dar":"BOWL","Suyash Sharma":"BOWL","Bhuvneshwar Kumar":"BOWL","Nuwan Thushara":"BOWL","Yash Dayal":"BOWL","Jacob Duffy":"BOWL","Abinandan Singh":"BOWL",
    "Ajinkya Rahane":"BAT","Rinku Singh":"BAT","Angkrish Raghuvanshi":"BAT","Manish Pandey":"BAT","Finn Allen":"WK","Rovman Powell":"BAT","Tim Seifert":"WK","Rahul Tripathi":"BAT","Sunil Narine":"ALL","Cameron Green":"ALL","Rachin Ravindra":"ALL","Ramandeep Singh":"ALL","Anukul Roy":"ALL","Varun Chakravarthy":"BOWL","Matheesha Pathirana":"BOWL","Blessing Muzarabani":"BOWL","Navdeep Saini":"BOWL","Vaibhav Arora":"BOWL","Kartik Tyagi":"BOWL","Prashant Solanki":"BOWL","Umran Malik":"BOWL",
    "Travis Head":"BAT","Ishan Kishan":"WK","Heinrich Klaasen":"WK","Abhishek Sharma":"ALL","Nitish Kumar Reddy":"ALL","Aniket Verma":"BAT","Liam Livingstone":"ALL","Kamindu Mendis":"ALL","Harshal Patel":"ALL","Brydon Carse":"ALL","David Payne":"BOWL","Pat Cummins":"ALL","Jaydev Unadkat":"BOWL","Zeeshan Ansari":"BOWL","Shivam Mavi":"BOWL",
    "Yashasvi Jaiswal":"BAT","Dhruv Jurel":"WK","Shimron Hetmyer":"BAT","Riyan Parag":"ALL","Shubham Dubey":"BAT","Vaibhav Suryavanshi":"BAT","Donovan Ferreira":"BAT","Ravindra Jadeja":"ALL","Yudhvir Singh Charak":"ALL","Dasun Shanaka":"ALL","Jofra Archer":"BOWL","Ravi Bishnoi":"BOWL","Tushar Deshpande":"BOWL","Sandeep Sharma":"BOWL","Nandre Burger":"BOWL","Kuldeep Sen":"BOWL","Kwena Maphaka":"BOWL",
    "Shubman Gill":"BAT","Jos Buttler":"WK","B Sai Sudharsan":"BAT","Kumar Kushagra":"WK","Anuj Rawat":"WK","Tom Banton":"WK","Glenn Phillips":"ALL","Shahrukh Khan":"BAT","Washington Sundar":"ALL","Rahul Tewatia":"ALL","Nishant Sindhu":"ALL","Sai Kishore":"ALL","Jayant Yadav":"ALL","Jason Holder":"ALL","Kagiso Rabada":"BOWL","Mohammed Siraj":"BOWL","Rashid Khan":"ALL","Prasidh Krishna":"BOWL","Ishant Sharma":"BOWL","Luke Wood":"BOWL","Gurnoor Brar":"BOWL",
    "Shreyas Iyer":"BAT","Prabhsimran Singh":"WK","Shashank Singh":"BAT","Nehal Wadhera":"BAT","Priyansh Arya":"BAT","Musheer Khan":"ALL","Vishnu Vinod":"WK","Marcus Stoinis":"ALL","Marco Jansen":"ALL","Azmatullah Omarzai":"ALL","Harpreet Brar":"ALL","Cooper Connolly":"ALL","Mitch Owen":"BAT","Ben Dwarshuis":"BOWL","Arshdeep Singh":"BOWL","Yuzvendra Chahal":"BOWL","Vyshak Vijaykumar":"BOWL","Lockie Ferguson":"BOWL","Xavier Bartlett":"BOWL","Yash Thakur":"BOWL",
    "KL Rahul":"WK","Karun Nair":"BAT","Prithvi Shaw":"BAT","Abishek Porel":"WK","Tristan Stubbs":"BAT","Pathum Nissanka":"BAT","David Miller":"BAT","Axar Patel":"ALL","Nitish Rana":"ALL","Sameer Rizvi":"ALL","Ashutosh Sharma":"ALL","Vipraj Nigam":"ALL","Mitchell Starc":"BOWL","Kuldeep Yadav":"BOWL","T Natarajan":"BOWL","Mukesh Kumar":"BOWL","Kyle Jamieson":"ALL","Dushmantha Chameera":"BOWL","Lungisani Ngidi":"BOWL","Auqib Nabi":"BOWL",
    "Rishabh Pant":"WK","Nicholas Pooran":"WK","Aiden Markram":"BAT","Josh Inglis":"WK","Matthew Breetzke":"BAT","Abdul Samad":"ALL","Shahbaz Ahmed":"ALL","Wanindu Hasaranga":"ALL","Arshin Kulkarni":"ALL","Ayush Badoni":"ALL","Mitchell Marsh":"ALL","Mohammad Shami":"BOWL","Avesh Khan":"BOWL","M Siddharth":"BOWL","Anrich Nortje":"BOWL","Mayank Yadav":"BOWL","Mohsin Khan":"BOWL","Akshat Raghuwanshi":"BAT",
}
ROLE_COLORS = {"BAT":"#4ADE80","BOWL":"#F87171","ALL":"#FACC15","WK":"#60A5FA"}
IPL_VENUES = ["Wankhede Stadium","MA Chidambaram Stadium","Eden Gardens","M Chinnaswamy Stadium","Arun Jaitley Stadium","Rajiv Gandhi Intl. Stadium","Sawai Mansingh Stadium","Narendra Modi Stadium","Punjab Cricket Association Stadium","BRSABV Ekana Cricket Stadium","HPCA Stadium Dharamshala","Barsapara Cricket Stadium","Shaheed Veer Narayan Singh Stadium"]
FEATURE_LABELS = {'team1_enc':'Team 1 strength','team2_enc':'Team 2 strength','venue_enc':'Venue factor','t1_won_toss':'Toss result','toss_bat_first':'Bat/field decision','toss_winner_bats':'Toss winner bats','t1_form':'Team 1 form','t2_form':'Team 2 form','form_diff':'Form gap','venue_toss_rate':'Venue toss impact','venue_matches':'Venue experience','h2h_t1_winrate':'Head-to-head','season_norm':'Season recency'}

# ── MODEL ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    b = "models"
    try:
        model      = joblib.load(f"{b}/xgb_ipl_model.pkl")
        le_team    = joblib.load(f"{b}/le_team.pkl")
        le_venue   = joblib.load(f"{b}/le_venue.pkl")
        features   = joblib.load(f"{b}/features.pkl")
        feature_df = pd.read_csv(f"{b}/feature_df.csv")
        return model, le_team, le_venue, features, feature_df, None
    except Exception as e:
        return None,None,None,None,None,str(e)

model,le_team,le_venue,FEATURES,feature_df,load_error = load_model_artifacts()

# ── HELPERS ────────────────────────────────────────────────────────────────────
def get_role(p): return PLAYER_ROLES.get(p,"BAT")
def xi_summary(xi):
    roles=[get_role(p) for p in xi]
    return {r:roles.count(r) for r in ["BAT","BOWL","ALL","WK"]}

def ml_predict(team1,team2,venue,t1_form,t2_form,h2h,t1_won,toss_bat):
    try:    t1_enc=le_team.transform([team1])[0]
    except: t1_enc=0
    try:    t2_enc=le_team.transform([team2])[0]
    except: t2_enc=0
    try:    v_enc=le_venue.transform([venue])[0]
    except: v_enc=0
    vrows=feature_df[feature_df['venue']==venue]
    vtr=float(vrows['venue_toss_rate'].mean()) if len(vrows)>0 else 0.5
    vmc=len(vrows)
    twb=int((t1_won==1 and toss_bat==1) or (t1_won==0 and toss_bat==0))
    min_s,max_s=feature_df['season'].min(),feature_df['season'].max()
    sn=min((2026-min_s)/(max_s-min_s+1e-9),1.0)
    row={'team1_enc':t1_enc,'team2_enc':t2_enc,'venue_enc':v_enc,'t1_won_toss':t1_won,'toss_bat_first':toss_bat,'toss_winner_bats':twb,'t1_form':t1_form,'t2_form':t2_form,'form_diff':round(t1_form-t2_form,4),'venue_toss_rate':round(vtr,4),'venue_matches':vmc,'h2h_t1_winrate':h2h,'season_norm':round(sn,4)}
    df_in=pd.DataFrame([row])[FEATURES]
    import xgboost as xgb
    dmat=xgb.DMatrix(df_in)
    prob=model.predict_proba(df_in)[0]
    contribs=model.get_booster().predict(dmat,pred_contribs=True)[0]
    return prob[1],prob[0],dict(zip(FEATURES,contribs[:-1]))

def shap_chart(contribs,team1,team2):
    sc=sorted(contribs.items(),key=lambda x:abs(x[1]),reverse=True)[:8]
    labels=[FEATURE_LABELS.get(k,k) for k,_ in sc]; values=[v for _,v in sc]
    colors=["#4ADE80" if v>0 else "#F87171" for v in values]
    fig,ax=plt.subplots(figsize=(7,3.5)); fig.patch.set_facecolor('#0F1C3F'); ax.set_facecolor('#0A1628')
    ax.barh(labels[::-1],values[::-1],color=colors[::-1],height=0.55)
    ax.axvline(0,color='#334155',linewidth=1); ax.set_xlabel(f'← {team2}   |   {team1} →',color='#64748B',fontsize=8)
    ax.tick_params(colors='#94A3B8',labelsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.legend(handles=[mpatches.Patch(color='#4ADE80',label=f'Favors {team1}'),mpatches.Patch(color='#F87171',label=f'Favors {team2}')],loc='lower right',facecolor='#0F1C3F',edgecolor='#334155',labelcolor='#94A3B8',fontsize=8)
    plt.tight_layout(); return fig

def shap_cards(contribs,team1,team2,c1,c2):
    for feat,val in sorted(contribs.items(),key=lambda x:abs(x[1]),reverse=True)[:5]:
        label=FEATURE_LABELS.get(feat,feat); favors=team1 if val>0 else team2
        color=c1 if val>0 else c2; icon="🟢" if val>0 else "🔴"; pct=min(abs(val)*300,100)
        st.markdown(f'<div class="reason-card"><div style="font-size:12px;font-weight:700;color:#E2E8F0;margin-bottom:4px">{icon} {label} <span style="color:{color};font-size:10px;margin-left:6px">→ {favors}</span></div><div style="height:4px;background:#1E293B;border-radius:2px"><div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:2px;opacity:0.7"></div></div></div>',unsafe_allow_html=True)

def prob_bar(t1p,t2p,t1,t2,c1,c2):
    p1=int(t1p*100)
    st.markdown(f'<div style="margin:1rem 0"><div style="display:flex;justify-content:space-between;margin-bottom:6px"><span style="color:{c1};font-weight:700;font-size:14px">{t1}</span><span style="color:{c2};font-weight:700;font-size:14px">{t2}</span></div><div style="height:14px;border-radius:7px;overflow:hidden;display:flex;background:#1E293B"><div style="width:{p1}%;background:{c1};opacity:0.85"></div><div style="width:{100-p1}%;background:{c2};opacity:0.85"></div></div><div style="display:flex;justify-content:space-between;margin-top:6px"><span style="color:{c1};font-size:26px;font-weight:900">{p1}%</span><span style="color:{c2};font-size:26px;font-weight:900">{100-p1}%</span></div></div>',unsafe_allow_html=True)

def wl_chip(r):
    bg="#14532d" if r=="W" else "#450a0a" if r=="L" else "#1c1917"
    color="#4ade80" if r=="W" else "#f87171" if r=="L" else "#a8a29e"
    return f'<span class="wl-chip" style="background:{bg};color:{color}">{r}</span>'

def render_player_card(player, pid_cache):
    role=get_role(player); rc=ROLE_COLORS.get(role,"#94A3B8")
    pid = pid_cache.get(player)
    stats = fetch_player_stats(pid) if pid else {}
    bat=stats.get("batting",{}); bowl=stats.get("bowling",{})
    html=f'<div class="reason-card"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px"><span style="color:#E2E8F0;font-weight:700;font-size:12px">{player}</span><span style="background:{rc}22;color:{rc};border:1px solid {rc}44;border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700">{role}</span></div>'
    if bat:
        html+=f'<div style="font-size:10px;color:#D4AF37;font-weight:700;margin-bottom:4px">🏏 T20 CAREER</div><div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px">'
        for label,val in [("Runs",bat.get("runs","—")),("Avg",bat.get("avg","—")),("SR",bat.get("sr","—")),("50s",bat.get("50s","—"))]:
            html+=f'<div class="stat-box"><div style="font-size:14px;font-weight:800;color:#4ADE80">{val}</div><div style="font-size:9px;color:#64748B">{label}</div></div>'
        html+='</div>'
    if bowl:
        html+=f'<div style="font-size:10px;color:#F87171;font-weight:700;margin-bottom:4px">🎯 BOWLING</div><div style="display:flex;gap:6px;flex-wrap:wrap">'
        for label,val in [("Wkts",bowl.get("wkts","—")),("Eco",bowl.get("eco","—")),("Avg",bowl.get("avg","—")),("Best",bowl.get("best","—"))]:
            html+=f'<div class="stat-box"><div style="font-size:14px;font-weight:800;color:#F87171">{val}</div><div style="font-size:9px;color:#64748B">{label}</div></div>'
        html+='</div>'
    if not bat and not bowl:
        html+=f'<div style="color:#475569;font-size:11px">Stats loading...</div>'
    html+='</div>'
    st.markdown(html,unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
for k,v in [("xi1",[]),("xi2",[]),("imp1",""),("imp2",""),("pred",None)]:
    if k not in st.session_state: st.session_state[k]=v

# ── FETCH LIVE DATA ────────────────────────────────────────────────────────────
with st.spinner("🔄 Fetching live IPL 2026 data from Cricinfo..."):
    pt_rows   = fetch_points_table()    # points table
    results   = fetch_match_results()   # all completed matches
    season_st = fetch_season_stats()    # Orange/Purple cap

# Build a quick lookup from points table
pt_lookup = {r["team"]: r for r in pt_rows}

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Live Points Table + Season Stats
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:Playfair Display,serif;font-size:20px;font-weight:900;color:#D4AF37;margin-bottom:2px">🏏 IPL 2026</div>',unsafe_allow_html=True)
    st.markdown('<div style="color:#475569;font-size:10px;letter-spacing:0.15em;margin-bottom:14px">LIVE FROM CRICINFO · 30 MIN CACHE</div>',unsafe_allow_html=True)

    # Points table
    st.markdown("**📊 Points Table**")
    if pt_rows:
        tbl='<table style="width:100%;border-collapse:collapse;font-size:11px">'
        tbl+='<tr style="color:#475569;border-bottom:1px solid rgba(255,255,255,0.06)"><th align="left" style="padding:3px 2px">#</th><th align="left">Team</th><th>P</th><th>W</th><th>L</th><th>Pts</th><th>NRR</th></tr>'
        for i,r in enumerate(pt_rows,1):
            tm=r["team"]; cl=TEAM_META.get(tm,{}).get("color","#fff"); sh=TEAM_META.get(tm,{}).get("short",tm[:3])
            nrr_str=f'{r["nrr"]:+.2f}' if r["nrr"]!=0 else "—"
            tbl+=f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04)"><td style="padding:5px 2px;color:#475569">{i}</td><td><span style="color:{cl};font-weight:700">{sh}</span></td><td align="center" style="color:#94A3B8">{r["played"]}</td><td align="center" style="color:#4ADE80">{r["won"]}</td><td align="center" style="color:#F87171">{r["lost"]}</td><td align="center" style="color:#D4AF37;font-weight:800">{r["pts"]}</td><td align="center" style="color:#64748B;font-size:10px">{nrr_str}</td></tr>'
        tbl+='</table>'
        st.markdown(tbl,unsafe_allow_html=True)
    else:
        st.caption("Points table unavailable — Cricinfo may be unreachable.")

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

    # Orange + Purple cap
    st.markdown("**🏅 Season Leaders**")
    bat_leaders = season_st.get("batting",[])
    bowl_leaders = season_st.get("bowling",[])
    if bat_leaders:
        st.markdown('<div style="font-size:10px;color:#F97316;font-weight:700;margin-bottom:4px">🧡 Orange Cap</div>',unsafe_allow_html=True)
        for p in bat_leaders[:3]:
            cl=TEAM_META.get(p["team"],{}).get("color","#fff")
            st.markdown(f'<div style="font-size:11px;color:#E2E8F0;margin:2px 0"><span style="color:{cl};font-weight:700">{TEAM_META.get(p["team"],{}).get("short","")}</span> {p["name"]} — <b style="color:#4ADE80">{p["runs"]} runs</b> <span style="color:#64748B">SR:{p["sr"]}</span></div>',unsafe_allow_html=True)
    if bowl_leaders:
        st.markdown('<div style="font-size:10px;color:#A855F7;font-weight:700;margin:8px 0 4px">💜 Purple Cap</div>',unsafe_allow_html=True)
        for p in bowl_leaders[:3]:
            cl=TEAM_META.get(p["team"],{}).get("color","#fff")
            st.markdown(f'<div style="font-size:11px;color:#E2E8F0;margin:2px 0"><span style="color:{cl};font-weight:700">{TEAM_META.get(p["team"],{}).get("short","")}</span> {p["name"]} — <b style="color:#F87171">{p["wkts"]} wkts</b> <span style="color:#64748B">Eco:{p["eco"]}</span></div>',unsafe_allow_html=True)

    if not bat_leaders and not bowl_leaders:
        st.caption("Season stats unavailable.")

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    if st.button("🔄 Refresh live data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="font-family:Playfair Display,serif;font-size:32px;font-weight:900;color:#D4AF37;letter-spacing:0.04em">🏏 IPL PREDICTOR 2026</div>',unsafe_allow_html=True)
st.markdown('<div style="color:#475569;font-size:11px;letter-spacing:0.2em;margin-bottom:8px">XGBOOST + SHAP · LIVE CRICINFO DATA · PLAYING XII</div>',unsafe_allow_html=True)
st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Model not found in `models/`. {load_error}"); st.stop()

# ── MATCH SETUP ────────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Match Setup")
c1,c2,c3=st.columns([2,1,2])
with c1: team1=st.selectbox("🔵 Team 1",TEAMS,index=0,key="sel_t1")
with c2: st.markdown("<div style='padding-top:28px;text-align:center;color:#475569;font-weight:800;font-size:18px'>VS</div>",unsafe_allow_html=True)
with c3: team2=st.selectbox("🔴 Team 2",[t for t in TEAMS if t!=team1],index=3,key="sel_t2")
venue=st.selectbox("📍 Venue",IPL_VENUES,key="sel_venue")

phase=st.radio("Phase",["PRE-TOSS","POST-TOSS"],horizontal=True,key="phase_radio",label_visibility="collapsed")
is_post=phase=="POST-TOSS"
bc="#3B82F6" if not is_post else "#D4AF37"
st.markdown(f'<span style="background:{bc}22;border:1px solid {bc}55;color:{bc};padding:3px 12px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:0.15em">{phase} MODE</span>',unsafe_allow_html=True)

t1_won=0; toss_bat=1
if is_post:
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    tc1,tc2=st.columns(2)
    with tc1:
        tw=st.radio("🎯 Toss Winner",[team1,team2],horizontal=True,key="tw_r"); t1_won=int(tw==team1)
    with tc2:
        td=st.radio("Decision",["bat","field"],horizontal=True,key="td_r"); toss_bat=int(td=="bat")

st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

# ── FORM + H2H — live from Cricinfo ───────────────────────────────────────────
t1_form_val, t1_wl = compute_team_form(results, team1)
t2_form_val, t2_wl = compute_team_form(results, team2)
h2h_meetings, h2h_rate = compute_h2h(results, team1, team2)

col_form1,col_h2h,col_form2=st.columns([5,4,5])
with col_form1:
    c1c=TEAM_META[team1]["color"]
    t1pt=pt_lookup.get(team1,{})
    st.markdown(f'<div style="font-size:12px;font-weight:700;color:{c1c};margin-bottom:6px">{TEAM_META[team1]["short"]} — {t1pt.get("won",0)}W {t1pt.get("lost",0)}L · {t1pt.get("pts",0)} pts</div>',unsafe_allow_html=True)
    chips="".join([wl_chip(r) for r in t1_wl]) or '<span style="color:#475569;font-size:11px">No matches yet</span>'
    st.markdown(f'<div>Last 5: {chips}</div>',unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;color:#64748B;margin-top:4px">Form: <b style="color:{c1c}">{t1_form_val*100:.0f}%</b> · NRR: <b style="color:{c1c}">{t1pt.get("nrr",0):+.2f}</b></div>',unsafe_allow_html=True)

with col_h2h:
    st.markdown('<div style="text-align:center;font-size:11px;font-weight:700;color:#D4AF37;margin-bottom:8px">🤝 HEAD TO HEAD</div>',unsafe_allow_html=True)
    if h2h_meetings:
        t1_wins=sum(1 for w,_,_,_ in h2h_meetings if w==team1)
        t2_wins=len(h2h_meetings)-t1_wins
        st.markdown(f'<div style="display:flex;justify-content:center;align-items:center;gap:16px"><span style="font-size:24px;font-weight:900;color:{TEAM_META[team1]["color"]}">{t1_wins}</span><span style="color:#475569;font-size:11px">last {len(h2h_meetings)}</span><span style="font-size:24px;font-weight:900;color:{TEAM_META[team2]["color"]}">{t2_wins}</span></div>',unsafe_allow_html=True)
        for w,date,margin,mvenue in reversed(h2h_meetings[-3:]):
            wc=TEAM_META[team1]["color"] if w==team1 else TEAM_META[team2]["color"]
            ws=TEAM_META.get(w,{}).get("short",w)
            st.markdown(f'<div style="font-size:10px;color:#475569;text-align:center">{date} · <span style="color:{wc};font-weight:700">{ws}</span> won</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;color:#475569;font-size:11px">No meetings in current data</div>',unsafe_allow_html=True)

with col_form2:
    c2c=TEAM_META[team2]["color"]
    t2pt=pt_lookup.get(team2,{})
    st.markdown(f'<div style="font-size:12px;font-weight:700;color:{c2c};margin-bottom:6px">{TEAM_META[team2]["short"]} — {t2pt.get("won",0)}W {t2pt.get("lost",0)}L · {t2pt.get("pts",0)} pts</div>',unsafe_allow_html=True)
    chips2="".join([wl_chip(r) for r in t2_wl]) or '<span style="color:#475569;font-size:11px">No matches yet</span>'
    st.markdown(f'<div>Last 5: {chips2}</div>',unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;color:#64748B;margin-top:4px">Form: <b style="color:{c2c}">{t2_form_val*100:.0f}%</b> · NRR: <b style="color:{c2c}">{t2pt.get("nrr",0):+.2f}</b></div>',unsafe_allow_html=True)

st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

# ── PLAYING XII ────────────────────────────────────────────────────────────────
st.markdown("### 🏏 Playing XII")
st.caption("Select 11 + Impact Player · Expand stats panel to see live T20 career stats from Cricinfo")

def squad_selector(team_name,xi_key,imp_key,pid_cache):
    squad=TEAM_SQUADS[team_name]; color=TEAM_META[team_name]["color"]; short=TEAM_META[team_name]["short"]
    xi=st.session_state[xi_key]; imp=st.session_state[imp_key]; done=len(xi)==11 and bool(imp)
    badge_bdr="#4ADE80" if done else color
    st.markdown(f'<div style="background:rgba(255,255,255,0.02);border:1px solid {badge_bdr}44;border-radius:10px;padding:10px 14px;margin-bottom:10px"><span style="color:{color};font-weight:800;font-size:14px">{short}</span><span style="color:#64748B;font-size:11px;margin-left:8px">{team_name}</span><span style="float:right;font-size:11px;color:{"#4ADE80" if done else "#D4AF37"};font-weight:700">{"✅ Ready" if done else f"XI:{len(xi)}/11"}</span></div>',unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;color:#D4AF37;font-weight:700;margin-bottom:4px">📋 PLAYING XI</div>',unsafe_allow_html=True)
    selected=st.multiselect("XI",squad,default=xi,max_selections=11,key=f"ms_{xi_key}",label_visibility="collapsed",format_func=lambda p:f"{p}  [{get_role(p)}]")
    st.session_state[xi_key]=selected

    non_xi=[p for p in squad if p not in selected]
    st.markdown('<div style="font-size:11px;color:#F59E0B;font-weight:700;margin:8px 0 4px">⚡ IMPACT PLAYER</div>',unsafe_allow_html=True)
    if non_xi:
        imp_opts=["— None —"]+non_xi
        cur_idx=imp_opts.index(imp) if imp in imp_opts else 0
        chosen=st.selectbox("Impact",imp_opts,index=cur_idx,key=f"imp_{xi_key}",label_visibility="collapsed",format_func=lambda p:f"⚡ {p}  [{get_role(p)}]" if p!="— None —" else "— None —")
        st.session_state[imp_key]="" if chosen=="— None —" else chosen
    else:
        st.caption("Select XI first")

    cur_xi=st.session_state[xi_key]; cur_imp=st.session_state[imp_key]
    if cur_xi:
        sm=xi_summary(cur_xi)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("🏏",sm["BAT"]); m2.metric("🎯",sm["BOWL"]); m3.metric("⚡",sm["ALL"]); m4.metric("🧤",sm["WK"])
        with st.expander(f"📈 Player Stats — {short} (live from Cricinfo)", expanded=False):
            st.caption("T20 career stats · Fetching on demand · May take a moment")
            all12=cur_xi+([cur_imp] if cur_imp else [])
            for p in all12:
                if p not in pid_cache:
                    pid_cache[p]=fetch_player_id(p)
                render_player_card(p, pid_cache)

col_t1,col_sep,col_t2=st.columns([10,1,10])
if "pid_cache" not in st.session_state: st.session_state.pid_cache={}
with col_t1: squad_selector(team1,"xi1","imp1",st.session_state.pid_cache)
with col_sep: st.markdown("<div style='border-left:1px solid rgba(255,255,255,0.06);height:100%;margin:0 auto;width:1px'></div>",unsafe_allow_html=True)
with col_t2: squad_selector(team2,"xi2","imp2",st.session_state.pid_cache)

st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

# ── PREDICT ────────────────────────────────────────────────────────────────────
xi1_ok=len(st.session_state.xi1)==11; xi2_ok=len(st.session_state.xi2)==11
imp1_ok=bool(st.session_state.imp1); imp2_ok=bool(st.session_state.imp2)
ready=xi1_ok and xi2_ok and imp1_ok and imp2_ok

if not xi1_ok or not xi2_ok:
    st.warning(f"Select 11 — {TEAM_META[team1]['short']}: {len(st.session_state.xi1)}/11 · {TEAM_META[team2]['short']}: {len(st.session_state.xi2)}/11")
elif not imp1_ok or not imp2_ok:
    missing=[TEAM_META[t]['short'] for t,ok in [(team1,imp1_ok),(team2,imp2_ok)] if not ok]
    st.warning(f"⚡ Set Impact Player for: {', '.join(missing)}")

if st.button(f"🔮 {phase} PREDICT",disabled=not ready,use_container_width=True,type="primary",key="pred_btn"):
    with st.spinner("Running XGBoost + SHAP..."):
        try:
            t1p,t2p,shap_c=ml_predict(team1,team2,venue,t1_form_val,t2_form_val,h2h_rate,t1_won,toss_bat)
            toss_info=f"{tw} won · chose to {td}" if is_post else "Pre-toss"
            st.session_state.pred={"t1p":t1p,"t2p":t2p,"shap":shap_c,"team1":team1,"team2":team2,"phase":phase,"toss_info":toss_info,"xi1":list(st.session_state.xi1),"xi2":list(st.session_state.xi2),"imp1":st.session_state.imp1,"imp2":st.session_state.imp2}
        except Exception as e:
            st.error(f"Error: {e}")

# ── RESULTS ────────────────────────────────────────────────────────────────────
if st.session_state.pred:
    p=st.session_state.pred; rt1=p["team1"]; rt2=p["team2"]
    t1p=p["t1p"]; t2p=p["t2p"]; winner=rt1 if t1p>t2p else rt2
    conf=max(t1p,t2p); cl="HIGH" if conf>0.65 else "MEDIUM" if conf>0.55 else "LOW"
    cc="#4ADE80" if cl=="HIGH" else "#FACC15" if cl=="MEDIUM" else "#F87171"
    c1c=TEAM_META[rt1]["color"]; c2c=TEAM_META[rt2]["color"]

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown("## 🔮 Prediction")
    st.markdown(f'<div class="winner-banner"><div style="font-size:11px;letter-spacing:0.2em;color:#D4AF37;margin-bottom:8px;font-weight:700">{p["phase"]} · XGBOOST + SHAP</div><div style="font-family:Playfair Display,serif;font-size:34px;font-weight:900;color:white;margin-bottom:16px">🏆 {winner}</div><span style="background:{cc}22;border:1px solid {cc}66;color:{cc};padding:5px 18px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:0.1em">{cl} CONFIDENCE &nbsp;·&nbsp; {conf*100:.1f}%</span></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    prob_bar(t1p,t2p,rt1,rt2,c1c,c2c)
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

    left,right=st.columns(2)
    with left:
        st.markdown("#### 📊 SHAP — Feature Impact")
        st.pyplot(shap_chart(p["shap"],rt1,rt2),use_container_width=True)
    with right:
        st.markdown("#### 🧠 Why?")
        shap_cards(p["shap"],rt1,rt2,c1c,c2c)

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    b1,b2=st.columns(2)
    with b1:
        st.markdown("#### 🏏 XII Summary")
        for tm,xk,ik in [(rt1,"xi1","imp1"),(rt2,"xi2","imp2")]:
            sm=xi_summary(p[xk]); col=c1c if tm==rt1 else c2c
            sh=TEAM_META[tm]["short"]; imp=p[ik]; imp_role=get_role(imp) if imp else ""
            st.markdown(f'<div class="reason-card"><div style="font-size:12px;font-weight:700;color:{col};margin-bottom:4px">{sh}</div><div style="font-size:11px;color:#64748B">🏏{sm["BAT"]} &nbsp;🎯{sm["BOWL"]} &nbsp;⚡{sm["ALL"]} &nbsp;🧤{sm["WK"]}</div>{"<div style=font-size:11px;color:#F59E0B;margin-top:4px;font-weight:600>⚡ "+imp+" ["+imp_role+"]</div>" if imp else ""}</div>',unsafe_allow_html=True)
    with b2:
        st.markdown("#### 📈 Match Context")
        vrows=feature_df[feature_df['venue']==venue]; vtr=vrows['venue_toss_rate'].mean() if len(vrows)>0 else 0.5
        st.markdown(f'<div class="reason-card"><div style="font-size:11px;color:#64748B;margin-bottom:2px">🎯 Toss</div><div style="font-size:13px;color:#E2E8F0;font-weight:600">{p["toss_info"]}</div></div><div class="reason-card"><div style="font-size:11px;color:#64748B;margin-bottom:2px">📍 {venue.split(",")[0]}</div><div style="font-size:13px;color:#E2E8F0;font-weight:600">{len(vrows)} IPL matches · Toss→win: {vtr*100:.0f}%</div></div><div class="reason-card"><div style="font-size:11px;color:#64748B;margin-bottom:2px">📊 Form (live Cricinfo)</div><div style="font-size:13px;color:#E2E8F0;font-weight:600">{TEAM_META[rt1]["short"]}: {t1_form_val*100:.0f}% &nbsp;·&nbsp; {TEAM_META[rt2]["short"]}: {t2_form_val*100:.0f}%</div></div><div class="reason-card"><div style="font-size:11px;color:#64748B;margin-bottom:2px">🤝 H2H ({TEAM_META[rt1]["short"]})</div><div style="font-size:13px;color:#E2E8F0;font-weight:600">{h2h_rate*100:.0f}% · {len(h2h_meetings)} meetings</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.caption("📌 Accuracy ~60-63% · ROC-AUC 0.69 · Form & H2H auto from Cricinfo · Data refreshes every 30 min.")