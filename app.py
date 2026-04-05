import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="IPL Predictor 2026", page_icon="🏏",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Playfair Display', serif !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1200px !important; }
#MainMenu, footer, header { visibility: hidden; }
.divider { height:2px; background:linear-gradient(90deg,transparent,#D4AF37,transparent); margin:1.2rem 0; }
.team-box { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:16px; margin-bottom:16px; }
.reason-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:10px; padding:12px 16px; margin:6px 0; }
.winner-banner { background:linear-gradient(135deg,rgba(212,175,55,0.15),rgba(212,175,55,0.05)); border:1px solid rgba(212,175,55,0.4); border-radius:20px; padding:2rem; text-align:center; }
.player-btn { font-size:11px !important; padding:4px 8px !important; }
div[data-testid="stButton"] > button { font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ── SQUADS ───────────────────────────────────────────────────────────────────
TEAM_SQUADS = {
    "Mumbai Indians": {
        "short":"MI","color":"#004BA0",
        "players":["Rohit Sharma","Suryakumar Yadav","Hardik Pandya","Tilak Varma",
                   "Ryan Rickelton","Quinton de Kock","Naman Dhir","Sherfane Rutherford",
                   "Jasprit Bumrah","Trent Boult","Deepak Chahar","Mitchell Santner",
                   "Will Jacks","Shardul Thakur","Corbin Bosch","Allah Ghazanfar",
                   "Robin Minz","Raj Angad Bawa","Atharva Ankolekar","Mayank Markande",
                   "Danish Malewar","Raghu Sharma","Ashwani Kumar","Mohammad Izhar"]},
    "Chennai Super Kings": {
        "short":"CSK","color":"#D4AF00",
        "players":["Ruturaj Gaikwad","MS Dhoni","Sanju Samson","Dewald Brevis",
                   "Ayush Mhatre","Kartik Sharma","Sarfaraz Khan","Urvil Patel",
                   "Shivam Dube","Jamie Overton","Ramakrishna Ghosh","Matthew Short",
                   "Prashant Veer","Aman Khan","Zak Foulkes","Shreyas Gopal",
                   "Khaleel Ahmed","Anshul Kamboj","Noor Ahmad","Rahul Chahar",
                   "Gurjapneet Singh","Akeal Hosein","Matt Henry","Spencer Johnson","Mukesh Choudhary"]},
    "Royal Challengers Bengaluru": {
        "short":"RCB","color":"#EC1C24",
        "players":["Virat Kohli","Rajat Patidar","Devdutt Padikkal","Phil Salt",
                   "Jitesh Sharma","Jordan Cox","Krunal Pandya","Venkatesh Iyer",
                   "Tim David","Romario Shepherd","Jacob Bethell","Swapnil Singh",
                   "Vicky Ostwal","Mangesh Yadav","Josh Hazlewood","Rasikh Dar",
                   "Suyash Sharma","Bhuvneshwar Kumar","Nuwan Thushara","Yash Dayal",
                   "Jacob Duffy","Abinandan Singh","Kanishk Chouhan","Satvik Deswal"]},
    "Kolkata Knight Riders": {
        "short":"KKR","color":"#7B5EA7",
        "players":["Ajinkya Rahane","Rinku Singh","Angkrish Raghuvanshi","Manish Pandey",
                   "Finn Allen","Rovman Powell","Tim Seifert","Rahul Tripathi",
                   "Sunil Narine","Cameron Green","Rachin Ravindra","Ramandeep Singh",
                   "Anukul Roy","Sarthak Ranjan","Daksh Kamra","Varun Chakravarthy",
                   "Matheesha Pathirana","Blessing Muzarabani","Navdeep Saini","Vaibhav Arora",
                   "Kartik Tyagi","Prashant Solanki","Saurabh Dubey","Umran Malik","Tejasvi Singh"]},
    "Sunrisers Hyderabad": {
        "short":"SRH","color":"#FF6B00",
        "players":["Travis Head","Ishan Kishan","Heinrich Klaasen","Abhishek Sharma",
                   "Nitish Kumar Reddy","Aniket Verma","Liam Livingstone","Kamindu Mendis",
                   "Harshal Patel","Brydon Carse","David Payne","Pat Cummins",
                   "Jaydev Unadkat","Zeeshan Ansari","Shivam Mavi","Smaran Ravichandran",
                   "Salil Arora","Harsh Dubey","Eshan Malinga","Onkar Tarmale",
                   "Amit Kumar","Praful Hinge","Sakib Hussain","Shivang Kumar"]},
    "Rajasthan Royals": {
        "short":"RR","color":"#EA1A85",
        "players":["Yashasvi Jaiswal","Dhruv Jurel","Shimron Hetmyer","Riyan Parag",
                   "Shubham Dubey","Vaibhav Suryavanshi","Donovan Ferreira","Himmat Singh",
                   "Ravindra Jadeja","Yudhvir Singh Charak","Dasun Shanaka","Jofra Archer",
                   "Ravi Bishnoi","Tushar Deshpande","Sandeep Sharma","Nandre Burger",
                   "Kuldeep Sen","Kwena Maphaka","Adam Milne","Sushant Mishra",
                   "Yash Raj Punja","Vignesh Puthur","Brijesh Sharma","Lhuan-Dre Pretorius","Ravi Singh"]},
    "Gujarat Titans": {
        "short":"GT","color":"#00C2C7",
        "players":["Shubman Gill","Jos Buttler","B Sai Sudharsan","Kumar Kushagra",
                   "Anuj Rawat","Tom Banton","Glenn Phillips","Shahrukh Khan",
                   "Washington Sundar","Rahul Tewatia","Nishant Sindhu","Sai Kishore",
                   "Jayant Yadav","Jason Holder","Mohd Arshad Khan","Kagiso Rabada",
                   "Mohammed Siraj","Rashid Khan","Prasidh Krishna","Ishant Sharma",
                   "Luke Wood","Gurnoor Brar","Manav Suthar","Ashok Sharma","Kulwant Khejroliya"]},
    "Punjab Kings": {
        "short":"PBKS","color":"#ED1B24",
        "players":["Shreyas Iyer","Prabhsimran Singh","Shashank Singh","Nehal Wadhera",
                   "Priyansh Arya","Musheer Khan","Vishnu Vinod","Harnoor Pannu",
                   "Marcus Stoinis","Marco Jansen","Azmatullah Omarzai","Harpreet Brar",
                   "Cooper Connolly","Mitch Owen","Ben Dwarshuis","Suryansh Shedge",
                   "Arshdeep Singh","Yuzvendra Chahal","Vyshak Vijaykumar","Lockie Ferguson",
                   "Xavier Bartlett","Yash Thakur","Pravin Dubey","Vishal Nishad","Pyla Avinash"]},
    "Delhi Capitals": {
        "short":"DC","color":"#0078BC",
        "players":["KL Rahul","Karun Nair","Prithvi Shaw","Abishek Porel",
                   "Tristan Stubbs","Pathum Nissanka","David Miller","Sahil Parakh",
                   "Axar Patel","Nitish Rana","Sameer Rizvi","Ashutosh Sharma",
                   "Vipraj Nigam","Ajay Mandal","Tripurana Vijay","Madhav Tiwari",
                   "Mitchell Starc","Kuldeep Yadav","T Natarajan","Mukesh Kumar",
                   "Kyle Jamieson","Dushmantha Chameera","Lungisani Ngidi","Auqib Nabi"]},
    "Lucknow Super Giants": {
        "short":"LSG","color":"#A72056",
        "players":["Rishabh Pant","Nicholas Pooran","Aiden Markram","Josh Inglis",
                   "Matthew Breetzke","Akshat Raghuwanshi","Abdul Samad","Shahbaz Ahmed",
                   "Wanindu Hasaranga","Arshin Kulkarni","Ayush Badoni","Mitchell Marsh",
                   "Mohammad Shami","Avesh Khan","M Siddharth","Digvesh Singh",
                   "Akash Singh","Prince Yadav","Arjun Tendulkar","Naman Tiwari",
                   "Anrich Nortje","Mayank Yadav","Mohsin Khan","Himmat Singh","Mukul Choudhary"]},
}

PLAYER_ROLES = {
    "Rohit Sharma":"BAT","Suryakumar Yadav":"BAT","Hardik Pandya":"ALL","Tilak Varma":"BAT",
    "Ryan Rickelton":"WK","Quinton de Kock":"WK","Naman Dhir":"ALL","Sherfane Rutherford":"ALL",
    "Jasprit Bumrah":"BOWL","Trent Boult":"BOWL","Deepak Chahar":"BOWL","Mitchell Santner":"ALL",
    "Will Jacks":"ALL","Shardul Thakur":"ALL","Corbin Bosch":"ALL","Allah Ghazanfar":"BOWL",
    "Robin Minz":"WK","Raj Angad Bawa":"ALL","Atharva Ankolekar":"ALL","Mayank Markande":"BOWL",
    "Ruturaj Gaikwad":"BAT","MS Dhoni":"WK","Sanju Samson":"WK","Dewald Brevis":"ALL",
    "Ayush Mhatre":"BAT","Kartik Sharma":"BAT","Sarfaraz Khan":"BAT","Urvil Patel":"WK",
    "Shivam Dube":"ALL","Jamie Overton":"ALL","Ramakrishna Ghosh":"ALL","Matthew Short":"ALL",
    "Prashant Veer":"ALL","Zak Foulkes":"BOWL","Shreyas Gopal":"ALL","Khaleel Ahmed":"BOWL",
    "Anshul Kamboj":"BOWL","Noor Ahmad":"BOWL","Rahul Chahar":"BOWL","Gurjapneet Singh":"BOWL",
    "Akeal Hosein":"BOWL","Matt Henry":"BOWL","Spencer Johnson":"BOWL","Mukesh Choudhary":"BOWL",
    "Virat Kohli":"BAT","Rajat Patidar":"BAT","Devdutt Padikkal":"BAT","Phil Salt":"WK",
    "Jitesh Sharma":"WK","Jordan Cox":"WK","Krunal Pandya":"ALL","Venkatesh Iyer":"ALL",
    "Tim David":"ALL","Romario Shepherd":"ALL","Jacob Bethell":"ALL","Swapnil Singh":"ALL",
    "Vicky Ostwal":"BOWL","Mangesh Yadav":"BOWL","Josh Hazlewood":"BOWL","Rasikh Dar":"BOWL",
    "Suyash Sharma":"BOWL","Bhuvneshwar Kumar":"BOWL","Nuwan Thushara":"BOWL","Yash Dayal":"BOWL",
    "Jacob Duffy":"BOWL","Abinandan Singh":"BOWL","Satvik Deswal":"WK",
    "Ajinkya Rahane":"BAT","Rinku Singh":"BAT","Angkrish Raghuvanshi":"BAT","Manish Pandey":"BAT",
    "Finn Allen":"WK","Rovman Powell":"BAT","Tim Seifert":"WK","Rahul Tripathi":"BAT",
    "Sunil Narine":"ALL","Cameron Green":"ALL","Rachin Ravindra":"ALL","Ramandeep Singh":"ALL",
    "Anukul Roy":"ALL","Varun Chakravarthy":"BOWL","Matheesha Pathirana":"BOWL",
    "Blessing Muzarabani":"BOWL","Navdeep Saini":"BOWL","Vaibhav Arora":"BOWL",
    "Kartik Tyagi":"BOWL","Prashant Solanki":"BOWL","Saurabh Dubey":"BOWL","Umran Malik":"BOWL",
    "Travis Head":"BAT","Ishan Kishan":"WK","Heinrich Klaasen":"WK","Abhishek Sharma":"ALL",
    "Nitish Kumar Reddy":"ALL","Aniket Verma":"BAT","Liam Livingstone":"ALL","Kamindu Mendis":"ALL",
    "Harshal Patel":"ALL","Brydon Carse":"ALL","David Payne":"BOWL","Pat Cummins":"ALL",
    "Jaydev Unadkat":"BOWL","Zeeshan Ansari":"BOWL","Shivam Mavi":"BOWL",
    "Yashasvi Jaiswal":"BAT","Dhruv Jurel":"WK","Shimron Hetmyer":"BAT","Riyan Parag":"ALL",
    "Shubham Dubey":"BAT","Vaibhav Suryavanshi":"BAT","Donovan Ferreira":"BAT",
    "Ravindra Jadeja":"ALL","Yudhvir Singh Charak":"ALL","Dasun Shanaka":"ALL",
    "Jofra Archer":"BOWL","Ravi Bishnoi":"BOWL","Tushar Deshpande":"BOWL",
    "Sandeep Sharma":"BOWL","Nandre Burger":"BOWL","Kuldeep Sen":"BOWL","Kwena Maphaka":"BOWL",
    "Adam Milne":"BOWL","Lhuan-Dre Pretorius":"BAT",
    "Shubman Gill":"BAT","Jos Buttler":"WK","B Sai Sudharsan":"BAT","Kumar Kushagra":"WK",
    "Anuj Rawat":"WK","Tom Banton":"WK","Glenn Phillips":"ALL","Shahrukh Khan":"BAT",
    "Washington Sundar":"ALL","Rahul Tewatia":"ALL","Nishant Sindhu":"ALL","Sai Kishore":"ALL",
    "Jayant Yadav":"ALL","Jason Holder":"ALL","Kagiso Rabada":"BOWL","Mohammed Siraj":"BOWL",
    "Rashid Khan":"ALL","Prasidh Krishna":"BOWL","Ishant Sharma":"BOWL","Luke Wood":"BOWL",
    "Gurnoor Brar":"BOWL","Manav Suthar":"BOWL","Ashok Sharma":"BOWL",
    "Shreyas Iyer":"BAT","Prabhsimran Singh":"WK","Shashank Singh":"BAT","Nehal Wadhera":"BAT",
    "Priyansh Arya":"BAT","Musheer Khan":"ALL","Vishnu Vinod":"WK","Marcus Stoinis":"ALL",
    "Marco Jansen":"ALL","Azmatullah Omarzai":"ALL","Harpreet Brar":"ALL","Cooper Connolly":"ALL",
    "Mitch Owen":"BAT","Ben Dwarshuis":"BOWL","Arshdeep Singh":"BOWL","Yuzvendra Chahal":"BOWL",
    "Vyshak Vijaykumar":"BOWL","Lockie Ferguson":"BOWL","Xavier Bartlett":"BOWL","Yash Thakur":"BOWL",
    "KL Rahul":"WK","Karun Nair":"BAT","Prithvi Shaw":"BAT","Abishek Porel":"WK",
    "Tristan Stubbs":"BAT","Pathum Nissanka":"BAT","David Miller":"BAT","Axar Patel":"ALL",
    "Nitish Rana":"ALL","Sameer Rizvi":"ALL","Ashutosh Sharma":"ALL","Vipraj Nigam":"ALL",
    "Mitchell Starc":"BOWL","Kuldeep Yadav":"BOWL","T Natarajan":"BOWL","Mukesh Kumar":"BOWL",
    "Kyle Jamieson":"ALL","Dushmantha Chameera":"BOWL","Lungisani Ngidi":"BOWL","Auqib Nabi":"BOWL",
    "Rishabh Pant":"WK","Nicholas Pooran":"WK","Aiden Markram":"BAT","Josh Inglis":"WK",
    "Matthew Breetzke":"BAT","Abdul Samad":"ALL","Shahbaz Ahmed":"ALL","Wanindu Hasaranga":"ALL",
    "Arshin Kulkarni":"ALL","Ayush Badoni":"ALL","Mitchell Marsh":"ALL","Mohammad Shami":"BOWL",
    "Avesh Khan":"BOWL","M Siddharth":"BOWL","Anrich Nortje":"BOWL","Mayank Yadav":"BOWL",
    "Mohsin Khan":"BOWL","Arjun Tendulkar":"BOWL","Akshat Raghuwanshi":"BAT",
    "Digvesh Singh":"BOWL","Akash Singh":"BOWL","Prince Yadav":"BOWL","Himmat Singh":"BAT",
}

ROLE_COLORS = {"BAT":"#4ADE80","BOWL":"#F87171","ALL":"#FACC15","WK":"#60A5FA"}
IPL_VENUES = [
    "Wankhede Stadium","MA Chidambaram Stadium","Eden Gardens","M Chinnaswamy Stadium",
    "Arun Jaitley Stadium","Rajiv Gandhi Intl. Stadium","Sawai Mansingh Stadium",
    "Narendra Modi Stadium","Punjab Cricket Association Stadium","BRSABV Ekana Cricket Stadium",
    "HPCA Stadium Dharamshala","Barsapara Cricket Stadium","Shaheed Veer Narayan Singh Stadium",
]
TEAMS = list(TEAM_SQUADS.keys())
FEATURE_LABELS = {
    'team1_enc':'Team 1 strength','team2_enc':'Team 2 strength','venue_enc':'Venue factor',
    't1_won_toss':'Toss result','toss_bat_first':'Bat/field decision',
    'toss_winner_bats':'Toss winner bats','t1_form':'Team 1 form','t2_form':'Team 2 form',
    'form_diff':'Form gap','venue_toss_rate':'Venue toss impact',
    'venue_matches':'Venue experience','h2h_t1_winrate':'Head-to-head','season_norm':'Season recency',
}

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    b = "models"
    try:
        model      = joblib.load(f"{b}/xgb_ipl_model.pkl")
        le_team    = joblib.load(f"{b}/le_team.pkl")
        le_venue   = joblib.load(f"{b}/le_venue.pkl")
        features   = joblib.load(f"{b}/features.pkl")
        explainer  = joblib.load(f"{b}/shap_explainer.pkl")
        feature_df = pd.read_csv(f"{b}/feature_df.csv")
        return model, le_team, le_venue, features, explainer, feature_df, None
    except Exception as e:
        return None,None,None,None,None,None,str(e)

model,le_team,le_venue,FEATURES,explainer,feature_df,load_error = load_artifacts()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_role(p): return PLAYER_ROLES.get(p,"BAT")

def xi_summary(xi):
    roles = [get_role(p) for p in xi]
    return {r:roles.count(r) for r in ["BAT","BOWL","ALL","WK"]}

def ml_predict(team1,team2,venue,t1_form,t2_form,h2h,t1_won,toss_bat):
    try:    t1_enc = le_team.transform([team1])[0]
    except: t1_enc = 0
    try:    t2_enc = le_team.transform([team2])[0]
    except: t2_enc = 0
    try:    v_enc  = le_venue.transform([venue])[0]
    except: v_enc  = 0
    vrows = feature_df[feature_df['venue']==venue]
    vtr   = float(vrows['venue_toss_rate'].mean()) if len(vrows)>0 else 0.5
    vmc   = len(vrows)
    twb   = int((t1_won==1 and toss_bat==1) or (t1_won==0 and toss_bat==0))
    min_s,max_s = feature_df['season'].min(),feature_df['season'].max()
    sn = min((2026-min_s)/(max_s-min_s+1e-9),1.0)
    row = {'team1_enc':t1_enc,'team2_enc':t2_enc,'venue_enc':v_enc,
           't1_won_toss':t1_won,'toss_bat_first':toss_bat,'toss_winner_bats':twb,
           't1_form':t1_form,'t2_form':t2_form,'form_diff':round(t1_form-t2_form,4),
           'venue_toss_rate':round(vtr,4),'venue_matches':vmc,
           'h2h_t1_winrate':h2h,'season_norm':round(sn,4)}
    df_in = pd.DataFrame([row])[FEATURES]
    prob  = model.predict_proba(df_in)[0]
    sv    = explainer.shap_values(df_in)[0]
    return prob[1],prob[0],dict(zip(FEATURES,sv))

def shap_chart(contribs,team1,team2):
    sc = sorted(contribs.items(),key=lambda x:abs(x[1]),reverse=True)[:8]
    labels = [FEATURE_LABELS.get(k,k) for k,_ in sc]
    values = [v for _,v in sc]
    colors = ["#4ADE80" if v>0 else "#F87171" for v in values]
    fig,ax = plt.subplots(figsize=(7,3.5))
    fig.patch.set_facecolor('#0F1C3F'); ax.set_facecolor('#0A1628')
    ax.barh(labels[::-1],values[::-1],color=colors[::-1],height=0.55)
    ax.axvline(0,color='#334155',linewidth=1)
    ax.set_xlabel(f'← {team2}   |   {team1} →',color='#64748B',fontsize=8)
    ax.tick_params(colors='#94A3B8',labelsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.legend(handles=[mpatches.Patch(color='#4ADE80',label=f'Favors {team1}'),
                       mpatches.Patch(color='#F87171',label=f'Favors {team2}')],
              loc='lower right',facecolor='#0F1C3F',edgecolor='#334155',labelcolor='#94A3B8',fontsize=8)
    plt.tight_layout(); return fig

def shap_cards(contribs,team1,team2,c1,c2):
    for feat,val in sorted(contribs.items(),key=lambda x:abs(x[1]),reverse=True)[:5]:
        label  = FEATURE_LABELS.get(feat,feat)
        favors = team1 if val>0 else team2
        color  = c1 if val>0 else c2
        icon   = "🟢" if val>0 else "🔴"
        pct    = min(abs(val)*300,100)
        st.markdown(f"""
        <div class="reason-card">
          <div style="font-size:12px;font-weight:700;color:#E2E8F0;margin-bottom:4px">
            {icon} {label} <span style="color:{color};font-size:10px;margin-left:6px">→ {favors}</span>
          </div>
          <div style="height:4px;background:#1E293B;border-radius:2px">
            <div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:2px;opacity:0.7"></div>
          </div>
        </div>""",unsafe_allow_html=True)

def prob_bar(t1p,t2p,t1,t2,c1,c2):
    p1=int(t1p*100)
    st.markdown(f"""
    <div style="margin:1rem 0">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="color:{c1};font-weight:700;font-size:14px">{t1}</span>
        <span style="color:{c2};font-weight:700;font-size:14px">{t2}</span>
      </div>
      <div style="height:14px;border-radius:7px;overflow:hidden;display:flex;background:#1E293B">
        <div style="width:{p1}%;background:{c1};opacity:0.85"></div>
        <div style="width:{100-p1}%;background:{c2};opacity:0.85"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px">
        <span style="color:{c1};font-size:26px;font-weight:900">{p1}%</span>
        <span style="color:{c2};font-size:26px;font-weight:900">{100-p1}%</span>
      </div>
    </div>""",unsafe_allow_html=True)

# ── SQUAD SELECTOR — Cricbuzz-style vertical, no tabs ────────────────────────
def squad_selector(team_name, xi_key, imp_key):
    """
    Renders a full vertical squad selector for one team.
    xi_key  → session_state key for list of 11 selected players
    imp_key → session_state key for impact player string
    Uses multiselect for XI (no rerun bug) and radio for impact player.
    """
    squad = TEAM_SQUADS[team_name]["players"]
    color = TEAM_SQUADS[team_name]["color"]
    short = TEAM_SQUADS[team_name]["short"]

    xi  = st.session_state[xi_key]
    imp = st.session_state[imp_key]

    xi_count  = len(xi)
    imp_count = 1 if imp else 0
    done = xi_count == 11 and imp_count == 1

    # Header
    badge_bg  = "#1a3a1a" if done else "#1a1a2e"
    badge_bdr = "#4ADE80" if done else color
    st.markdown(
        f'<div style="background:{badge_bg};border:1px solid {badge_bdr};border-radius:10px;'
        f'padding:10px 14px;margin-bottom:10px">'
        f'<span style="color:{color};font-weight:800;font-size:15px">{short}</span>'
        f'<span style="color:#64748B;font-size:12px;margin-left:10px">{team_name}</span>'
        f'<span style="float:right;font-size:12px;color:{"#4ADE80" if done else "#D4AF37"};font-weight:700">'
        f'{"✅ Ready" if done else f"XI: {xi_count}/11  ⚡: {imp_count}/1"}</span></div>',
        unsafe_allow_html=True
    )

    # ── STEP 1: Playing XI via multiselect ──────────────────────────────────
    # multiselect persists perfectly in Streamlit — no rerun issues at all
    st.markdown(f"<div style='font-size:12px;color:#D4AF37;font-weight:700;margin-bottom:4px'>📋 PLAYING XI  <span style='color:#475569'>(select exactly 11)</span></div>", unsafe_allow_html=True)

    selected = st.multiselect(
        label="Playing XI",
        options=squad,
        default=xi,
        max_selections=11,
        key=f"ms_{xi_key}",
        label_visibility="collapsed",
        format_func=lambda p: f"{p}  [{get_role(p)}]"
    )
    # Sync back to session state immediately
    st.session_state[xi_key] = selected

    # ── STEP 2: Impact Player via radio ─────────────────────────────────────
    non_xi = [p for p in squad if p not in selected]
    st.markdown(f"<div style='font-size:12px;color:#F59E0B;font-weight:700;margin:10px 0 4px'>⚡ IMPACT PLAYER  <span style='color:#475569'>(from remaining squad, NOT in XI)</span></div>", unsafe_allow_html=True)

    if non_xi:
        imp_options = ["— None —"] + non_xi
        # Find current index safely
        if imp in non_xi:
            imp_idx = imp_options.index(imp)
        else:
            imp_idx = 0
            st.session_state[imp_key] = ""

        chosen_imp = st.selectbox(
            label="Impact Player",
            options=imp_options,
            index=imp_idx,
            key=f"imp_sel_{xi_key}",
            label_visibility="collapsed",
            format_func=lambda p: f"⚡ {p}  [{get_role(p)}]" if p!="— None —" else "— None —"
        )
        st.session_state[imp_key] = "" if chosen_imp=="— None —" else chosen_imp
    else:
        st.caption("Select your XI first — impact player comes from remaining squad.")

    # ── Summary chips ────────────────────────────────────────────────────────
    cur_xi  = st.session_state[xi_key]
    cur_imp = st.session_state[imp_key]
    if cur_xi:
        sm = xi_summary(cur_xi)
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("🏏",sm["BAT"]); mc2.metric("🎯",sm["BOWL"])
        mc3.metric("⚡",sm["ALL"]); mc4.metric("🧤",sm["WK"])

        all12 = cur_xi + ([cur_imp] if cur_imp else [])
        chips = " ".join([
            (f'<span style="background:rgba(245,158,11,0.2);border:1px solid #F59E0B;'
             f'border-radius:5px;padding:2px 7px;font-size:10px;color:#F59E0B;margin:2px;display:inline-block">⚡{p}</span>'
             if p==cur_imp else
             f'<span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);'
             f'border-radius:5px;padding:2px 7px;font-size:10px;color:#94A3B8;margin:2px;display:inline-block">{p}</span>')
            for p in all12
        ])
        st.markdown(chips, unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k,v in [("xi1",[]),("xi2",[]),("imp1",""),("imp2",""),("pred",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem 0 0.5rem">
  <div style="font-family:'Playfair Display',serif;font-size:38px;font-weight:900;color:#D4AF37;letter-spacing:0.04em">🏏 IPL PREDICTOR 2026</div>
  <div style="color:#475569;font-size:11px;letter-spacing:0.2em;margin-top:4px;font-weight:600">XGBOOST + SHAP &nbsp;·&nbsp; 2026 SQUADS &nbsp;·&nbsp; XI + IMPACT PLAYER</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Model not found in `models/` folder.\n{load_error}"); st.stop()
st.success("✅ Model loaded · SHAP ready", icon="🔬")

# ── MATCH SETUP ───────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Match Setup")
c1,c2,c3 = st.columns([2,1,2])
with c1: team1 = st.selectbox("🔵 Team 1", TEAMS, index=0, key="sel_team1")
with c2: st.markdown("<div style='padding-top:28px;text-align:center;color:#475569;font-weight:800;font-size:20px'>VS</div>",unsafe_allow_html=True)
with c3: team2 = st.selectbox("🔴 Team 2",[t for t in TEAMS if t!=team1],index=3,key="sel_team2")
venue = st.selectbox("📍 Venue", IPL_VENUES, key="sel_venue")

# ── PHASE ─────────────────────────────────────────────────────────────────────
phase = st.radio("Prediction Phase", ["PRE-TOSS","POST-TOSS"],
                 horizontal=True, key="phase_radio",
                 label_visibility="collapsed")
is_post = phase=="POST-TOSS"
bc = "#3B82F6" if not is_post else "#D4AF37"
st.markdown(f'<span style="background:{bc}22;border:1px solid {bc}55;color:{bc};padding:4px 14px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:0.15em">{phase} MODE</span>',unsafe_allow_html=True)

# ── TOSS ──────────────────────────────────────────────────────────────────────
t1_won=0; toss_bat=1
if is_post:
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown("**🎯 Toss Result**")
    tc1,tc2 = st.columns(2)
    with tc1:
        tw = st.radio("Toss Winner",[team1,team2],horizontal=True,key="toss_winner_radio")
        t1_won = int(tw==team1)
    with tc2:
        td = st.radio("Decision",["bat","field"],horizontal=True,key="toss_dec_radio")
        toss_bat = int(td=="bat")

st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

# ── FORM + H2H ────────────────────────────────────────────────────────────────
st.markdown("### 📊 Form & Head-to-Head")
st.caption("Update based on current IPL 2026 standings before each match.")
fc1,fc2,fc3 = st.columns(3)
with fc1: t1_form = st.slider(f"{TEAM_SQUADS[team1]['short']} form (last 5)",0.0,1.0,0.6,0.1,key="sl_t1f")
with fc2: t2_form = st.slider(f"{TEAM_SQUADS[team2]['short']} form (last 5)",0.0,1.0,0.5,0.1,key="sl_t2f")
with fc3: h2h     = st.slider(f"{TEAM_SQUADS[team1]['short']} H2H win rate", 0.0,1.0,0.5,0.05,key="sl_h2h")

st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

# ── PLAYING XII — vertical, both teams side by side ───────────────────────────
st.markdown("### 🏏 Select Playing XII (XI + Impact Player)")
st.caption("Both teams shown together — no switching needed")

col_t1, col_sep, col_t2 = st.columns([10,1,10])
with col_t1:
    squad_selector(team1, "xi1", "imp1")
with col_sep:
    st.markdown("<div style='border-left:1px solid rgba(255,255,255,0.08);height:100%;margin:0 auto;width:1px'></div>",unsafe_allow_html=True)
with col_t2:
    squad_selector(team2, "xi2", "imp2")

st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

# ── PREDICT ───────────────────────────────────────────────────────────────────
xi1_ok  = len(st.session_state.xi1)==11
xi2_ok  = len(st.session_state.xi2)==11
imp1_ok = bool(st.session_state.imp1)
imp2_ok = bool(st.session_state.imp2)
ready   = xi1_ok and xi2_ok and imp1_ok and imp2_ok

if not xi1_ok or not xi2_ok:
    st.warning(f"Select 11 players for both teams — {TEAM_SQUADS[team1]['short']}: {len(st.session_state.xi1)}/11 · {TEAM_SQUADS[team2]['short']}: {len(st.session_state.xi2)}/11")
elif not imp1_ok or not imp2_ok:
    missing = [TEAM_SQUADS[t]['short'] for t,ok in [(team1,imp1_ok),(team2,imp2_ok)] if not ok]
    st.warning(f"⚡ Set Impact Player for: {', '.join(missing)}")

if st.button(f"🔮 {phase} PREDICT", disabled=not ready,
             use_container_width=True, type="primary", key="predict_btn"):
    with st.spinner("Running XGBoost + SHAP..."):
        try:
            t1p,t2p,shap_c = ml_predict(team1,team2,venue,t1_form,t2_form,h2h,t1_won,toss_bat)
            st.session_state.pred = {
                "t1p":t1p,"t2p":t2p,"shap":shap_c,
                "team1":team1,"team2":team2,"phase":phase,
                "toss_info": f"{tw} won · chose to {td}" if is_post else "Pre-toss",
                "xi1":list(st.session_state.xi1),"xi2":list(st.session_state.xi2),
                "imp1":st.session_state.imp1,"imp2":st.session_state.imp2,
            }
        except Exception as e:
            st.error(f"Error: {e}")

# ── RESULTS ───────────────────────────────────────────────────────────────────
if st.session_state.pred:
    p   = st.session_state.pred
    rt1 = p["team1"]; rt2 = p["team2"]
    t1p = p["t1p"];   t2p = p["t2p"]
    winner = rt1 if t1p>t2p else rt2
    conf   = max(t1p,t2p)
    cl     = "HIGH" if conf>0.65 else "MEDIUM" if conf>0.55 else "LOW"
    cc     = "#4ADE80" if cl=="HIGH" else "#FACC15" if cl=="MEDIUM" else "#F87171"
    c1c    = TEAM_SQUADS[rt1]["color"]
    c2c    = TEAM_SQUADS[rt2]["color"]

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown("## 🔮 Prediction")

    st.markdown(f"""
    <div class="winner-banner">
      <div style="font-size:11px;letter-spacing:0.2em;color:#D4AF37;margin-bottom:8px;font-weight:700">{p['phase']} · XGBOOST + SHAP</div>
      <div style="font-family:'Playfair Display',serif;font-size:36px;font-weight:900;color:white;margin-bottom:16px">🏆 {winner}</div>
      <span style="background:{cc}22;border:1px solid {cc}66;color:{cc};padding:5px 18px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:0.1em">
        {cl} CONFIDENCE &nbsp;·&nbsp; {conf*100:.1f}%
      </span>
    </div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    prob_bar(t1p,t2p,rt1,rt2,c1c,c2c)
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

    left,right = st.columns(2)
    with left:
        st.markdown("#### 📊 SHAP Feature Impact")
        st.caption("Which features drove this prediction")
        st.pyplot(shap_chart(p["shap"],rt1,rt2),use_container_width=True)
    with right:
        st.markdown("#### 🧠 Why?")
        st.caption("Top SHAP factors explained")
        shap_cards(p["shap"],rt1,rt2,c1c,c2c)

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    b1,b2 = st.columns(2)

    with b1:
        st.markdown("#### 🏏 XII Summary")
        for tm,xk,ik in [(rt1,"xi1","imp1"),(rt2,"xi2","imp2")]:
            sm  = xi_summary(p[xk])
            col = c1c if tm==rt1 else c2c
            imp = p[ik]; imp_role = get_role(imp) if imp else ""
            st.markdown(f"""
            <div class="reason-card">
              <div style="font-size:12px;font-weight:700;color:{col};margin-bottom:4px">{TEAM_SQUADS[tm]['short']}</div>
              <div style="font-size:11px;color:#64748B">🏏{sm['BAT']} &nbsp;🎯{sm['BOWL']} &nbsp;⚡{sm['ALL']} &nbsp;🧤{sm['WK']}</div>
              {"<div style='font-size:11px;color:#F59E0B;margin-top:4px;font-weight:600'>⚡ "+imp+" ["+imp_role+"]</div>" if imp else ""}
            </div>""",unsafe_allow_html=True)

    with b2:
        st.markdown("#### 📈 Match Context")
        vrows = feature_df[feature_df['venue']==venue]
        vtr   = vrows['venue_toss_rate'].mean() if len(vrows)>0 else 0.5
        st.markdown(f"""
        <div class="reason-card"><div style="font-size:11px;color:#64748B;margin-bottom:2px">🎯 Toss</div>
          <div style="font-size:13px;color:#E2E8F0;font-weight:600">{p['toss_info']}</div></div>
        <div class="reason-card"><div style="font-size:11px;color:#64748B;margin-bottom:2px">📍 {venue.split(",")[0]}</div>
          <div style="font-size:13px;color:#E2E8F0;font-weight:600">{len(vrows)} IPL matches · Toss→win: {vtr*100:.0f}%</div></div>
        <div class="reason-card"><div style="font-size:11px;color:#64748B;margin-bottom:2px">📊 Form</div>
          <div style="font-size:13px;color:#E2E8F0;font-weight:600">{TEAM_SQUADS[rt1]['short']}: {t1_form*100:.0f}% &nbsp;·&nbsp; {TEAM_SQUADS[rt2]['short']}: {t2_form*100:.0f}%</div></div>
        <div class="reason-card"><div style="font-size:11px;color:#64748B;margin-bottom:2px">🤝 H2H ({TEAM_SQUADS[rt1]['short']})</div>
          <div style="font-size:13px;color:#E2E8F0;font-weight:600">{h2h*100:.0f}% win rate · last 10 meetings</div></div>
        """,unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.caption("📌 Accuracy ~60-63% · ROC-AUC 0.69 · Trained IPL 2008-2024 · Use as informed guidance.")
