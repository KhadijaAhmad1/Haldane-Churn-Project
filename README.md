# Telecom Customer Churn Prediction & AI Retention System


## Project Overview

This project addresses a telecommunications company facing high customer churn. It delivers two things:

1. **A machine learning pipeline** that predicts which subscribers are likely to churn, trained on 3,150 real customer records, achieving a ROC-AUC of 0.984.

2. **An AI Retention Hub** — a browser-based tool that uses Generative AI (Claude by Anthropic) to automate retention workflows: risk scanning, personalised email drafting, offer design, analyst chat, and strategic action planning.

The technical approach is directly transferable to Haldane Group's real business challenge of retaining high-value trade customers, demonstrating how this same AI architecture would work embedded in their existing K8 ERP and Phocas BI systems.

---

## Repository Structure

```
haldane-churn-project/
│
├── analysis/
│   ├── churn_analysis.py       # Full ML pipeline: EDA, model training, evaluation
│   └── visualizations.py       # All 5 standalone presentation-ready charts
│
├── frontend/
│   └── index.html              # AI Retention Hub — runs in any browser, no install
│
├── backend/
│   └── app.py                  # Optional Flask proxy (keeps API key server-side)
│
├── .env.example                # Template — copy to .env and add your API key
├── .gitignore                  # Ensures .env and secrets are never committed
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Minimum Task Requirements — Completion Status

| Requirement | Status | Where |
|---|---|---|
| 3+ visualisations identifying churn factors | ✅ 5 charts | `analysis/visualizations.py` |
| Train and test an ML model to predict churn | ✅ Random Forest, ROC-AUC 0.984 | `analysis/churn_analysis.py` |
| Explain and justify the technical solution | ✅ Documented throughout | Code comments + this README |
| Where and how Generative AI was used | ✅ 3 documented uses | See GenAI section below |
| Key findings from the analysis | ✅ In visualisations + hub | See Findings section below |
| Areas where Haldane could use AI for retention | ✅ In hub + README | See AI Strategy section below |

---

## Machine Learning Pipeline (`analysis/churn_analysis.py`)

### Dataset
- 3,150 telecom subscribers, 13 features, 15.7% churn rate (495 churned)
- Features: call failures, complaints, subscription length, charge amount, usage seconds, call frequency, SMS frequency, distinct calls, age group, tariff plan, account status, age, customer value

### Models Trained
| Model | ROC-AUC | CV ROC-AUC (5-fold) |
|---|---|---|
| **Random Forest** ⭐ | **0.984** | **0.982 ± 0.003** |
| Logistic Regression | 0.924 | 0.934 ± 0.008 |
| Gradient Boosting | 0.971 | — |

### Class Imbalance Handling
Class imbalance (84.3% retained vs 15.7% churned) handled via `class_weight='balanced'` — no oversampling required.

### How to Run

```bash
pip install pandas matplotlib seaborn scikit-learn openpyxl
# Place Churn_Dataset.xlsx in the analysis/ folder, then:
python analysis/churn_analysis.py
```

---

## Visualisations (`analysis/visualizations.py`)

Five standalone, presentation-ready charts. Each has plain-English labels, insight callouts, and a "how to read" footer for both technical and non-technical audiences.

| File | Title | Audience |
|---|---|---|
| `viz1_overview.png` | Customer Churn Overview Dashboard | Stakeholders / board |
| `viz2_distributions.png` | Churned vs Retained Subscriber Behaviour | Data team / IT |
| `viz3_segments.png` | Which Segments Are Most At Risk? | Marketing / retention |
| `viz4_value.png` | Revenue at Risk — Customer Value vs Churn | CFO / leadership |
| `viz5_model.png` | Machine Learning Model Results | Technical reviewers |

```bash
# Place Churn_Dataset.xlsx in the same folder, then:
python analysis/visualizations.py
# → Saves 5 PNG files to the same folder
```

---

## Key Findings

**Top churn predictors (Random Forest feature importance):**

| Rank | Feature | Importance | Business meaning |
|---|---|---|---|
| 1 | Account Status | 18.8% | Inactive accounts are the strongest churn signal |
| 2 | Seconds of Use | 13.6% | Low usage precedes departure |
| 3 | Call Frequency | 13.6% | Fewer calls = early warning sign |
| 4 | Complaint History | 12.7% | Complained subscribers churn 4× more |
| 5 | Customer Value | 9.4% | High-value churners = biggest revenue risk |

**Segment findings:**
- Pay-as-you-go subscribers churn at nearly double the rate of contract customers
- Youngest age group (18–25) shows highest churn rate across all tariff plans
- Inactive accounts churn at dramatically higher rates regardless of other factors

---

## AI Retention Hub (`frontend/index.html`)

A fully functional browser-based tool with six AI-powered modules. Supports four AI providers — no installation required.

### Modules

| Module | What it does |
|---|---|
| Overview | KPI dashboard, feature importance chart, high-risk watchlist |
| Risk Scanner | Enter subscriber data → AI risk score + recommended actions |
| Retention Email | Auto-generates personalised outreach emails for at-risk subscribers |
| Offer Builder | Designs targeted retention offers by segment and churn reason |
| AI Analyst | Chat interface with full dataset context baked in |
| Action Plan | Generates structured 30/90-day retention plans with KPIs |

### Supported AI Providers

| Provider | Model | Cost |
|---|---|---|
| Claude (Anthropic) | claude-sonnet-4 | Paid — recommended for final demo |
| GPT-4o (OpenAI) | gpt-4o | Paid |
| Gemini Flash-Lite (Google) | gemini-2.0-flash-lite | **Free tier** |
| Llama 3.3 (Groq) | llama-3.3-70b-versatile | **Free tier** |

### How to Open

```bash
# Option 1 — Direct (Claude/OpenAI work from file://)
# Just open frontend/index.html in your browser

# Option 2 — Local server (needed for Gemini/Groq due to CORS)
python -m http.server 8080
# Open: http://localhost:8080/frontend/index.html

# Option 3 — GitHub Pages (permanent shareable URL)
# Push to GitHub → Settings → Pages → Source: main branch
```

---

## GenAI Integration — Where and How

Generative AI (Claude by Anthropic) is used in three documented ways:

**1. Code structuring and development**
Claude was used to structure the ML pipeline, generate helper functions, and debug visualisation code — demonstrating how GenAI accelerates data science workflows.

**2. Customer risk report generation**
The Risk Scanner sends subscriber profile data to the Claude API and receives a plain-English risk assessment with specific recommended actions — translating ML outputs into language a retention agent can act on immediately.

**3. Retention content automation**
The Email Generator, Offer Builder, and Action Plan modules use Claude to generate personalised emails, structured offers, and strategic plans — all grounded in the actual dataset context injected into the system prompt.

**Why Claude:** Strong instruction-following, high-quality professional writing, and Anthropic's responsible AI focus — aligned with embedding trustworthy AI into business operations.

---

## AI Strategy for Haldane Group

| Timeframe | Application |
|---|---|
| 0–6 months | Weekly at-risk customer list from K8 ERP data · AI-generated account manager emails via Phocas BI · Complaint triage automation |
| 6–18 months | Customer lifetime value scoring · Predictive stock ordering · Churn early-warning dashboard in Phocas |
| 18+ months | Trade counter AI assistant · Automated pricing intelligence · Full CRM integration |

---

## API Key Security

| Practice | Implementation |
|---|---|
| `.env` in `.gitignore` | Real keys never committed to GitHub |
| `.env.example` committed | Shows key structure without exposing secrets |
| Frontend prompts at runtime | No hardcoded credentials in source code |
| `backend/app.py` available | Key stays server-side for production deployment |

**If you accidentally commit a key — rotate it immediately at console.anthropic.com, then:**
```bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" HEAD
git push origin --force
```

---

## Full Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/haldane-churn-project.git
cd haldane-churn-project

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Environment setup
cp .env.example .env
# Add your key: ANTHROPIC_API_KEY=sk-ant-...

# 4. Run ML analysis and generate charts
python analysis/churn_analysis.py
python analysis/visualizations.py

# 5. Open the AI Retention Hub
open frontend/index.html
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Machine learning | Python · scikit-learn · pandas · numpy |
| Visualisation | matplotlib · seaborn |
| Frontend | HTML5 · CSS3 · Vanilla JavaScript |
| Generative AI | Anthropic Claude (primary) · OpenAI · Gemini · Groq |
| Backend proxy | Python Flask |
| Version control | Git / GitHub |

---

## Submission Details

- **Role:** AI & Data Analyst – KTP Associate · ref: 26/113195
- **Partnership:** Haldane Group Limited × Queen's University Belfast
- **Deadline:** 5:00 pm, Friday 10 April 2026
- **Format:** Recorded presentation, maximum 10 minutes

> *"This project demonstrates how machine learning and generative AI can be combined into a deployable retention system — moving from raw data to actionable business intelligence. The same architecture that predicts telecom subscriber churn can be embedded directly into Haldane Group's customer operations to protect trade account revenue."*
