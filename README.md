# 🧪 A/B Testing Simulator with Advanced Analytics

## Overview
This project simulates a real-world A/B test using synthetic user-level interaction data across two variants (A = Control, B = Test). It includes traditional hypothesis testing, Bayesian inference, power analysis, and uplift modeling for deeper experimentation insights.

## Objective
- Evaluate if Variant B outperforms A in conversion and revenue metrics
- Apply Frequentist and Bayesian A/B testing techniques
- Analyze user segments to identify differential treatment effects

## Dataset
The dataset (`ab_test_synthetic_data.csv`) includes:
- **Group**: A or B (control vs test)
- **Converted**: 1 if user converted, 0 otherwise
- **Revenue**: Generated revenue per user (0 if not converted)
- **Segment**: User channel (Mobile, Web, Email)

## Key Features
- ✅ Chi-Square Test for conversion rate difference
- 💰 T-Test for revenue difference
- 📉 Power Analysis to estimate required sample size
- 🧠 Bayesian Simulation with posterior conversion rates
- 📊 Uplift Modeling via Logistic Regression with segments

## How to Run
```bash
pip install pandas numpy seaborn matplotlib scipy scikit-learn
python ab_testing_simulator.py
```

## Visualizations
- Conversion rate bars
- Revenue distribution (converted users)
- Bayesian posterior plots
- Segment-wise uplift prediction

## File Structure
- `ab_test_synthetic_data.csv` — Input dataset
- `ab_testing_simulator.py` — Main analysis script

## Future Enhancements
- Streamlit-based simulator
- Exportable HTML summary report
- Integration with real experimentation platforms
