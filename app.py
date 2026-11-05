import gradio as gr
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import traceback

class ResidualDiabetesFilterNN(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )
        self.residual_head = nn.Linear(32, 1)
        self.lgb_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x_tab, lgb_logit):
        features = self.feature_extractor(x_tab)
        delta = self.residual_head(features).squeeze(-1)
        return (self.lgb_weight * lgb_logit) + delta

# Load Model Artifacts
print("Loading model artifacts...")
heart_payload = joblib.load('heart_disease_ensemble.joblib')
diabetes_payload = joblib.load('diabetes_residual_pipeline.joblib')

# Version-safe imputation precomputation
imputer_stats = pd.Series(
    diabetes_payload['imputer'].statistics_, 
    index=diabetes_payload['encoded_feature_names']
)

# Reconstitute PyTorch Residual Nets
tabular_dim = diabetes_payload['tabular_dim']
diabetes_nn_models = []
for state in diabetes_payload['pytorch_state_dicts']:
    net = ResidualDiabetesFilterNN(tabular_dim)
    net.load_state_dict(state)
    net.eval()
    diabetes_nn_models.append(net)

print("Artifacts loaded successfully!")

def get_cardiac_recommendations(is_high_risk, bp, max_hr, exang):
    if is_high_risk:
        return f"""
        <div style="margin-top: 18px; padding: 18px; background: #231b1b; border: 1px solid #c0392b; border-radius: 10px;">
            <h3 style="color: #e74c3c; margin-top: 0;">⚠️ Targeted Cardiovascular Intervention Protocol</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                <div style="background: #1a1616; padding: 12px; border-radius: 8px;">
                    <h4 style="color: #f39c12; margin: 0 0 6px 0;">🥗 Heart-Healthy Dietary Plan (DASH / Mediterranean)</h4>
                    <ul style="color: #bdc3c7; font-size: 0.9em; padding-left: 18px; margin: 0; line-height: 1.5;">
                        <li><strong>Sodium Restriction:</strong> Cap dietary sodium strictly under 1,500–2,000 mg/day to alleviate vascular pressure.</li>
                        <li><strong>Saturated Fat Cap:</strong> Keep saturated fats under 6% of total caloric intake; eliminate trans fats and processed meats.</li>
                        <li><strong>Endothelial Support:</strong> Consume 2 servings/week of fatty fish (omega-3 EPA/DHA) and extra virgin olive oil.</li>
                        <li><strong>Soluble Fiber:</strong> Aim for 30g+ daily from oats, psyllium, and legumes to promote LDL clearance.</li>
                    </ul>
                </div>
                <div style="background: #1a1616; padding: 12px; border-radius: 8px;">
                    <h4 style="color: #f39c12; margin: 0 0 6px 0;">🏃 Prescribed Exercise Protocol</h4>
                    <ul style="color: #bdc3c7; font-size: 0.9em; padding-left: 18px; margin: 0; line-height: 1.5;">
                        <li><strong>Pacing & Zone 2:</strong> 150 min/week of moderate continuous cardio (brisk flat walking, recumbent cycling). Avoid sudden HIIT or straining maneuvers.</li>
                        <li><strong>Exertion Warning:</strong> {'Exercise-induced angina was noted. Cease training immediately if experiencing tightness or radiation to left arm.' if exang == "Yes" else 'Keep exertion at conversational pace (RPE 4-5 out of 10).'}</li>
                        <li><strong>Isometric Precautions:</strong> Avoid heavy overhead pressing or breath-holding (Valsalva maneuver) which cause dangerous BP spikes.</li>
                        <li><strong>Clinical Clearance:</strong> Undergo a formal cardiologist-supervised treadmill ECG test before new training routines.</li>
                    </ul>
                </div>
            </div>
        </div>
        """
    else:
        return """
        <div style="margin-top: 18px; padding: 18px; background: #16221c; border: 1px solid #27ae60; border-radius: 10px;">
            <h3 style="color: #2ecc71; margin-top: 0;">🛡️ Cardiovascular Maintenance & Longevity Protocol</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                <div style="background: #121a16; padding: 12px; border-radius: 8px;">
                    <h4 style="color: #2ecc71; margin: 0 0 6px 0;">🥗 Cardioprotective Diet</h4>
                    <ul style="color: #bdc3c7; font-size: 0.9em; padding-left: 18px; margin: 0; line-height: 1.5;">
                        <li>Emphasize Mediterranean patterns: unrefined grains, leafy greens, walnuts, and seasonal fruits.</li>
                        <li>Maintain balanced potassium and magnesium intake to support healthy arterial elasticity.</li>
                        <li>Limit intake of sugar-sweetened beverages and ultra-processed baked snacks.</li>
                    </ul>
                </div>
                <div style="background: #121a16; padding: 12px; border-radius: 8px;">
                    <h4 style="color: #2ecc71; margin: 0 0 6px 0;">🏃 Conditioning & Aerobic Fitness</h4>
                    <ul style="color: #bdc3c7; font-size: 0.9em; padding-left: 18px; margin: 0; line-height: 1.5;">
                        <li>Aim for 150–300 minutes of weekly moderate aerobic activity or 75 minutes of vigorous exercise.</li>
                        <li>Include 2–3 full-body resistance training sessions weekly to preserve muscle and arterial compliance.</li>
                        <li>Target regular daily movement: 8,000–10,000 steps per day to minimize prolonged venous pooling.</li>
                    </ul>
                </div>
            </div>
        </div>
        """

def get_diabetes_recommendations(is_high_risk, bmi, diet_score, act_min):
    if is_high_risk:
        return f"""
        <div style="margin-top: 18px; padding: 18px; background: #231b1b; border: 1px solid #c0392b; border-radius: 10px;">
            <h3 style="color: #e74c3c; margin-top: 0;">⚠️ Glycemic & Insulin Sensitivity Protocol</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                <div style="background: #1a1616; padding: 12px; border-radius: 8px;">
                    <h4 style="color: #f39c12; margin: 0 0 6px 0;">🥗 Low-Glycemic Load Nutritional Strategy</h4>
                    <ul style="color: #bdc3c7; font-size: 0.9em; padding-left: 18px; margin: 0; line-height: 1.5;">
                        <li><strong>Carbohydrate Pairing:</strong> Never eat starchy carbohydrates alone; always buffer them with fiber, lean proteins, or healthy fats.</li>
                        <li><strong>Sugar & Starch Elimination:</strong> Replace refined grains (white flour, polished rice, soda) with complex carbs (lentils, barley, beans).</li>
                        <li><strong>Meal Timing:</strong> Implement early time-restricted feeding (e.g., 10-hour daytime eating window) to improve peripheral insulin response.</li>
                        <li><strong>Weight Goal:</strong> {'A 5–7% reduction in body weight can significantly reverse prediabetic insulin resistance.' if bmi > 25 else 'Maintain current lean mass while shifting body composition away from visceral abdominal fat.'}</li>
                    </ul>
                </div>
                <div style="background: #1a1616; padding: 12px; border-radius: 8px;">
                    <h4 style="color: #f39c12; margin: 0 0 6px 0;">🏋️ Exercise for GLUT4 Translocation</h4>
                    <ul style="color: #bdc3c7; font-size: 0.9em; padding-left: 18px; margin: 0; line-height: 1.5;">
                        <li><strong>Post-Prandial Walks:</strong> A 10–15 minute brisk walk immediately after heavy meals forces muscle contraction to clear post-meal glucose spikes without requiring excess insulin.</li>
                        <li><strong>Hypertrophy Resistance Training:</strong> 3 days/week of progressive overload on large muscle groups (squats, deadlifts, presses). Skeletal muscle is your body's largest glucose disposal sink.</li>
                        <li><strong>Break Up Sedentary Time:</strong> Stand and perform light stretches or calf raises for 2 minutes after every 30 minutes of desk work.</li>
                    </ul>
                </div>
            </div>
        </div>
        """
    else:
        return """
        <div style="margin-top: 18px; padding: 18px; background: #16221c; border: 1px solid #27ae60; border-radius: 10px;">
            <h3 style="color: #2ecc71; margin-top: 0;">🛡️ Metabolic Health & Glucose Homeostasis Protocol</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                <div style="background: #121a16; padding: 12px; border-radius: 8px;">
                    <h4 style="color: #2ecc71; margin: 0 0 6px 0;">🥗 High-Satiety Metabolic Nutrition</h4>
                    <ul style="color: #bdc3c7; font-size: 0.9em; padding-left: 18px; margin: 0; line-height: 1.5;">
                        <li>Prioritize a whole-food diet rich in dietary polyphenols, berries, cruciferous vegetables, and fermented foods.</li>
                        <li>Maintain consistent daily fiber goals (35g+) to promote gut microbiome short-chain fatty acid (SCFA) production.</li>
                        <li>Keep late-night eating to a minimum to maintain natural overnight fasting insulin sensitivity.</li>
                    </ul>
                </div>
                <div style="background: #121a16; padding: 12px; border-radius: 8px;">
                    <h4 style="color: #2ecc71; margin: 0 0 6px 0;">🏋️ Physical Activity & Metabolic Optimization</h4>
                    <ul style="color: #bdc3c7; font-size: 0.9em; padding-left: 18px; margin: 0; line-height: 1.5;">
                        <li>Maintain 150+ minutes of weekly varied aerobic conditioning (jogging, cycling, lap swimming).</li>
                        <li>Maintain lean muscle tissue through consistent full-body compound resistance training twice weekly.</li>
                        <li>Prioritize 7–8 hours of restorative sleep; poor sleep is a primary non-dietary cause of acute insulin resistance.</li>
                    </ul>
                </div>
            </div>
        </div>
        """

def predict_heart(age, sex, cp, bp, max_hr, exang, chol, fbs, ekg, st_dep, slope, vessels, thal):
    try:
        sex_val = 1 if sex == "Male" else 0
        cp_map = {"Typical Angina (1)": 1, "Atypical Angina (2)": 2, "Non-anginal Pain (3)": 3, "Asymptomatic (4)": 4}
        exang_val = 1 if exang == "Yes" else 0

        user_dict = {
            'Age': float(age),
            'Sex': sex_val,
            'Chest pain type': cp_map[cp],
            'BP': float(bp),
            'Max HR': float(max_hr),
            'Exercise angina': exang_val,
            'Cholesterol': float(chol) if (chol is not None and chol > 0) else np.nan,
            'FBS over 120': (1 if fbs == "Yes" else 0) if fbs != "Not Tested" else np.nan,
            'EKG results': int(ekg.split()[0]) if ekg != "Not Tested" else np.nan,
            'ST depression': float(st_dep) if (st_dep is not None) else np.nan,
            'Slope of ST': int(slope.split()[0]) if slope != "Not Tested" else np.nan,
            'Number of vessels fluro': float(vessels) if (vessels is not None) else np.nan,
            'Thallium': int(thal.split()[0]) if thal != "Not Tested" else np.nan
        }

        feature_order = heart_payload['feature_order']
        row = {c: user_dict.get(c, np.nan) for c in feature_order}
        df_input = pd.DataFrame([row])

        preds = [m.predict_proba(df_input)[:, 1][0] for m in heart_payload['models']]
        prob = float(np.mean(preds))

        is_triage = any(pd.isna(user_dict[f]) for f in heart_payload['optional_features'])
        thresh = heart_payload.get('threshold_triage', 0.43) if is_triage else heart_payload.get('threshold_full', 0.43)

        is_high = prob >= thresh
        risk_label = "HIGH RISK (Presence Detected)" if is_high else "LOW RISK (Absence)"
        badge_bg = "#e74c3c" if is_high else "#27ae60"
        card_border = "#e74c3c" if is_high else "#2ecc71"
        mode_text = "Home Triage Mode (Clinical lab inputs omitted)" if is_triage else "Comprehensive Clinical Diagnostic Suite"

        recommendation_html = get_cardiac_recommendations(is_high, bp, max_hr, exang)

        return f"""
        <div style="padding: 24px; border-radius: 14px; background: #161a22; border: 1px solid {card_border}; box-shadow: 0 8px 24px rgba(0,0,0,0.45); font-family: system-ui, -apple-system, sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #283040; padding-bottom: 14px;">
                <span style="background-color: {badge_bg}; color: white; padding: 6px 16px; border-radius: 30px; font-weight: 700; font-size: 0.85em; letter-spacing: 0.5px;">
                    {risk_label}
                </span>
                <span style="color: #8b949e; font-size: 0.88em;">Mode: <strong style="color: #c9d1d9;">{mode_text}</strong></span>
            </div>
            
            <div style="margin: 20px 0 10px 0;">
                <p style="color: #8b949e; margin: 0 0 4px 0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Predicted Cardiac Risk</p>
                <h1 style="color: {'#ff7b72' if is_high else '#56d364'}; margin: 0; font-size: 2.8rem; font-weight: 800;">
                    {prob * 100:.1f}%
                </h1>
                <p style="color: #8b949e; margin: 4px 0 0 0; font-size: 0.88em;">Calibrated Decision Threshold: <strong style="color: #f0f6fc;">{thresh:.2f}</strong></p>
            </div>

            <div style="background-color: #0d1117; padding: 14px; border-radius: 8px; border-left: 4px solid #58a6ff; font-size: 0.9em; color: #c9d1d9; margin-top: 15px;">
                <strong>Statistical Confidence:</strong> Ensemble cross-validated accuracy is <strong>88.9%</strong> on complete cardiac profiles and <strong>83.2%</strong> in home triage mode.
            </div>

            {recommendation_html}
        </div>
        """
    except Exception as e:
        return f"<div style='color: #ff7b72; padding: 15px;'><strong>Inference Error:</strong> {str(e)}<pre>{traceback.format_exc()}</pre></div>"

def predict_diabetes(age, gender, ethnicity, edu, income, emp, alc, act, diet, sleep, screen, 
                     bmi, whr, sbp, dbp, hr, fam, hyp, cvd, smoke,
                     chol_tot, hdl, ldl, trig):
    try:
        user_dict = {
            'age': float(age),
            'gender': str(gender),
            'ethnicity': str(ethnicity),
            'education_level': str(edu),
            'income_level': str(income),
            'employment_status': str(emp),
            'alcohol_consumption_per_week': float(alc),
            'physical_activity_minutes_per_week': float(act),
            'diet_score': float(diet),
            'sleep_hours_per_day': float(sleep),
            'screen_time_hours_per_day': float(screen),
            'bmi': float(bmi),
            'waist_to_hip_ratio': float(whr),
            'systolic_bp': float(sbp),
            'diastolic_bp': float(dbp),
            'heart_rate': float(hr),
            'family_history_diabetes': 1.0 if fam == "Yes" else 0.0,
            'hypertension_history': 1.0 if hyp == "Yes" else 0.0,
            'cardiovascular_history': 1.0 if cvd == "Yes" else 0.0,
            'smoking_status': str(smoke),
            'cholesterol_total': float(chol_tot) if (chol_tot is not None and chol_tot > 0) else np.nan,
            'hdl_cholesterol': float(hdl) if (hdl is not None and hdl > 0) else np.nan,
            'ldl_cholesterol': float(ldl) if (ldl is not None and ldl > 0) else np.nan,
            'triglycerides': float(trig) if (trig is not None and trig > 0) else np.nan
        }

        # 1. Stage 1: LightGBM
        row_raw = {c: user_dict.get(c, np.nan) for c in diabetes_payload['all_features']}
        df_lgb = pd.DataFrame([row_raw])
        for c in diabetes_payload['categorical_features']:
            df_lgb[c] = df_lgb[c].astype('category')

        lgb_preds = [m.predict_proba(df_lgb)[:, 1][0] for m in diabetes_payload['lgbm_models']]
        mean_lgb = float(np.mean(lgb_preds))

        is_triage = any(pd.isna(user_dict[f]) for f in diabetes_payload['optional_features'])

        # 2. Stage 2: Deep Residual Filter
        d = pd.DataFrame([row_raw])
        if not pd.isna(d['triglycerides'].iloc[0]) and not pd.isna(d['hdl_cholesterol'].iloc[0]):
            d['tg_to_hdl'] = d['triglycerides'] / (d['hdl_cholesterol'] + 1e-5)
            d['non_hdl'] = d['cholesterol_total'] - d['hdl_cholesterol']
        else:
            d['tg_to_hdl'] = np.nan
            d['non_hdl'] = np.nan
            
        d['bmi_x_age'] = (d['bmi'] * d['age']) / 100.0
        d['map'] = d['diastolic_bp'] + (d['systolic_bp'] - d['diastolic_bp']) / 3.0
        d['sedentary_ratio'] = d['screen_time_hours_per_day'] / (d['physical_activity_minutes_per_week'] / 60.0 + 0.1)

        df_encoded = pd.get_dummies(d, columns=diabetes_payload['categorical_features'], drop_first=True)
        
        for col in diabetes_payload['encoded_feature_names']:
            if col not in df_encoded.columns:
                df_encoded[col] = 0.0
            else:
                df_encoded[col] = df_encoded[col].astype(float)
                
        df_aligned = df_encoded[diabetes_payload['encoded_feature_names']].copy()

        # Robust Imputation & Scaling
        df_imputed = df_aligned.fillna(imputer_stats)
        x_scaled = diabetes_payload['scaler'].transform(df_imputed)

        # PyTorch Residual Forward Pass
        eps = 1e-6
        clipped_lgb = np.clip(mean_lgb, eps, 1.0 - eps)
        lgb_logit = float(np.log(clipped_lgb / (1.0 - clipped_lgb)))

        t_x = torch.tensor(x_scaled, dtype=torch.float32)
        t_logit = torch.tensor([lgb_logit], dtype=torch.float32)

        with torch.no_grad():
            nn_preds = [torch.sigmoid(m(t_x, t_logit)).item() for m in diabetes_nn_models]
        final_prob = float(np.mean(nn_preds))

        thresh = float(diabetes_payload.get('threshold', 0.5020))
        is_high = final_prob >= thresh
        risk_label = "HIGH RISK (Diabetic Markers Detected)" if is_high else "LOW RISK / NORMAL"
        badge_bg = "#e74c3c" if is_high else "#27ae60"
        card_border = "#e74c3c" if is_high else "#2ecc71"
        mode_text = "Home Lifestyle & Biometric Triage" if is_triage else "Complete Clinical Lipid Panel"

        recommendation_html = get_diabetes_recommendations(is_high, bmi, diet, act)

        return f"""
        <div style="padding: 24px; border-radius: 14px; background: #161a22; border: 1px solid {card_border}; box-shadow: 0 8px 24px rgba(0,0,0,0.45); font-family: system-ui, -apple-system, sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #283040; padding-bottom: 14px;">
                <span style="background-color: {badge_bg}; color: white; padding: 6px 16px; border-radius: 30px; font-weight: 700; font-size: 0.85em; letter-spacing: 0.5px;">
                    {risk_label}
                </span>
                <span style="color: #8b949e; font-size: 0.88em;">Mode: <strong style="color: #c9d1d9;">{mode_text}</strong></span>
            </div>
            
            <div style="margin: 20px 0 10px 0;">
                <p style="color: #8b949e; margin: 0 0 4px 0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Predicted Diabetes Probability</p>
                <h1 style="color: {'#ff7b72' if is_high else '#56d364'}; margin: 0; font-size: 2.8rem; font-weight: 800;">
                    {final_prob * 100:.1f}%
                </h1>
                <p style="color: #8b949e; margin: 4px 0 0 0; font-size: 0.88em;">Calibrated Decision Threshold: <strong style="color: #f0f6fc;">{thresh:.4f}</strong></p>
            </div>

            <div style="background-color: #0d1117; padding: 14px; border-radius: 8px; border-left: 4px solid #bc8cff; font-size: 0.9em; color: #c9d1d9; margin-top: 15px;">
                <strong>Architecture:</strong> Two-Stage LightGBM + Deep PyTorch Residual Filter with false-positive suppression active.
            </div>

            {recommendation_html}
        </div>
        """
    except Exception as e:
        return f"<div style='color: #ff7b72; padding: 15px;'><strong>Inference Error:</strong> {str(e)}<pre>{traceback.format_exc()}</pre></div>"

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate"
)

with gr.Blocks(theme=custom_theme, title="CardioMetabolic Risk AI Suite") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 25px 0 10px 0;">
        <h1 style="font-size: 2.4rem; font-weight: 800; color: #58a6ff; margin-bottom: 5px;">🩺 CardioMetabolic AI Diagnostic Suite</h1>
        <p style="color: #8b949e; font-size: 1.1rem; max-width: 750px; margin: 0 auto;">
            Dual-Pipeline Predictive Screening with Missingness-Tolerant Gradient Boosting & Deep Neural Residual Networks.
        </p>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("❤️ Heart Disease Assessment"):
            gr.Markdown("### 📋 Step 1: Self-Reported Vitals (Home Accessible)")
            with gr.Row():
                hd_age = gr.Slider(20, 90, value=55, label="Age (Years)", step=1)
                hd_sex = gr.Radio(["Male", "Female"], value="Male", label="Biological Sex")
                hd_cp = gr.Dropdown(["Typical Angina (1)", "Atypical Angina (2)", "Non-anginal Pain (3)", "Asymptomatic (4)"], 
                                    value="Atypical Angina (2)", label="Chest Pain Type (See 'Clinical Info' tab)")
            with gr.Row():
                hd_bp = gr.Number(value=130, label="Resting Systolic Blood Pressure (mm Hg)")
                hd_max_hr = gr.Number(value=150, label="Maximum Heart Rate Achieved (bpm)")
                hd_exang = gr.Radio(["No", "Yes"], value="No", label="Exercise-Induced Angina / Tightness")

            with gr.Accordion("🔬 Advanced Clinical Tests & Stress Labs (Optional - Leave default if not tested)", open=False):
                with gr.Row():
                    hd_chol = gr.Number(value=0, label="Serum Cholesterol (mg/dl) [0 = Unknown]")
                    hd_fbs = gr.Radio(["Not Tested", "No", "Yes"], value="Not Tested", label="Fasting Blood Sugar > 120 mg/dl")
                    hd_ekg = gr.Dropdown(["Not Tested", "0 (Normal)", "1 (ST-T Wave Abnormality)", "2 (Left Ventricular Hypertrophy)"], 
                                         value="Not Tested", label="Resting EKG Results")
                with gr.Row():
                    hd_st_dep = gr.Number(value=None, label="ST Depression Induced by Exercise")
                    hd_slope = gr.Dropdown(["Not Tested", "1 (Upsloping)", "2 (Flat)", "3 (Downsloping)"], 
                                           value="Not Tested", label="Slope of Peak Exercise ST Segment")
                    hd_vessels = gr.Slider(0, 3, value=0, step=1, label="Major Vessels Colored by Fluoroscopy (0-3)")
                    hd_thal = gr.Dropdown(["Not Tested", "3 (Normal)", "6 (Fixed Defect)", "7 (Reversible Defect)"], 
                                           value="Not Tested", label="Thallium Scintigraphy Stress Test")

            hd_btn = gr.Button("🔍 Run Cardiac Diagnostic Evaluation", variant="primary", size="lg")
            hd_output = gr.HTML(label="Diagnostic Assessment Report")

            hd_btn.click(
                predict_heart,
                inputs=[hd_age, hd_sex, hd_cp, hd_bp, hd_max_hr, hd_exang, hd_chol, hd_fbs, hd_ekg, hd_st_dep, hd_slope, hd_vessels, hd_thal],
                outputs=hd_output
            )

        with gr.TabItem("🩸 Diabetes Metabolic Screening"):
            gr.Markdown("### 📋 Step 1: Lifestyle & Home Biometrics (Mandatory)")
            with gr.Row():
                db_age = gr.Slider(18, 100, value=45, label="Age (Years)", step=1)
                db_gender = gr.Dropdown(["Male", "Female"], value="Female", label="Gender")
                db_ethnicity = gr.Dropdown(["White", "Black", "Hispanic", "Asian", "Other"], value="White", label="Ethnicity")
            with gr.Row():
                db_edu = gr.Dropdown(["Highschool", "Graduate", "Postgraduate"], value="Highschool", label="Education Level")
                db_income = gr.Dropdown(["Low", "Lower-Middle", "Upper-Middle", "High"], value="Lower-Middle", label="Income Tier")
                db_emp = gr.Dropdown(["Employed", "Unemployed", "Retired"], value="Employed", label="Employment Status")
            with gr.Row():
                db_bmi = gr.Number(value=27.5, label="BMI (Body Mass Index)")
                db_whr = gr.Number(value=0.85, label="Waist-to-Hip Ratio")
                db_sbp = gr.Number(value=122, label="Systolic Blood Pressure (mm Hg)")
                db_dbp = gr.Number(value=80, label="Diastolic Blood Pressure (mm Hg)")
                db_hr = gr.Number(value=74, label="Resting Heart Rate (bpm)")
            with gr.Row():
                db_act = gr.Slider(0, 600, value=120, label="Physical Activity (Minutes/Week)", step=10)
                db_diet = gr.Slider(0.0, 10.0, value=6.0, label="Diet Quality Score (0 = Poor, 10 = Ideal)", step=0.1)
                db_sleep = gr.Slider(3.0, 12.0, value=7.0, label="Sleep Duration (Hours/Day)", step=0.5)
                db_screen = gr.Slider(0.0, 16.0, value=6.0, label="Screen Time (Hours/Day)", step=0.5)
                db_alc = gr.Slider(0, 30, value=2, label="Alcohol Intake (Drinks/Week)", step=1)
            with gr.Row():
                db_smoke = gr.Dropdown(["Never", "Former", "Current"], value="Never", label="Smoking Status")
                db_fam = gr.Radio(["No", "Yes"], value="No", label="Immediate Family History of Diabetes")
                db_hyp = gr.Radio(["No", "Yes"], value="No", label="Clinical Hypertension History")
                db_cvd = gr.Radio(["No", "Yes"], value="No", label="Cardiovascular Disease History")

            with gr.Accordion("🧪 Advanced Lipid Panel Blood Draw (Optional - Leave 0 if not tested)", open=False):
                with gr.Row():
                    db_chol = gr.Number(value=0, label="Total Cholesterol (mg/dL) [0 = Unknown]")
                    db_hdl = gr.Number(value=0, label="HDL ('Good') Cholesterol (mg/dL) [0 = Unknown]")
                    db_ldl = gr.Number(value=0, label="LDL ('Bad') Cholesterol (mg/dL) [0 = Unknown]")
                    db_trig = gr.Number(value=0, label="Triglycerides (mg/dL) [0 = Unknown]")

            db_btn = gr.Button("🔍 Run Diabetes Risk Evaluation", variant="primary", size="lg")
            db_output = gr.HTML(label="Diagnostic Assessment Report")

            db_btn.click(
                predict_diabetes,
                inputs=[
                    db_age, db_gender, db_ethnicity, db_edu, db_income, db_emp, db_alc, db_act, db_diet, db_sleep, db_screen,
                    db_bmi, db_whr, db_sbp, db_dbp, db_hr, db_fam, db_hyp, db_cvd, db_smoke,
                    db_chol, db_hdl, db_ldl, db_trig
                ],
                outputs=db_output
            )

        with gr.TabItem("ℹ️ Clinical Info & Project Architecture"):
            gr.Markdown("""
            ### 🩺 Clinical Guide: Chest Pain Classifications (Diamond-Forrester Standard)
            Chest pain is clinically classified based on three primary criteria:
            1. Substernal chest discomfort with typical character and duration.
            2. Provoked by exertion or emotional stress.
            3. Relieved by rest or sublingual nitroglycerin within minutes.

            * **Typical Angina (Type 1):** Meets **all 3** criteria. Has the strongest correlation with critical coronary artery blockage.
            * **Atypical Angina (Type 2):** Meets **2 of the 3** criteria. Frequently observed in female and diabetic patients where cardiac pain presents atypically (such as shortness of breath, jaw, or upper back pain).
            * **Non-Anginal Pain (Type 3):** Meets **1 or none** of the criteria. Typical of musculoskeletal soreness, acid reflux (GERD), or pleurisy.
            * **Asymptomatic (Type 4):** Total lack of chest discomfort. Common in long-term diabetic patients due to autonomic neuropathy dulling visceral heart sensations ("silent ischemia").

            ---

            ### 🔬 Technical System Architecture
            
            #### 1. Heart Disease Pipeline
            * **Ensemble Architecture:** 5-Fold Stratified LightGBM Gradient-Boosted Decision Trees.
            * **Dynamic Missingness Tolerance:** Features trained via missingness dropout augmentation, achieving **91.0% ROC-AUC** in Home Triage Mode and **95.5% ROC-AUC** in Full Clinical Mode.
            * **Calibrated Operating Point:** Set to $\\tau = 0.43$ via Youden's Index to prioritize clinical recall.

            #### 2. Diabetes Metabolic Screening Pipeline
            * **Two-Stage Cascade Architecture:**
                * **Stage 1 (Primary Triage):** 5-Fold LightGBM estimating baseline likelihood across 20 non-invasive features.
                * **Stage 2 (False-Positive Suppressor):** Deep PyTorch Multi-Layer Perceptron (MLP) with a residual skip connection.
            * **Impact:** Reduced false alarms by **over 55,000 cases** while achieving an optimal operating threshold of $\\tau = 0.5020$.
            """)

# Launch App
if __name__ == "__main__":
    demo.launch(inbrowser=True)