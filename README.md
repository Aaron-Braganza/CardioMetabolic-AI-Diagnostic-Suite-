# CardioMetabolic AI Diagnostic Suite

An end-to-end clinical machine learning framework for simultaneous screening of Cardiovascular Disease and Type-2 Diabetes Mellitus.

##  Key Features
- **Dual-Mode Cardiac Screening:** Operates via a 5-fold LightGBM ensemble in Home Triage Mode (6 vitals, 91.0% ROC-AUC) or Comprehensive Mode (13 labs, 95.5% ROC-AUC) utilizing native tree missingness routing.
- **Two-Stage Diabetes Cascade:** Filters baseline risk with LightGBM and refines boundary cases with a PyTorch Deep Residual MLP, cutting false positives by 35.1% (tau = 0.5020).
- **Interactive Gradio Interface:** Clean, local web-based dashboard with dynamic DASH, Mediterranean, and GLUT4 exercise prescriptions.