# Data Dictionary: CDC Diabetes Health Indicators Dataset

## Overview

This data dictionary provides detailed descriptions of all features in the CDC Diabetes Health Indicators Dataset, derived from the Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey.

**Dataset Size**: balanced subset of ~50,000 survey responses (original size: 253,680)
**Features**: 21 health and lifestyle indicators  
**Target Variable**: `Diabetes_binary`  
**Source**: Centers for Disease Control and Prevention (CDC)  
**License**: Public Domain (U.S. Government Work)

---

## Understanding the Data

### Survey Context
The BRFSS is the nation's premier health-related telephone survey system, collecting state data about U.S. residents' health-related risk behaviors, chronic health conditions, and use of preventive services. This dataset represents a cleaned subset focusing on diabetes risk factors.

### Data Collection Method
- Conducted via telephone interviews (landline and cellular)
- Standardized questionnaire across all states
- Self-reported health information
- Anonymous and de-identified

---

## Feature Descriptions

### Target Variable

| Feature | Description | Values | Notes |
|---------|-------------|--------|-------|
| **Diabetes_binary** | Diabetes diagnosis status | 0 = No diabetes<br>1 = Prediabetes or diabetes | **This is what we're predicting**. Binary classification problem. |

---

### Cardiovascular & Metabolic Features

| Feature | Description | Type | Values | Clinical Significance |
|---------|-------------|------|--------|----------------------|
| **HighBP** | High blood pressure diagnosis | Binary | 0 = No<br>1 = Yes | Strong diabetes risk factor. Hypertension and diabetes often co-occur (metabolic syndrome). |
| **HighChol** | High cholesterol diagnosis | Binary | 0 = No<br>1 = Yes | Cardiovascular risk factor. Lipid disorders common in diabetes. |
| **CholCheck** | Cholesterol check in past 5 years | Binary | 0 = No<br>1 = Yes | Indicates preventive care access and health awareness. |
| **BMI** | Body Mass Index | Continuous | Range: 12-98 | **Key obesity indicator**. Strong predictor of type 2 diabetes risk. Higher BMI = higher risk. |
| **Stroke** | History of stroke | Binary | 0 = No<br>1 = Yes | Vascular complication. May indicate advanced disease or high cardiovascular risk. |
| **HeartDiseaseorAttack** | Coronary heart disease or myocardial infarction | Binary | 0 = No<br>1 = Yes | Major cardiovascular complication. Strong association with diabetes. |

---

### Lifestyle & Behavioral Features

| Feature | Description | Type | Values | Clinical Significance |
|---------|-------------|------|--------|----------------------|
| **Smoker** | Has smoked at least 100 cigarettes in lifetime | Binary | 0 = No<br>1 = Yes | Smoking increases diabetes risk and worsens complications. |
| **PhysActivity** | Physical activity in past 30 days (outside work) | Binary | 0 = No<br>1 = Yes | **Protective factor**. Exercise improves insulin sensitivity and glucose control. |
| **Fruits** | Consumes fruit 1 or more times per day | Binary | 0 = No<br>1 = Yes | Dietary quality indicator. Part of healthy eating pattern. |
| **Veggies** | Consumes vegetables 1 or more times per day | Binary | 0 = No<br>1 = Yes | Dietary quality indicator. Fiber and nutrient intake. |
| **HvyAlcoholConsump** | Heavy alcohol consumption | Binary | 0 = No<br>1 = Yes | Adult men: >14 drinks/week<br>Adult women: >7 drinks/week. Risk factor for metabolic issues. |

---

### Healthcare Access Features

| Feature | Description | Type | Values | Clinical Significance |
|---------|-------------|------|--------|----------------------|
| **AnyHealthcare** | Any form of health care coverage | Binary | 0 = No<br>1 = Yes | Access to preventive care and early detection. Social determinant of health. |
| **NoDocbcCost** | Could not see doctor due to cost in past 12 months | Binary | 0 = No<br>1 = Yes | **Healthcare barrier**. Financial constraint affecting care access. |

---

### Self-Reported Health Status

| Feature | Description | Type | Values | Clinical Significance |
|---------|-------------|------|--------|----------------------|
| **GenHlth** | General health status | Ordinal | 1 = Excellent<br>2 = Very good<br>3 = Good<br>4 = Fair<br>5 = Poor | Overall health perception. Lower values (better health) correlate with lower diabetes risk. |
| **MentHlth** | Days of poor mental health in past 30 days | Continuous | Range: 0-30 | Mental health comorbidity. Depression common in diabetes patients. |
| **PhysHlth** | Days of poor physical health in past 30 days | Continuous | Range: 0-30 | Physical health burden. May reflect complications or comorbidities. |
| **DiffWalk** | Serious difficulty walking or climbing stairs | Binary | 0 = No<br>1 = Yes | Mobility limitation. May indicate neuropathy, obesity, or other complications. |

---

### Demographic Features

| Feature | Description | Type | Values | Clinical Significance |
|---------|-------------|------|--------|----------------------|
| **Sex** | Biological sex | Binary | 0 = Female<br>1 = Male | Men and women have different diabetes risk profiles. |
| **Age** | Age category | Ordinal | 1 = 18-24<br>2 = 25-29<br>3 = 30-34<br>4 = 35-39<br>5 = 40-44<br>6 = 45-49<br>7 = 50-54<br>8 = 55-59<br>9 = 60-64<br>10 = 65-69<br>11 = 70-74<br>12 = 75-79<br>13 = 80+ | **Strong predictor**. Diabetes risk increases with age. Type 2 diabetes typically develops in adults. |
| **Education** | Education level | Ordinal | 1 = Never attended or kindergarten only<br>2 = Grades 1-8<br>3 = Grades 9-11<br>4 = Grade 12 or GED<br>5 = College 1-3 years<br>6 = College 4+ years | Socioeconomic indicator. Correlates with health literacy and access to care. |
| **Income** | Annual household income | Ordinal | 1 = <$10,000<br>2 = $10,000-$15,000<br>3 = $15,000-$20,000<br>4 = $20,000-$25,000<br>5 = $25,000-$35,000<br>6 = $35,000-$50,000<br>7 = $50,000-$75,000<br>8 = $75,000+ | Socioeconomic status. Affects nutrition, healthcare access, stress levels. |

---

### Data Quality Considerations

**Strengths:**
- Large sample size (50,000+ responses)
- Standardized data collection methodology
- Represents diverse U.S. population
- Comprehensive feature set covering major risk factors

**Limitations:**
- Self-reported data (subject to recall bias and social desirability bias)
- No laboratory test results (glucose levels, HbA1c)
- Binary features lose granularity (e.g., BMI is continuous but many features are yes/no)
- Cross-sectional data (cannot establish causation or track disease progression)
- Prediabetes and diabetes combined into single positive class

### Clinical Interpretation Guidelines

**High-Risk Profile Typically Shows:**
- Higher age (50+)
- Elevated BMI (>30)
- High blood pressure
- High cholesterol
- Poor general health rating
- Cardiovascular complications
- Limited physical activity

**Protective Factors:**
- Regular physical activity
- Healthy diet (fruits and vegetables)
- Normal BMI (<25)
- Younger age
- Good general health
- No smoking history

---

## Feature Correlations to Watch For

### Expected Strong Correlations with Diabetes:
1. **Age** - Risk increases significantly with age
2. **BMI** - Obesity is primary modifiable risk factor
3. **HighBP** - Often part of metabolic syndrome
4. **GenHlth** - Diabetes affects overall health perception

### Multicollinearity Concerns:
- `HighBP`, `HighChol`, `BMI` may be correlated (metabolic syndrome cluster)
- `AnyHealthcare` and `NoDocbcCost` inversely related
- `MentHlth` and `PhysHlth` may correlate
- `Age` may correlate with cardiovascular complications

---

## Feature Engineering Opportunities

While not required for this project, in production systems you might consider:

**Interaction Features:**
- BMI × Age (obesity risk increases with age)
- HighBP × HighChol (metabolic syndrome indicator)
- PhysActivity × BMI (exercise protective effect varies by weight)

**Derived Features:**
- Cardiovascular risk score (combining stroke, heart disease, high BP, high cholesterol)
- Lifestyle health score (combining physical activity, diet, smoking)
- Healthcare access composite (combining healthcare coverage and cost barriers)

**Binning:**
- Age groups (young adult, middle age, elderly)
- BMI categories (underweight, normal, overweight, obese)
- Mental/physical health days (none, mild, moderate, severe)

---

## Ethical Considerations

### Bias & Fairness
- **Age bias**: Older adults overrepresented in diabetes cases (clinically accurate but model may perform poorly on younger adults)
- **Socioeconomic bias**: Income and education correlate with diabetes risk, but using them directly may perpetuate health disparities
- **Access bias**: Healthcare coverage affects diagnosis (undiagnosed diabetes in uninsured populations)

### Responsible Use
- Model is for **screening**, not diagnosis
- False negatives (missed cases) have serious health consequences
- Should be used to prioritize follow-up testing, not replace clinical judgment
- Consider fairness metrics across demographic subgroups

---

## References & Additional Resources

**Data Source:**
- CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015
- Available at: https://www.cdc.gov/brfss/

**Clinical Guidelines:**
- American Diabetes Association: https://diabetes.org/
- CDC Diabetes Prevention Program: https://www.cdc.gov/diabetes/prevention/

**Related Research:**
- Diabetes risk prediction models
- Metabolic syndrome and cardiovascular disease
- Social determinants of health in diabetes

---

## Questions to Consider During Your Analysis

1. **Which features have the strongest correlation with diabetes?**
2. **Are there any surprising relationships in the data?**
4. **Which false predictions (false positives vs false negatives) are more concerning in a healthcare context?**
5. **How might this model perform differently across age groups or socioeconomic strata?**
6. **What additional features would improve prediction accuracy?**
7. **How do lifestyle factors (physical activity, diet) compare to medical factors (blood pressure, BMI) in importance?**

---

*This data dictionary should be your primary reference throughout the project. Refer back to it when interpreting features, analyzing correlations, and explaining model predictions.*