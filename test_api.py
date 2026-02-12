import requests

url = "http://127.0.0.1:5000/predict"

data = {
  "health_service_area": "Western NY",
  "hospital_county": "Allegany",
  "age_group": "70 or Older",
  "zip_code_3_digits": "148",
  "gender": "F",
  "race": "White",
  "ethnicity": "Not Span/Hispanic",
  "type_of_admission": "Elective",
  "discharge_year": 2016,
  "ccs_diagnosis_code": 203,
  "ccs_diagnosis_description": "Osteoarthritis",
  "ccs_procedure_code": 153,
  "ccs_procedure_description": "HIP REPLACEMENT,TOT/PRT",
  "apr_drg_code": 301,
  "apr_drg_description": "Hip joint replacement",
  "apr_mdc_code": 8,
  "apr_mdc_description": "Diseases and Disorders of the Musculoskeletal System and Conn Tissue",
  "apr_severity_of_illness_code": 2,
  "apr_severity_of_illness_description": "Moderate",
  "apr_risk_of_mortality": "Minor",
  "apr_medical_surgical_description": "Surgical",
  "total_charges": 35681.75
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())
