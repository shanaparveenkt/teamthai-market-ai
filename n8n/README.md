# n8n Automation

n8n is used to automate the Team Thai market intelligence pipeline.

## Intended workflow

Schedule
↓
Collect / update data
↓
Run Python data cleaning
↓
Run market analysis
↓
Run AI insight engine
↓
Save AI report
↓
Notify / distribute report

## Important

n8n is the automation layer.

Python performs data processing.

The LLM performs interpretation and recommendation generation.

Streamlit presents the results.