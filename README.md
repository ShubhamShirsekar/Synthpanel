# SynthPanel

AI-powered synthetic customer interview workflow for BNPP Personal Finance (Azure Hackathon 2026).

This repository contains:
- A data-to-persona pipeline notebook (feature engineering, clustering, profiling, persona generation)
- A Streamlit app to run synthetic interviews against generated personas
- An automated test script for multi-turn interview validation

## Project Structure

```
.
|-- pipeline_config.json
|-- requirements.txt
|-- Step1_PythonPipeline.ipynb
|-- synthpanel_app.py
|-- test_synthpanel.py
|-- dashboards/
|   `-- synthpanel_heatmap.html
`-- output/
		|-- cluster_profiles.json
		|-- clustered_customers.csv
		|-- test_results.csv
		`-- personas/
				|-- cluster_0_persona.json
				|-- cluster_1_persona.json
				|-- cluster_2_persona.json
				|-- cluster_3_persona.json
				|-- cluster_4_persona.json
				`-- cluster_5_persona.json
```

## What Each File Does

- `Step1_PythonPipeline.ipynb`
	- Loads config from `pipeline_config.json`
	- Preprocesses and engineers features
	- Runs K-Means + HDBSCAN + Hierarchical clustering
	- Creates ensemble labels and cluster profiles
	- Generates persona JSONs with Azure OpenAI
- `synthpanel_app.py`
	- Streamlit app for synthetic customer interviews
	- Uses two-model setup: main chat model + lightweight validation model
	- Applies two validation layers and confidence scoring (GREEN/AMBER/RED)
	- Supports session export and heatmap tab
- `test_synthpanel.py`
	- Automated interview test runner across personas
	- Writes validation outcomes to CSV
- `dashboards/synthpanel_heatmap.html`
	- Embedded heatmap view shown in Streamlit app
- `output/`
	- Stores generated clustering/profile/persona/test artifacts

## Prerequisites

- Python 3.10+ (recommended)
- Azure OpenAI access with deployed models

## Installation

From the `project` folder:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the parent folder of `project` (the code loads `../.env`).

Required:

```env
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
```

Optional:

```env
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_ENDPOINT_NANO=...
AZURE_OPENAI_API_KEY_NANO=...
AZURE_OPENAI_API_VERSION_NANO=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT_NANO=gpt-4.1-nano
```

## Data and Config Notes

- Pipeline settings are in `pipeline_config.json`.
- Current config expects input dataset at `../data/Dataset.csv` relative to `project`.
- Pipeline outputs are configured under `project/output`.

## Run the Pipeline Notebook

Open and run `Step1_PythonPipeline.ipynb` top to bottom.

High-level notebook flow:
1. Load config and dataset
2. Feature engineering + encoding
3. Multi-model clustering and ensemble voting
4. Cluster profiling export
5. Persona generation export via Azure OpenAI

Expected outputs include:
- `output/cluster_profiles.json`
- `output/clustered_customers.csv`
- `output/personas/cluster_*_persona.json`

## Run the Streamlit App

From `project`:

```bash
streamlit run synthpanel_app.py
```

In the app:
1. Select a persona
2. Generate an artificial customer
3. Start interview turns
4. Inspect confidence/validation details
5. Export session JSON when needed

## Run Automated Tests

From `project`:

```bash
python test_synthpanel.py
```

Test output is written to:
- `test_results.csv` (root-level output from the script)

## Validation Logic Summary

The interview flow applies two validation layers:
- Layer 1: persona consistency, factual grounding, product knowledge, emotional register, hallucination checks
- Layer 2: statistical plausibility against cluster profile + INE benchmark alignment

Confidence label rules:
- `GREEN`: no major risk flags
- `AMBER`: soft-risk signals (income/future claims/statistical soft mismatch)
- `RED`: explicit validation failures in Layer 1

## Troubleshooting

- Missing persona files: ensure `output/personas/*.json` exists (run notebook persona generation cell).
- Azure errors: verify endpoint, key, deployment names, and API versions in `.env`.
- Empty heatmap tab: confirm `dashboards/synthpanel_heatmap.html` exists.
