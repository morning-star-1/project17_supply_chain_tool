# Architecture

## Overview
- Streamlit UI for inputs and charts
- Forecasting, inventory, and simulation helpers in one file

## Data flow
CSV input -> forecast -> inventory policy -> simulation -> KPIs

## Key decisions
- Keep logic in a single script for clarity
- Use simple models for explainability
