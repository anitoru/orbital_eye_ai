---
title: ORBITAL EYE AI
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: OCT retinal monitoring prototype for astronaut eye health
---

# ORBITAL EYE AI — public demo

AI-assisted OCT monitoring prototype for astronaut eye health (NASA Space Apps Challenge 2026).

**Research prototype. Not a medical device. Does not diagnose Spaceflight-Associated Neuro-ocular Syndrome (SANS).**

- Fictional astronauts with **synthetic** OCT-like scans demonstrate baseline → in-flight → post-flight monitoring.
- Uploads are analysed for the visitor only, not saved, and deleted automatically.
- Retinal layer U-Net trained on the public Duke DME OCT dataset (Chiu et al., Biomed. Opt. Express 2015) — not on astronaut data.
- Measurements are in pixels; alert thresholds are illustrative only.

Source code: https://github.com/anitoru/orbital_eye_ai
