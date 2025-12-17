# 🐍 Snake AI by Lance Jepsen (Neuroevolution in Python) 
### Neuroevolutionary Snake Agent in Python

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![AI](https://img.shields.io/badge/AI-Neuroevolution-purple)
![Game AI](https://img.shields.io/badge/Game%20AI-Snake-green)
![Status](https://img.shields.io/badge/Status-Actively%20Evolving-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

🚀 **Snake AI by Lance Jepsen** is a live-learning Snake game powered by **neuroevolution** — not hard‑coded rules, not pre‑trained models, but an AI that *learns from scratch* by evolving neural networks generation after generation.

This repository contains **two experiences**:
1. 🧠 **An AI that learns to play Snake in real time**
2. 🎮 **A classic Snake game YOU can play yourself**

---

## 🧠 What Makes This Project Special?

Unlike typical Snake AI demos, this project features:

- ✅ Live learning via **neuroevolution**
- ✅ Ray‑based perception (walls, obstacles, tail, food)
- ✅ Short‑term memory to avoid loops
- ✅ Anti‑circle fitness shaping
- ✅ Curriculum learning with moving obstacles
- ✅ Real‑time visualization (Turtle)
- ✅ Save & resume learning anytime

No pretrained weights.  
No shortcuts.  
Just evolution. 🧬

---

## 📂 Repository Structure

```
├── snake_ai.py                  # 🤖 Main Snake AI (neuroevolution + live demo)
├── snake_you_play.py            # 🎮 Classic Snake — YOU play!
├── README.md                    # 📘 You are here
```

---

## 🎮 Play Snake Yourself (No AI)

Run:
```bash
python snake_you_play.py
```

Controls:
- Arrow keys ⬅️⬆️⬇️➡️ to move
- Close the window to quit

Pure arcade Snake fun 🕹️

---

## 🤖 Run the Snake AI (Watch It Learn)

```bash
python live_snake_neuroevo_turtle.py
```

Controls:
- Space → Pause / Resume
- Q → Quit and save progress
- R → Reset learning (delete save file)

Watch the AI:
- Crash
- Adapt
- Learn
- Dominate

---

## 🧬 How the AI Learns

- Each snake is a neural network
- A genetic algorithm selects the best performers
- Networks evolve via mutation and crossover
- Fitness rewards:
  - Eating food 🍎
  - Moving toward food
  - Survival
- Penalties discourage:
  - Circling
  - Oscillation
  - Self‑trapping

---

## 🔍 Keywords

Snake AI · Neuroevolution · Genetic Algorithm · Python · Game AI · Machine Learning · Artificial Intelligence

---

## 👤 Author

**Lance Jepsen**  
Python · AI · Data Science · Automation

If you enjoy this project, ⭐ the repo and share it, that’s how open source grows.
