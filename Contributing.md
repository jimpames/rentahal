# Contributing to RENT-A-HAL 🤖

Welcome to the RENT-A-HAL community! MTOR (Multi-Tronic Operating Realm) is an open-source, speech-first AI operating system — and we need builders like you to help shape it.

> **"Computer... let the realm begin."**

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [How to Submit a Pull Request](#how-to-submit-a-pull-request)
- [What We Need Help With](#what-we-need-help-with)
- [AI Worker Contributions](#ai-worker-contributions)
- [Coding Standards](#coding-standards)
- [Reporting Bugs](#reporting-bugs)
- [Community](#community)

---

## Code of Conduct

RENT-A-HAL is built for the people, by the people. Be respectful, constructive, and collaborative. We're all here to build something great.

---

## Ways to Contribute

You don't have to write code to contribute. Here's how you can help:

- 🐛 **Report bugs** via GitHub Issues
- 💡 **Suggest features** via GitHub Discussions
- 🔧 **Submit code** via Pull Requests
- 📖 **Improve documentation**
- 🤖 **Build and share new AI workers** (see below)
- 🧪 **Test on new hardware** (Win11, RTX cards, iPhone)
- 🌍 **Spread the word** — X, Discord, Reddit

---

## Getting Started

### 1. Fork the repo

Click **Fork** at the top right of [github.com/jimpames/rentahal](https://github.com/jimpames/rentahal).

### 2. Clone your fork

```bash
git clone https://github.com/YOUR_USERNAME/rentahal.git
cd rentahal
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run locally

```bash
python webgui.py
```

Then open your browser to `http://localhost:5000` and say: **"Computer..."**

### 5. Create a branch for your changes

```bash
git checkout -b feature/your-feature-name
```

---

## How to Submit a Pull Request

1. Make your changes on your branch
2. Test that everything still works locally
3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Open a **Pull Request** against `jimpames/rentahal:main`
5. Fill out the PR description — what does it do, why, any screenshots or test notes

Jim ([@jimpames](https://github.com/jimpames)) reviews all PRs. Be patient and responsive to feedback.

---

## What We Need Help With

Check the [Issues tab](https://github.com/jimpames/rentahal/issues) for open tasks. Look especially for:

- `good first issue` — great if you're new to the project
- `bug` — confirmed bugs that need fixing
- `enhancement` — new features or improvements
- `documentation` — writing and clarity improvements
- `ai-worker` — new AI worker integrations

Don't see an issue for what you want to build? Open one first and discuss before writing a big PR.

---

## AI Worker Contributions

One of the most impactful ways to contribute is building a new AI worker node. Workers are modular and self-registering — they plug into the MTOR broker and announce their capabilities.

A worker needs to:
- Expose a FastAPI interface
- Report health status (model, uptime, latency, VRAM)
- Accept and return structured JSON messages
- Register itself with the broker on startup

See `llama fast api AI worker` and `Llava AI worker` in the repo for reference implementations. New worker ideas:
- Whisper (local STT)
- TTS workers (Coqui, Piper)
- Code execution workers
- RAG/document workers
- Custom Python agents

---

## Coding Standards

- **Python**: Follow PEP8. Use descriptive variable names.
- **JavaScript**: ES6+, keep it clean and commented.
- **Comments**: Comment your intent, not just what the code does.
- **No breaking changes** to the broker API without discussion first.
- Keep PRs focused — one feature or fix per PR.

---

## Reporting Bugs

Open a [GitHub Issue](https://github.com/jimpames/rentahal/issues/new) and include:

- What you were doing
- What you expected to happen
- What actually happened
- Your OS, GPU, Python version
- Any relevant logs from the debug console

---

## Community

- 🌐 **Live demo**: [rentahal.com](https://rentahal.com)
- 🐦 **X / Twitter**: [@rentahal](https://x.com/rentahal)
- 💼 **LinkedIn**: [Jim Ames](https://www.linkedin.com/in/jimpames/)
- 🪙 **Token ($9000)**: [pump.fun](https://pump.fun/coin/3eazihmAw8yNHhgoNaNr8aBGaBaoLcwZVCBDPnrSpump)

GitHub Discussions is open — use it for questions, ideas, and showing off what you've built on MTOR.

---

## License

RENT-A-HAL is released under **GPLv3**. All contributions must be compatible with this license. No closed-source forks. No patents. The realm stays open — forever.

---

*Built with ❤️ by Jim Ames and the MTOR community.*
