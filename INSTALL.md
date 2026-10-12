# Installing RENT-A-HAL (MTOR) 🤖

> **"Computer... let the realm begin."**

This guide walks you through setting up RENT-A-HAL on your own machine. You'll be running your own local AI realm with speech input/output, vision, and image generation — no cloud required.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start (Windows)](#quick-start-windows)
- [Quick Start (Linux/Mac)](#quick-start-linuxmac)
- [Installing Ollama](#installing-ollama)
- [Starting the AI Workers](#starting-the-ai-workers)
- [Running the Web GUI](#running-the-web-gui)
- [Verify Your Realm](#verify-your-realm)
- [Optional: Stable Diffusion Worker](#optional-stable-diffusion-worker)
- [Optional: Claude API Worker](#optional-claude-api-worker)
- [Troubleshooting](#troubleshooting)
- [Community & Support](#community--support)

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10, Ubuntu 20.04, macOS 12 | Windows 11, Ubuntu 22.04 |
| RAM | 8GB | 16GB+ |
| GPU | None (CPU mode) | NVIDIA RTX (8GB+ VRAM) |
| Python | 3.10+ | 3.11 |
| Storage | 10GB free | 50GB+ (for models) |
| Browser | Chrome, Edge, Firefox | Chrome (for speech) |

> **Note:** Speech input/output works best in Chrome or Edge. Firefox has limited Web Speech API support.

> **GPU:** MTOR runs in CPU mode without a GPU, but AI responses will be significantly slower. An NVIDIA RTX card with 8GB+ VRAM is strongly recommended for a good experience. See [Supported RTX cards](https://github.com/jimpames/rentahal/blob/main/supported%20RTX).

---

## Quick Start (Windows)

### 1. Clone the repo

```bash
git clone https://github.com/jimpames/rentahal.git
cd rentahal
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Uncomment the torch lines in `requirements.txt` before installing:
> ```
> torch==2.4.1+cu124
> torchaudio==2.4.1+cu124
> torchvision==0.19.1+cu124
> ```
> Then install PyTorch separately:
> ```bash
> pip install torch==2.4.1+cu124 torchaudio==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124
> ```

### 4. Install and start Ollama

See [Installing Ollama](#installing-ollama) below.

### 5. Start the broker

```bash
python webgui.py
```

### 6. Open your browser

Go to: **http://localhost:5000**

Say: **"Computer..."** — and the realm begins. 🚀

---

## Quick Start (Linux/Mac)

### 1. Clone the repo

```bash
git clone https://github.com/jimpames/rentahal.git
cd rentahal
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

See [Installing Ollama](#installing-ollama) below.

### 5. Start the broker

```bash
python webgui.py
```

### 6. Open your browser

Go to: **http://localhost:5000**

---

## Installing Ollama

Ollama runs local LLM models (Llama, Llava) on your machine.

### Install Ollama

**Windows/Mac:** Download from [ollama.com](https://ollama.com/download)

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull a model

```bash
# For chat (Llama 3)
ollama pull llama3

# For vision (Llava)
ollama pull llava
```

### Start Ollama

```bash
ollama serve
```

Ollama runs on port **11434** by default.

---

## Starting the AI Workers

MTOR uses a three-node architecture. Start each worker in a separate terminal.

### Chat Worker (Llama)

```bash
cd rentahal
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
uvicorn main:app_llama --host 0.0.0.0 --port 8000
```

### Vision Worker (Llava)

```bash
uvicorn main:app_llava --host 0.0.0.0 --port 8001
```

> Workers automatically register themselves with the broker on startup. Check the debug console in your browser to confirm they appear as **healthy**.

---

## Running the Web GUI

The web GUI is the human interface — speech in, text out, image generation, vision.

```bash
python webgui.py
```

Then open: **http://localhost:5000**

**Interface features:**
- 🎤 **Speech input** — say "Computer..." to wake it
- 💬 **Text chat** — type queries directly
- 👁️ **Vision** — upload an image for analysis
- 🎨 **Imagine** — generate images via Stable Diffusion
- 📊 **Debug console** — live worker health and queue status

---

## Verify Your Realm

Once everything is running, check the debug console in the browser. You should see:

- ✅ **LLAMA** — healthy, active
- ✅ **LLAVA** — healthy, active (if vision worker started)
- ✅ **Queue** — processing normally
- ✅ **WebSocket** — connected

If workers show as **unhealthy**, see [Troubleshooting](#troubleshooting).

---

## Optional: Stable Diffusion Worker

For image generation ("imagine" mode):

1. Install [Automatic1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
2. Start it with the API flag:
   ```bash
   ./webui.sh --api   # Linux/Mac
   webui-user.bat --api   # Windows
   ```
3. It runs on port **7860** by default
4. MTOR auto-detects it on startup

> Requires a GPU with 6GB+ VRAM for Stable Diffusion 1.5

---

## Optional: Claude API Worker

To use Anthropic's Claude as an AI worker:

1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. Add it to your `.env` file:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```
3. The Claude worker activates automatically when the key is present

---

## Troubleshooting

**Workers show as unhealthy:**
- Make sure Ollama is running: `ollama serve`
- Check the model is pulled: `ollama list`
- Verify ports 8000/8001 aren't blocked by firewall

**Speech input not working:**
- Use Chrome or Edge — Firefox has limited support
- Allow microphone access when prompted
- Check browser console for Web Speech API errors

**Slow responses:**
- CPU mode is slow by design — a GPU dramatically improves speed
- Try a smaller model: `ollama pull llama3:8b`

**Port conflicts:**
- If port 5000 is in use: `python webgui.py --port 5001`
- Check existing processes: `netstat -ano | findstr :5000` (Windows)

**pip install fails:**
- Make sure your virtual environment is activated
- Try upgrading pip: `pip install --upgrade pip`
- Python 3.10+ required: `python --version`

---

## Community & Support

- 🌐 **Live demo**: [rentahal.com](https://rentahal.com)
- 🐦 **X / Twitter**: [@rentahal](https://x.com/rentahal)
- 💼 **LinkedIn**: [Jim Ames](https://www.linkedin.com/in/jimpames/)
- 📖 **500-page theory doc**: [MTOR Theory of Operation (free PDF)](https://github.com/jimpames/RENTAHAL-FOUNDATION/blob/main/MTOR-the-OS.pdf)
- 🎥 **Demo video**: [YouTube](https://youtu.be/k8xWLwzsHZ8?si=O55Z9m6HoUXUb3q2)
- 🪙 **Token ($9000)**: [pump.fun](https://pump.fun/coin/3eazihmAw8yNHhgoNaNr8aBGaBaoLcwZVCBDPnrSpump)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/jimpames/rentahal/discussions)

---

## Easy Installer

Don't want to do this manually? Jim provides a Dropbox easy-install package:

👉 [Download Easy Installer](https://www.dropbox.com/scl/fo/63h0vr5lxmrjpvq4a5fm3/AJuFfb4GfpblX4h4Q6-ek6Q?rlkey=t7n3fytwkwm0ubw8cma95yv7y&st=nywszo8u&dl=0)

---

*Built with ❤️ by Jim Ames and the MTOR community. GPLv3 — forever open.*
