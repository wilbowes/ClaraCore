# ClaraCore

An open source AI-driven family home presence. ClaraCore uses memory to build a living picture of family life, with integrations into the services that matter — calendars, public transport, home automation, music, and the web.

If Alexa, Siri, or Gemini feel like products rather than presences, this is for you.

---

## What it is

ClaraCore is a conversational AI backend designed to run in your home, not in someone else's cloud. It connects to the services your household actually uses, remembers what matters, and builds context over time rather than treating every interaction as a fresh start.

It speaks to you via Telegram today, and can be paired with [EchoMuse](https://github.com/wilbowes/EchoMuse) for a home-wide voice presence using repurposed Amazon Echo Dot hardware.

---

## Memory architecture

Memory is modelled on human experience — fine-grained for recent events, progressively consolidated as time passes. Conversations are summarised at the end of each session, days are consolidated overnight, and a core memory layer accumulates what is enduringly true about your household. Designed for economic token use.

```
Conversation → session summary → day summary → core memory
```

---

## Integrations

| Integration | Required | Purpose |
|---|---|---|
| Anthropic API | ✅ | Language model (BYOLLM support planned) |
| Telegram | ✅ | Primary chat interface |
| Home Assistant | Optional | Device state, presence, media control |
| Apple Calendar (CalDAV) | Optional | Family calendar awareness |
| Music Assistant | Optional | Music playback control |
| Last.fm | Optional | Music taste graph |
| Brave Search | Optional | Web search and page fetch |
| PTV API | Optional | Melbourne public transport |
| llama.cpp | Optional | Local model support (health checks) |

Everything beyond Anthropic and Telegram is optional — each integration adds to the experience but none are required to get started.

---

## Voice

`voice_server.py` provides an STT/TTS interface using Faster Whisper and Piper. When paired with [EchoMuse](https://github.com/wilbowes/EchoMuse), it enables a home-wide voice presence — wake word detection, voice turns, and spoken responses through repurposed Echo Dot hardware.

---

## Getting started

### Prerequisites

- Docker and Docker Compose
- An [Anthropic API key](https://console.anthropic.com/)
- A Telegram bot token (via [@BotFather](https://t.me/botfather))
- A GPU is recommended for the voice server (Whisper large-v3)

### Setup

```bash
# Clone the repo
git clone https://github.com/wilbowes/ClaraCore.git
cd ClaraCore

# Configure environment
cp .env.example .env
# Edit .env with your API keys and service URLs

# Configure your presence
cp config.json.example config.json
# Edit config.json — model selection, memory intervals, users

# Configure prompts
cp prompts/system_static.txt.example prompts/system_static.txt
# Edit system_static.txt — give your presence a name, describe your household
# Repeat for other prompts as needed

# Start
docker compose up -d
```

### Prompts

The `prompts/` directory contains the templates that shape ClaraCore's personality, memory summarisation, and household awareness. Start with `system_static.txt` — this is where you define who your presence is and who it's talking to.

All prompt files are excluded from git. `.example` versions are provided as starting points.

---

## Architecture

```
Telegram / EchoMuse voice
        ↓
    bot.py (HTTP API :8766)
        ↓
  Claude (Anthropic API)
  + tool calls → integrations
        ↓
   memory pipeline
  (session → day → core)
```

The voice server runs as a separate container and connects to `bot.py` over HTTP.

---

## License

MIT
