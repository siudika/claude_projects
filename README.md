# 🤖 Claude Chat

A clean, secure chat interface for use with Anthropic API keys.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

---

## ✨ Features

- **💬 Chat with Claude** - Multi-model support (Sonnet, Opus, Haiku)
- **🔐 Encrypted storage** - Conversations encrypted locally using Fernet
- **📁 Anthropics Files API** - Utilize Anthropics Files API in your conversations
- **💾 Thread management** - Save and organize chat history
- **📊 Usage tracking** - Monitor token usage and API costs
- **📱 Mobile friendly** - Responsive design works on desktop and mobile

---

## 🚀 Quick Start

### 1. Clone & Install

git clone https://github.com/siudika/claude_chat.git
cd claude_chat
pip install -r requirements.txt

### 2. Run the App

streamlit run claude_gui.py

The app will open at [http://localhost:8501](http://localhost:8501)

### 3. First Launch Setup

On first launch, you'll see a setup screen:

1. Click the link to [Anthropic Console](https://console.anthropic.com/account/keys)
2. Copy your API key
3. Paste it in the app
4. Click **"Create .env & Launch"**

That's it! The app automatically:

- ✅ Generates an encryption key
- ✅ Creates your `.env` file
- ✅ Starts the chat

---

## 📖 Usage

| Action | How To |
|--------|--------|
| **New chat** | Click "➕ New Chat" in sidebar |
| **Attach files** | Check "📎 Attach Files", upload documents, select before sending |
| **Switch chats** | Click thread names in sidebar |
| **Change model** | Use dropdown in sidebar |
| **View usage** | Expand "📊 Usage (24h)" in sidebar |


---

## 🔒 Security

- 🔐 Conversations encrypted at rest using **Fernet (AES-128)**
- 🚫 `.env` file never committed to git
- 🔑 API key stored only locally
- 💾 All data stays on your computer

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| **"API key invalid"** | Ensure key starts with `sk-ant-` from [Anthropic Console](https://console.anthropic.com/account/keys) |
| **"CLAUDE_CHAT_KEY not found"** | Delete `.env` and restart app to regenerate |
| **"Can't decrypt old chats"** | Encryption key changed. Keep your `.env` backed up |
| **"Module not found"** | Run `pip install -r requirements.txt` |

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io/) - Web framework
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) - Claude API client
- [Cryptography](https://cryptography.io/) - Fernet encryption
- [streamlit-extras](https://github.com/arnaudmiribel/streamlit-extras) - Enhanced UI components
- [streamlit-option-menu](https://github.com/victoryhb/streamlit-option-menu) - Sidebar navigation

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- Built with [Claude](https://claude.ai) by Anthropic
- Inspired by the need for a simple, secure local chat interface

---

**Made with ❤️ for secure AI conversations**
