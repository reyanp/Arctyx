# DataFoundry - Synthetic Data Generation Platform

A full-stack application for generating synthetic data using weak supervision, generative models, and AI agents.

## 🚀 Quick Start

### One-Time Setup

```bash
# 1. Run installation script
./install.sh

# 2. Create .env file with your NVIDIA API key
nano .env
# Add: NVIDIA_API_KEY=nvapi-your-key-here
```

### Running the Application

```bash
# Start all services (Frontend + Backend + Agent)
./start.sh

# Open in browser: http://localhost:8080
```

### Stopping the Application

```bash
# Stop all services
./stop.sh
```

## 📋 Manual Setup (Alternative)

If you prefer to set up manually:

### Backend Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Create .env file
echo "NVIDIA_API_KEY=your_key_here" > .env

# Terminal 1: Start Flask API (Port 5000)
python backend/flask_api.py --port 5000

# Terminal 2: Start Orchestrator (Port 8000)
python backend/agents/serve_orchestrator.py
```

### Frontend Setup

```bash
# Terminal 3: Start Frontend (Port 8080)
cd frontend
npm install
npm run dev
```

## 🏗️ Architecture

### Services

- **Frontend** (Port 8080): React + TypeScript + Vite
- **Flask API** (Port 5000): Direct tool access for labeling, training, generation
- **Orchestrator** (Port 8000): AI agent for natural language workflows

### Tech Stack

**Backend:**
- Flask - RESTful API
- FastAPI - Agent orchestrator
- PyTorch - Deep learning models
- Snorkel - Weak supervision labeling
- LangChain + LangGraph - Agent orchestration
- NVIDIA NIMs - LLM services

**Frontend:**
- React + TypeScript
- Vite - Build tool
- TanStack Query - Data fetching
- Shadcn UI - Component library
- Tailwind CSS - Styling

## 📖 Documentation

- **Backend API**: See `backend/FLASK_API_README.md`
- **Agent Setup**: See `backend/AGENT_SETUP.md`
- **Quick Reference**: See `backend/API_QUICK_REFERENCE.md`

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
# Required for Agent Mode
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxx
```

Get your API key from [NVIDIA Build](https://build.nvidia.com/)

## 🎯 Features

### Manual Mode (Direct Control)
- **Upload Dataset**: CSV or Parquet files
- **Data Labeling**: Weak supervision with Snorkel
- **Model Training**: CVAE, CTGAN, VAE+GMM models
- **Data Generation**: Generate synthetic samples
- **Export**: Download as Parquet or CSV

### Agent Mode (AI-Powered)
- **Natural Language**: Describe your goal in plain English
- **Automated Workflow**: Agent orchestrates all steps
- **Self-Correcting**: Adjusts hyperparameters automatically
- **Multi-Step Tasks**: Handles complex workflows

## 📁 Project Structure

```
HackUTDNVIDIA/
├── backend/
│   ├── agents/                 # AI agent pipelines
│   ├── DataFoundry/           # Core data processing
│   ├── testing_data/          # Sample datasets
│   ├── output_data/           # Generated outputs
│   ├── flask_api.py           # Main API server
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/            # Page components
│   │   ├── lib/              # API client & utilities
│   │   └── hooks/            # Custom React hooks
│   └── package.json          # Node dependencies
├── logs/                      # Service logs
├── install.sh                # Installation script
├── start.sh                  # Start all services
├── stop.sh                   # Stop all services
└── .env                      # Environment variables (create this)
```

## 🛠️ Development

### Logs

View logs for each service:

```bash
# Flask API
tail -f logs/flask.log

# Orchestrator
tail -f logs/orchestrator.log

# Frontend
tail -f logs/frontend.log
```

### Testing

```bash
# Activate venv
source venv/bin/activate

# Test backend API
cd backend
python -m pytest

# Test agents
python -m agents.test_agents --test all
```

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Check what's using a port
lsof -ti:5000  # Flask
lsof -ti:8000  # Orchestrator
lsof -ti:8080  # Frontend

# Kill process on port
lsof -ti:5000 | xargs kill -9
```

### Services Won't Start

1. Make sure virtual environment is activated
2. Check all dependencies are installed
3. Verify `.env` file exists with API key
4. Check logs for specific errors

### Frontend Can't Connect to Backend

1. Verify Flask is running on port 5000
2. Check CORS is enabled in Flask
3. Verify `VITE_BACKEND_URL` in `frontend/.env` is set to `http://localhost:5000`

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

## 🙏 Acknowledgments

- NVIDIA for NIM services
- Snorkel AI for weak supervision
- LangChain for agent framework

