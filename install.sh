#!/bin/bash

# DataFoundry Installation Script
# Sets up the entire project: backend, frontend, and dependencies

set -e

PROJECT_ROOT="/home/reyan/Documents/HackUTDNVIDIA"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "DataFoundry - Installation"
echo "=================================================="

# Check Python version
echo ""
echo "🐍 Checking Python version..."
python3 --version

# Check Node version
echo ""
echo "📦 Checking Node version..."
node --version
npm --version

# Create virtual environment
echo ""
echo "🔧 Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Install backend dependencies
echo ""
echo "📦 Installing backend dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt
echo "   ✅ Backend dependencies installed"

# Install frontend dependencies
echo ""
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
echo "   ✅ Frontend dependencies installed"
cd ..

# Create logs directory
echo ""
echo "📁 Creating logs directory..."
mkdir -p logs
echo "   ✅ Logs directory created"

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo ""
    echo "   For Agent Mode to work, you need an NVIDIA API key."
    echo ""
    echo "   Steps:"
    echo "   1. Go to https://build.nvidia.com/"
    echo "   2. Sign up/login and generate an API key"
    echo "   3. Create a .env file:"
    echo ""
    echo "      nano .env"
    echo ""
    echo "   4. Add this line:"
    echo ""
    echo "      NVIDIA_API_KEY=nvapi-your-key-here"
    echo ""
    echo "   5. Save and exit (Ctrl+X, Y, Enter)"
    echo ""
else
    echo "✅ .env file found"
fi

echo ""
echo "=================================================="
echo "✅ Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. If you haven't already, set up your .env file (see above)"
echo "  2. Start all services: ./start.sh"
echo "  3. Open http://localhost:5173 in your browser"
echo ""
echo "Useful commands:"
echo "  ./start.sh  - Start all services"
echo "  ./stop.sh   - Stop all services"
echo ""
echo "=================================================="

