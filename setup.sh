#!/bin/bash
echo "🚀 Setting up Cough Detection AI Website..."

# Check for python
if ! command -v python3 &> /dev/null
then
    echo "❌ python3 could not be found. Please install it."
    exit
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies (this may take a minute)..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "👉 To start the server, run: source venv/bin/activate && python app.py"
