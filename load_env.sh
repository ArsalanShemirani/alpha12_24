#!/bin/bash

# Load environment variables for alpha12_24
# Usage: source load_env.sh

# Check if .env file exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    # Load and expand variables
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ $line =~ ^[[:space:]]*# ]] && continue
        [[ -z $line ]] && continue
        
        # Export the variable, expanding $(pwd) if present
        if [[ $line == *'$(pwd)'* ]]; then
            expanded_line=$(echo "$line" | sed "s|\$(pwd)|$(pwd)|g")
            export "$expanded_line"
        else
            export "$line"
        fi
    done < .env
    echo "✅ Environment variables loaded successfully!"
else
    echo "⚠️  .env file not found. Creating from template..."
    
    # Create .env file from template
    if [ -f "env_template.txt" ]; then
        cp env_template.txt .env
        echo "✅ .env file created from template!"
        echo "Loading environment variables..."
        # Load and expand variables
        while IFS= read -r line; do
            # Skip comments and empty lines
            [[ $line =~ ^[[:space:]]*# ]] && continue
            [[ -z $line ]] && continue
            
            # Export the variable, expanding $(pwd) if present
            if [[ $line == *'$(pwd)'* ]]; then
                expanded_line=$(echo "$line" | sed "s|\$(pwd)|$(pwd)|g")
                export "$expanded_line"
            else
                export "$line"
            fi
        done < .env
        echo "✅ Environment variables loaded successfully!"
    else
        echo "❌ Neither .env nor env_template.txt found!"
        echo "Please create a .env file with your Telegram bot credentials."
        exit 1
    fi
fi

# Verify Telegram credentials are loaded
if [ -n "$TG_BOT_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
    echo "✅ Telegram bot credentials loaded:"
    echo "   Bot Token: ${TG_BOT_TOKEN:0:10}..."
    echo "   Chat ID: $TG_CHAT_ID"
else
    echo "❌ Telegram bot credentials not found!"
    echo "Please check your .env file."
fi

echo ""
echo "🚀 You can now run the dashboard or tracker with Telegram alerts enabled!"
echo "   Dashboard: streamlit run src/dashboard/app.py"
echo "   Tracker: python src/daemon/tracker.py"
