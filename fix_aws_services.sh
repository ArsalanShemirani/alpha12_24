#!/bin/bash
# Fix AWS Alpha12_24 systemd services

echo "🔧 Fixing Alpha12_24 AWS Services"
echo "=================================="

# Stop all services first
echo "Stopping existing services..."
sudo systemctl stop alpha12-tracker.service
sudo systemctl stop alpha12-dashboard.service
sudo systemctl stop alpha12-autosignal.timer

# Get the actual user and paths
ACTUAL_USER=$(whoami)
PROJECT_DIR="/home/$ACTUAL_USER/alpha12_24"
VENV_DIR="$PROJECT_DIR/venv"

echo "Detected user: $ACTUAL_USER"
echo "Project directory: $PROJECT_DIR"
echo "Virtual environment: $VENV_DIR"

# Check if directories exist
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Project directory not found: $PROJECT_DIR"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found: $VENV_DIR"
    echo "Creating virtual environment..."
    cd "$PROJECT_DIR"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Verify Python executable
PYTHON_EXEC="$VENV_DIR/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "❌ Python executable not found: $PYTHON_EXEC"
    exit 1
fi

echo "✅ Python executable found: $PYTHON_EXEC"

# Create corrected service files
echo "Creating corrected service files..."

# Tracker Service
sudo tee /etc/systemd/system/alpha12-tracker.service > /dev/null << EOF
[Unit]
Description=Alpha12_24 Tracker Daemon
After=network.target
Wants=network.target

[Service]
Type=simple
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV_DIR/bin
Environment=PYTHONPATH=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON_EXEC -m src.daemon.tracker
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$PROJECT_DIR/runs $PROJECT_DIR/logs

[Install]
WantedBy=multi-user.target
EOF

# Dashboard Service
sudo tee /etc/systemd/system/alpha12-dashboard.service > /dev/null << EOF
[Unit]
Description=Alpha12_24 Dashboard
After=network.target
Wants=network.target

[Service]
Type=simple
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV_DIR/bin
Environment=PYTHONPATH=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV_DIR/bin/streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$PROJECT_DIR/runs $PROJECT_DIR/logs

[Install]
WantedBy=multi-user.target
EOF

# Autosignal Service
sudo tee /etc/systemd/system/alpha12-autosignal.service > /dev/null << EOF
[Unit]
Description=Alpha12_24 Autosignal Daemon
After=network.target
Wants=network.target

[Service]
Type=oneshot
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV_DIR/bin
Environment=PYTHONPATH=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON_EXEC -m src.daemon.autosignal
StandardOutput=journal
StandardError=journal
Restart=no

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$PROJECT_DIR/runs $PROJECT_DIR/logs

[Install]
WantedBy=multi-user.target
EOF

# Autosignal Timer
sudo tee /etc/systemd/system/alpha12-autosignal.timer > /dev/null << EOF
[Unit]
Description=Run Alpha12_24 Autosignal every hour
Requires=alpha12-autosignal.service

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Set proper permissions
echo "Setting proper permissions..."
sudo chown -R $ACTUAL_USER:$ACTUAL_USER $PROJECT_DIR
sudo chmod -R 755 $PROJECT_DIR

# Create required directories
echo "Creating required directories..."
mkdir -p $PROJECT_DIR/runs
mkdir -p $PROJECT_DIR/artifacts
mkdir -p $PROJECT_DIR/logs

# Set directory permissions
chmod 755 $PROJECT_DIR/runs
chmod 755 $PROJECT_DIR/artifacts
chmod 755 $PROJECT_DIR/logs

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload

# Enable services
echo "Enabling services..."
sudo systemctl enable alpha12-tracker.service
sudo systemctl enable alpha12-dashboard.service
sudo systemctl enable alpha12-autosignal.timer

# Start services
echo "Starting services..."
sudo systemctl start alpha12-tracker.service
sudo systemctl start alpha12-dashboard.service
sudo systemctl start alpha12-autosignal.timer

# Check status
echo ""
echo "🔍 Service Status:"
echo "=================="
echo "Tracker Service:"
sudo systemctl status alpha12-tracker.service --no-pager -l
echo ""
echo "Dashboard Service:"
sudo systemctl status alpha12-dashboard.service --no-pager -l
echo ""
echo "Autosignal Timer:"
sudo systemctl status alpha12-autosignal.timer --no-pager -l

echo ""
echo "📋 Next Steps:"
echo "=============="
echo "1. Check logs: sudo journalctl -u alpha12-tracker.service -f"
echo "2. Test dashboard: curl http://localhost:8501"
echo "3. Check tracker heartbeat: cat $PROJECT_DIR/runs/daemon_heartbeat.txt"
echo ""
echo "✅ Service fix completed!"
