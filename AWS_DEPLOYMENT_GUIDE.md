# 🚀 AWS DEPLOYMENT GUIDE
## Alpha12_24 Trading System - Complete Production Setup

### 📋 **SYSTEM OVERVIEW**

**Alpha12_24** is a comprehensive cryptocurrency trading system with:
- ✅ **Password-protected dashboard** (Streamlit)
- ✅ **24/7 automated trading** (Autosignal + Tracker daemons)
- ✅ **Enhanced confidence weighting** (Sentiment + Max Pain)
- ✅ **Real-time alerts** (Telegram integration)
- ✅ **Background training** (Continuous model improvement)
- ✅ **Multi-asset support** (BTC, ETH with short/long positions)

---

## 🏗️ **ARCHITECTURE**

```
AWS EC2 Instance (t3.medium or larger)
├── System Services (systemd)
│   ├── alpha12-autosignal.service    # Setup generation (1h intervals)
│   ├── alpha12-tracker.service       # 24/7 trade monitoring
│   └── alpha12-dashboard.service     # Web dashboard (port 8501)
├── Python Environment
│   ├── Virtual Environment (venv)
│   ├── Dependencies (requirements.txt)
│   └── Configuration (.env)
└── Data Storage
    ├── runs/ (trade data, setups, history)
    ├── artifacts/ (ML models)
    └── logs/ (system logs)
```

---

## 🚀 **STEP 1: AWS EC2 INSTANCE SETUP**

### 1.1 Launch EC2 Instance

```bash
# Recommended specifications
Instance Type: t3.medium (2 vCPU, 4 GB RAM)
Storage: 20 GB GP3 SSD
OS: Ubuntu 22.04 LTS
Security Group: 
  - SSH (22) - Your IP only
  - HTTP (80) - Optional
  - Custom (8501) - Dashboard access
```

### 1.2 Connect and Update System

```bash
# Connect to your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git curl wget htop
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
sudo apt install -y nginx supervisor

# Install TA-Lib dependencies (for advanced indicators)
sudo apt install -y libta-lib0 libta-lib-dev
```

---

## 🐍 **STEP 2: PYTHON ENVIRONMENT SETUP**

### 2.1 Clone Repository

```bash
# Clone your repository
git clone https://github.com/your-username/alpha12_24.git
cd alpha12_24

# Set proper permissions
chmod +x *.sh
```

### 2.2 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for production
pip install gunicorn supervisor
```

### 2.3 Install TA-Lib (Optional but Recommended)

```bash
# Install TA-Lib for advanced technical indicators
pip install TA-Lib

# If TA-Lib fails, use the simplified version (already included)
echo "TA-Lib installation completed or using simplified version"
```

---

## ⚙️ **STEP 3: CONFIGURATION SETUP**

### 3.1 Environment Configuration

```bash
# Create .env file with all settings
cat > .env << 'EOF'
# Dashboard Authentication
DASH_AUTH_ENABLED=1
DASH_USERNAME=admin
DASH_PASSWORD_HASH=sha256:4676eabfaf1fa14dd4d069590846d8adeb658c68dfe8d6f9cfe2fbbfee07228f

# Telegram Bot Configuration
TG_BOT_TOKEN=8119234697:AAE7dGn707CEXZo0hyzSHpzCkQtIklEhDkE
TG_CHAT_ID=-4873132631

# Alternative environment variable names (for compatibility)
TG_BOT=8119234697:AAE7dGn707CEXZo0hyzSHpzCkQtIklEhDkE
TG_CHAT=-4873132631

# Alpha12_24 Configuration
ALPHA12_SYMBOL=BTCUSDT
ALPHA12_INTERVAL=1h
ALPHA12_SLEEP=15
PYTHONPATH=$(pwd)

# Scheduling & data
ALPHA12_AUTOSIGNAL_INTERVAL=1h
ALPHA12_SOURCE="Composite (Binance Spot + Bybit derivs)"

# Caps
MAX_SETUPS_PER_DAY=2

# Gates
GATE_REGIME=1
GATE_OB=1
GATE_RR25=1
RR25_THRESH=0.00
OB_EDGE_DELTA=0.30

# Model arming threshold
MIN_CONF_ARM=0.60

# Confidence-based overrides
REGIME_OVERRIDE_CONF=0.72
OB_NEUTRAL_CONF=0.65

# Risk/sizing
ACCOUNT_BALANCE_USD=400
MAX_LEVERAGE=10
RISK_PER_TRADE_PCT=1.0
MIN_RR=1.8
K_ENTRY_ATR=0.25
K_STOP_ATR=1.0
VALID_BARS=24
ENTRY_BUFFER_BPS=5.0
TRIGGER_RULE="touch"

# Background Analysis Configuration
ALPHA12_TRAINING_INTERVAL=15m
ALPHA12_ANALYSIS_SLEEP=300
ALPHA12_ANALYSIS_DAYS=120
EOF
```

### 3.2 Create Required Directories

```bash
# Create necessary directories
mkdir -p runs artifacts logs

# Set proper permissions
chmod 755 runs artifacts logs
```

---

## 🔧 **STEP 4: SYSTEMD SERVICE SETUP**

### 4.1 Create Service Files

```bash
# Create autosignal service
sudo tee /etc/systemd/system/alpha12-autosignal.service > /dev/null << 'EOF'
[Unit]
Description=Alpha12_24 Autosignal Daemon
After=network.target
Wants=network.target

[Service]
Type=oneshot
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/alpha12_24
Environment=PATH=/home/ubuntu/alpha12_24/venv/bin
Environment=PYTHONPATH=/home/ubuntu/alpha12_24
ExecStart=/home/ubuntu/alpha12_24/venv/bin/python -m src.daemon.autosignal
StandardOutput=journal
StandardError=journal
Restart=no

[Install]
WantedBy=multi-user.target
EOF

# Create autosignal timer
sudo tee /etc/systemd/system/alpha12-autosignal.timer > /dev/null << 'EOF'
[Unit]
Description=Run Alpha12_24 Autosignal every hour
Requires=alpha12-autosignal.service

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Create tracker service
sudo tee /etc/systemd/system/alpha12-tracker.service > /dev/null << 'EOF'
[Unit]
Description=Alpha12_24 Tracker Daemon
After=network.target
Wants=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/alpha12_24
Environment=PATH=/home/ubuntu/alpha12_24/venv/bin
Environment=PYTHONPATH=/home/ubuntu/alpha12_24
ExecStart=/home/ubuntu/alpha12_24/venv/bin/python -m src.daemon.tracker
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create dashboard service
sudo tee /etc/systemd/system/alpha12-dashboard.service > /dev/null << 'EOF'
[Unit]
Description=Alpha12_24 Dashboard
After=network.target
Wants=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/alpha12_24
Environment=PATH=/home/ubuntu/alpha12_24/venv/bin
Environment=PYTHONPATH=/home/ubuntu/alpha12_24
ExecStart=/home/ubuntu/alpha12_24/venv/bin/streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

### 4.2 Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable alpha12-autosignal.timer
sudo systemctl enable alpha12-tracker.service
sudo systemctl enable alpha12-dashboard.service

# Start services
sudo systemctl start alpha12-autosignal.timer
sudo systemctl start alpha12-tracker.service
sudo systemctl start alpha12-dashboard.service

# Check status
sudo systemctl status alpha12-autosignal.timer
sudo systemctl status alpha12-tracker.service
sudo systemctl status alpha12-dashboard.service
```

---

## 🌐 **STEP 5: NGINX REVERSE PROXY (OPTIONAL)**

### 5.1 Configure Nginx

```bash
# Create nginx configuration
sudo tee /etc/nginx/sites-available/alpha12-dashboard > /dev/null << 'EOF'
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/alpha12-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🔍 **STEP 6: MONITORING & LOGS**

### 6.1 Create Monitoring Scripts

```bash
# Create monitoring script
cat > monitor_services.sh << 'EOF'
#!/bin/bash
echo "=== Alpha12_24 Service Status ==="
echo "Autosignal Timer:"
sudo systemctl status alpha12-autosignal.timer --no-pager -l
echo ""
echo "Tracker Service:"
sudo systemctl status alpha12-tracker.service --no-pager -l
echo ""
echo "Dashboard Service:"
sudo systemctl status alpha12-dashboard.service --no-pager -l
echo ""
echo "=== Recent Logs ==="
echo "Tracker Logs (last 10 lines):"
sudo journalctl -u alpha12-tracker.service -n 10 --no-pager
echo ""
echo "Dashboard Logs (last 10 lines):"
sudo journalctl -u alpha12-dashboard.service -n 10 --no-pager
EOF

chmod +x monitor_services.sh

# Create log rotation
sudo tee /etc/logrotate.d/alpha12 << 'EOF'
/home/ubuntu/alpha12_24/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
}
EOF
```

### 6.2 View Logs

```bash
# View service logs
sudo journalctl -u alpha12-tracker.service -f
sudo journalctl -u alpha12-dashboard.service -f

# View application logs
tail -f logs/alpha_tracker.log
tail -f logs/alpha_autosignal.log
```

---

## 🧪 **STEP 7: SYSTEM TESTING**

### 7.1 Test All Components

```bash
# Activate virtual environment
source venv/bin/activate

# Test core components
python -c "
import sys
sys.path.append('.')
from src.core.config import config
from src.features.engine import FeatureEngine
from src.data.deribit_free import DeribitFreeProvider
from src.data.real_sentiment import get_current_sentiment
from src.dashboard.auth import login_gate
print('✅ All core components loaded successfully')
"

# Test data providers
python -c "
from src.data.binance_free import BinanceFreeProvider
from src.data.bybit_free import BybitFreeProvider
print('✅ Data providers working')
"

# Test dashboard access
curl -I http://localhost:8501
```

### 7.2 Verify Services

```bash
# Check all services are running
./monitor_services.sh

# Check dashboard is accessible
curl -s http://localhost:8501 | head -5

# Check tracker heartbeat
cat runs/daemon_heartbeat.txt
```

---

## 🔐 **STEP 8: SECURITY HARDENING**

### 8.1 Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw allow ssh
sudo ufw allow 8501  # Dashboard port
sudo ufw allow 80    # HTTP (if using nginx)
sudo ufw enable

# Check firewall status
sudo ufw status
```

### 8.2 SSL Certificate (Optional)

```bash
# Install Certbot for SSL
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## 📊 **STEP 9: DASHBOARD ACCESS**

### 9.1 Access Information

```
Dashboard URL: http://your-instance-ip:8501
Username: admin
Password: a.M1.13?!
```

### 9.2 Dashboard Features

- **Real-time Trading**: Monitor active setups and trades
- **Performance Analytics**: View win rates, P&L, and metrics
- **Configuration**: Adjust trading parameters
- **Backtesting**: Test strategies on historical data
- **Live Metrics**: Real-time system status

---

## 🔄 **STEP 10: MAINTENANCE & UPDATES**

### 10.1 Update System

```bash
# Update code
cd /home/ubuntu/alpha12_24
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Restart services
sudo systemctl restart alpha12-tracker.service
sudo systemctl restart alpha12-dashboard.service
```

### 10.2 Backup Data

```bash
# Create backup script
cat > backup_data.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/ubuntu/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup important data
cp -r runs/ $BACKUP_DIR/
cp -r artifacts/ $BACKUP_DIR/
cp .env $BACKUP_DIR/

echo "Backup created: $BACKUP_DIR"
EOF

chmod +x backup_data.sh

# Add to crontab for daily backups
crontab -e
# Add: 0 2 * * * /home/ubuntu/alpha12_24/backup_data.sh
```

---

## 🚨 **TROUBLESHOOTING**

### Common Issues

1. **Service Not Starting**
   ```bash
   sudo systemctl status alpha12-tracker.service
   sudo journalctl -u alpha12-tracker.service -n 50
   ```

2. **Dashboard Not Accessible**
   ```bash
   # Check if service is running
   sudo systemctl status alpha12-dashboard.service
   
   # Check if port is open
   netstat -tlnp | grep 8501
   ```

3. **Permission Issues**
   ```bash
   # Fix ownership
   sudo chown -R ubuntu:ubuntu /home/ubuntu/alpha12_24
   ```

4. **Memory Issues**
   ```bash
   # Monitor system resources
   htop
   free -h
   df -h
   ```

---

## 📈 **PERFORMANCE MONITORING**

### System Metrics

```bash
# Monitor system performance
htop
iotop
nethogs

# Monitor disk usage
df -h
du -sh /home/ubuntu/alpha12_24/runs/

# Monitor logs
tail -f logs/alpha_tracker.log
```

### Trading Metrics

- **Win Rate**: Target ≥60%
- **Setup Generation**: ~2 setups/day (configurable)
- **Response Time**: <15 seconds for price updates
- **Uptime**: 99.9% target

---

## 🎯 **FINAL VERIFICATION CHECKLIST**

- [ ] All systemd services are running
- [ ] Dashboard is accessible at http://your-ip:8501
- [ ] Login with admin/a.M1.13?! works
- [ ] Tracker heartbeat is updating
- [ ] Telegram alerts are working
- [ ] Setup generation is working
- [ ] Trade monitoring is active
- [ ] Logs are being written
- [ ] Firewall is configured
- [ ] Backups are scheduled

---

## 🚀 **SYSTEM IS NOW READY FOR PRODUCTION!**

Your Alpha12_24 trading system is now fully deployed on AWS with:
- ✅ **24/7 automated trading**
- ✅ **Password-protected dashboard**
- ✅ **Enhanced confidence weighting**
- ✅ **Real-time alerts**
- ✅ **Background training**
- ✅ **System monitoring**

**Next Steps:**
1. Monitor the system for 24-48 hours
2. Verify all components are working correctly
3. Adjust trading parameters as needed
4. Set up additional monitoring if required

**Support:**
- Check logs: `sudo journalctl -u alpha12-*`
- Monitor services: `./monitor_services.sh`
- Dashboard: http://your-ip:8501

🎉 **Happy Trading!** 🎉
