# Systemd Service Templates for alpha12_24

## UI Service Template
```ini
[Unit]
Description=Alpha12_24 Dashboard UI
After=network.target

[Service]
Type=simple
User=alpha12
WorkingDirectory=/path/to/alpha12_24
Environment=PYTHONUNBUFFERED=1
ExecStart=/path/to/alpha12_24/.venv/bin/streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=5s
MemoryMax=400M
StandardOutput=append:/var/log/alpha12/ui.out.log
StandardError=append:/var/log/alpha12/ui.err.log

[Install]
WantedBy=multi-user.target
```

## Tracker Service Template
```ini
[Unit]
Description=Alpha12_24 Trade Tracker
After=network.target

[Service]
Type=simple
User=alpha12
WorkingDirectory=/path/to/alpha12_24
Environment=PYTHONUNBUFFERED=1
ExecStart=/path/to/alpha12_24/.venv/bin/python src/daemon/tracker.py
Restart=always
RestartSec=5s
MemoryMax=400M
StandardOutput=append:/var/log/alpha12/tracker.out.log
StandardError=append:/var/log/alpha12/tracker.err.log

[Install]
WantedBy=multi-user.target
```

## Autosignal Service Template
```ini
[Unit]
Description=Alpha12_24 Autosignal Generator
After=network.target

[Service]
Type=oneshot
User=alpha12
WorkingDirectory=/path/to/alpha12_24
Environment=PYTHONUNBUFFERED=1
ExecStart=/path/to/alpha12_24/.venv/bin/python src/daemon/autosignal.py
StandardOutput=append:/var/log/alpha12/autosignal.out.log
StandardError=append:/var/log/alpha12/autosignal.err.log

[Install]
WantedBy=multi-user.target
```

## Autosignal Timer Template
```ini
[Unit]
Description=Run Alpha12_24 Autosignal every hour
Requires=autosignal.service

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

## Setup Instructions

1. Create log directory:
```bash
sudo mkdir -p /var/log/alpha12
sudo chown alpha12:alpha12 /var/log/alpha12
```

2. Install services:
```bash
sudo cp alpha12-ui.service /etc/systemd/system/
sudo cp alpha12-tracker.service /etc/systemd/system/
sudo cp alpha12-autosignal.service /etc/systemd/system/
sudo cp alpha12-autosignal.timer /etc/systemd/system/
```

3. Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable alpha12-ui
sudo systemctl enable alpha12-tracker
sudo systemctl enable alpha12-autosignal.timer
sudo systemctl start alpha12-ui
sudo systemctl start alpha12-tracker
sudo systemctl start alpha12-autosignal.timer
```

4. Check status:
```bash
sudo systemctl status alpha12-ui
sudo systemctl status alpha12-tracker
sudo systemctl status alpha12-autosignal.timer
```

5. View logs:
```bash
sudo journalctl -u alpha12-ui -f
sudo journalctl -u alpha12-tracker -f
sudo tail -f /var/log/alpha12/*.log
```
