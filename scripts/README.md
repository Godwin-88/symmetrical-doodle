# Scripts Directory

## Startup Scripts

### Start All Services

**Windows:**
```powershell
.\start-all.ps1
```

**Linux/Mac:**
```bash
./start-all.sh
```

Starts:
- Database services (Docker)
- Python Intelligence Layer (port 8000)
- Rust Execution Core (port 8001)
- Rust Simulation Engine (port 8002)
- React Frontend (port 5173)

### Stop All Services

**Windows:**
```powershell
.\stop-all.ps1
```

**Linux/Mac:**
```bash
./stop-all.sh
```

## Other Scripts

- `test-deriv-connection.py` - Test Deriv API connection
- `test-integration.ps1` / `test-integration.sh` - Integration tests
- `validate-setup.ps1` / `validate-setup.sh` - System validation

## See Also

- `../STARTUP_GUIDE.md` - Complete startup documentation
- `../DERIV_INTEGRATION_GUIDE.md` - Deriv API guide
