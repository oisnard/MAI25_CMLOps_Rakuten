# Grafana Automatic Configuration

This directory contains the configuration files that enable Grafana to automatically connect to Prometheus and load dashboards when the container starts up.

## What Happens Automatically

When you run `docker-compose up`, Grafana will:

1. **Start with predefined credentials**: `admin/admin`
2. **Automatically connect to Prometheus** at `http://prometheus:9090`
3. **Load pre-configured dashboards** for monitoring your MLOps pipeline
4. **Set up data sources** without manual intervention

## Configuration Files

### `grafana.ini`
- Sets default admin credentials
- Configures basic Grafana settings
- Disables sign-up and anonymous access

### `provisioning/datasources/prometheus.yml`
- Automatically configures Prometheus as a data source
- Sets Prometheus as the default data source
- Configures connection parameters

### `provisioning/dashboards/dashboards.yml`
- Tells Grafana where to look for dashboard JSON files
- Enables automatic dashboard loading

### Dashboard Files
- `fastapi-monitoring.json`: Basic FastAPI metrics dashboard
- `comprehensive-monitoring.json`: Full MLOps monitoring dashboard

## How It Works

1. **Volume Mounts**: The Docker Compose file mounts these configuration files into the Grafana container
2. **Provisioning**: Grafana reads these files on startup and automatically configures everything
3. **No Manual Setup**: You can immediately access Grafana at `http://localhost:3000` with everything ready

## Accessing Grafana

- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: admin

## Customization

To add more dashboards:
1. Create new JSON dashboard files in the `provisioning/dashboards/` directory
2. Restart the Grafana container
3. Dashboards will be automatically loaded

To modify data sources:
1. Edit `provisioning/datasources/prometheus.yml`
2. Restart the Grafana container

## Troubleshooting

If Prometheus connection fails:
1. Check that Prometheus container is running: `docker ps | grep prometheus`
2. Verify Prometheus is accessible: `curl http://localhost:9090/api/v1/targets`
3. Check container networking: Ensure both containers are on the same network 