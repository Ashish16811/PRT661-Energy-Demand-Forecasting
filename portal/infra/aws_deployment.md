# AWS deployment blueprint

## Public web tier
- Build the repository Docker image in **Amazon ECR**.
- Deploy the web command to **AWS App Runner** (or ECS/Fargate behind an ALB).
- Point a custom domain through Route 53/ACM if required.
- The web container is read-only with respect to modelling: it queries PostgreSQL and renders Dash only.

## Persistent services
- **Amazon RDS for PostgreSQL**: `actual_demand`, `project_forecast`, `forecast_accuracy`, `aemo_forecast`, `model_registry`, `pipeline_runs`.
- **Amazon S3**: raw snapshots, monthly Stage 2 ZIPs, model bundles and optional exported CSVs. The supplied project uses local paths; in cloud, sync `data/` and `models/` to S3 or mount EFS before/after jobs.
- **AWS Secrets Manager**: database credentials and any future API keys.

## Hourly automation
Create an ECS/Fargate task from the same image with command:

```bash
python -m pipeline.hourly_job
```

Schedule it with Amazon EventBridge Scheduler once per hour, preferably a few minutes after the hour so source publication has time to complete. The job:
1. refreshes only current NEM month/new WA tail;
2. acquires a current 7-day weather vintage;
3. stores official AEMO operational-demand forecast data when available;
4. loads promoted models;
5. creates a new immutable forecast vintage;
6. verifies older forecasts whose actuals have arrived.

## Monthly retraining
Second ECS/Fargate task, command:

```bash
python -m pipeline.monthly_retrain
```

Schedule for the first day of each month after the previous month is complete. It does **not** repeat Stage 3 model selection/backtesting. It rebuilds Stage 2, retrains only the already-approved algorithms, applies a recent holdout deployment gate, and promotes accepted bundles.

## First deployment only
Run once:

```bash
python -m pipeline.bootstrap
python -m pipeline.monthly_retrain
python -m pipeline.hourly_job
```

Then start the public web service.

## Production hardening
- Put RDS in private subnets; only workers/web security groups can connect.
- Public users never receive AWS/database credentials.
- Enable App Runner/ALB access logs and CloudWatch alarms for failed scheduled tasks.
- Set RDS automated backups and S3 versioning/lifecycle rules.
- Add CloudFront/WAF if traffic becomes material.
