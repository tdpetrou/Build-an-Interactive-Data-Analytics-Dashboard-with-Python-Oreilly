cd /var/www/html/corona
dashboard_venv/bin/python update.py >> /var/log/corona_cron.log 2>&1 
systemctl restart corona