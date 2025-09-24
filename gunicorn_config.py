import multiprocessing

# Server socket
bind = "0.0.0.0:8010"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "dsaSeekhogpt"

# SSL
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile" 