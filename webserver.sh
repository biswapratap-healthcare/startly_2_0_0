#!/bin/bash
cd /app
Process1=$(pgrep -f -x "/usr/bin/python3 api.py")
if [ ! -z "$Process1" -a "$Process1" != " " ]; then
        echo "api.py Running"
else
        echo "api.py is not running"
 /usr/bin/python3 api.py 2>&1
 echo "api.py started"
fi

