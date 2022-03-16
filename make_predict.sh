#!/usr/bin/env bash

PORT=5000
echo "Port: $PORT"

# POST method predict
curl -X POST -H "Content-Type: application/json" \
   --data '["MLOPS is critical for robustness"]' \
      http://localhost:$PORT/predict

curl -X POST -H "Content-Type: application/json" \
   --data '["Containers are more or less interesting"]' \
      http://localhost:$PORT/predict