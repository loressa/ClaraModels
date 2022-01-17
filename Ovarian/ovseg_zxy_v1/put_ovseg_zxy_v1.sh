#!/bin/bash

URL="127.0.0.1:80"
model_name="ovseg_zxy_v1"
model_path="/home/ubuntu/models/ovseg_zxy"

curl -X PUT "http://$URL/admin/model/$model_name?native=true" -F "config=@$model_path/config_ovseg_zxy_v1.json;type=application/json" -F "data=@$model_path/ovseg_zxy_v1.ts"
