#!/usr/bin/env bash

aws cloudwatch delete-alarms --alarm-name 'pirl-head-idle'
INSTANCE_ID=`curl http://169.254.169.254/latest/meta-data/instance-id`
aws cloudwatch put-metric-alarm --alarm-name 'pirl-head-idle' \
    --namespace AWS/EC2 --metric-name CPUUtilization \
    --threshold 20 --comparison-operator LessThanThreshold \
    --statistic Average --period 3600 \
    --datapoints-to-alarm 12 --evaluation-periods 24 \
    --alarm-actions arn:aws:sns:us-west-2:286342508718:default \
    --dimensions "Name=InstanceId,Value=${INSTANCE_ID}"
