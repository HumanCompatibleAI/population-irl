#!/usr/bin/env bash

INSTANCE_ID=`ec2-metadata --instance-id | cut -d' ' -f 2`
EC2_REGION=`ec2-metadata --availability-zone | sed 's/[a-z]$//' | cut -d' ' -f 2`
aws cloudwatch delete-alarms --region ${EC2_REGION} --alarm-name 'pirl-head-idle'
aws cloudwatch put-metric-alarm --region ${EC2_REGION} --alarm-name 'pirl-head-idle' \
    --namespace AWS/EC2 --metric-name CPUUtilization \
    --threshold 20 --comparison-operator LessThanThreshold \
    --statistic Average --period 3600 \
    --datapoints-to-alarm 12 --evaluation-periods 24 \
    --alarm-actions arn:aws:sns:us-west-2:286342508718:default \
    --dimensions "Name=InstanceId,Value=${INSTANCE_ID}"
