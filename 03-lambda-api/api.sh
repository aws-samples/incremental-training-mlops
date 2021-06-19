#!/bin/bash

#OD_API='https://2lvd1jdy8l.execute-api.us-west-2.amazonaws.com/test/classify'
OD_API=$1

for f in $(ls ../02-increment-train/audios/*.wav)
do
    curl -X POST -H 'content-type: application/octet-stream' --data-binary @$f $OD_API | jq .
done
