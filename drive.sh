#!/bin/sh
project=${HOME}/workspace/jsclassifier/
pushd ./src/com/rikima/albert/js
java -Xmx1280M -jar ${project}/lib/js.jar train.js
popd