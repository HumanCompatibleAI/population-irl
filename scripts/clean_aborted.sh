#!/bin/bash

EXCLUDE="deepq|cache|old"
find -mindepth 1 -maxdepth 1 -type d '!' -exec test -e '{}/results.pkl' ';' -print | grep -E -v "${EXCLUDE}" 
