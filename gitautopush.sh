#!/bin/bash

# Script to commit all changes in development branch of git project
# and push to remote repository. Development branch must exist and
# credentials for pushing to remote repo must be set up for this to 
# work. This script is run by a cron job and takes a single parameter
# which is the top level folder of the git project.
# Logs for the cron job are in the file /var/spool/mail/rstones

cd /home/rstones/git/$1
git checkout development
git add ./
now=$(date)
git commit -m "Auto-commit at $now"
git push origin development
exit 0
