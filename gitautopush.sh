clear
#git checkout -b development
cd /home/rstones/git/quant_mech
git add ./
now=$(date)
git commit -m "Auto-commit at $now"
git push origin development
exit 0
