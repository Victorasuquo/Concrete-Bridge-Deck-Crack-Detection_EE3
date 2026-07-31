git checkout main
git pull origin main
git checkout -b docs/evaluation-notes
# create or edit: docs/evaluation_notes.md
git add .
git commit -m "docs: explain evaluation metrics and the imbalance problem"
git push origin docs/evaluation-notes
# then open the pull request on GitHub
