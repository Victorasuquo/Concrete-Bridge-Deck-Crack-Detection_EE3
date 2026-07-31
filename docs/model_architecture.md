git checkout main
git pull origin main
git checkout -b docs/model-architecture
# create or edit: docs/model_architecture.md
git add .
git commit -m "docs: document model architecture and training stages"
git push origin docs/model-architecture
# then open the pull request on GitHub
