# Training progress site

The static dashboard is in `site/`. GitHub Actions publishes it through GitHub
Pages when dashboard files are pushed to `master`.

For the first deployment, enable GitHub Pages for this repository and choose
**GitHub Actions** as the build and deployment source. The public URL is
<https://joconn819.github.io/poke-vgc-ai/>.

The dashboard reports aggregate evaluation data and experiment notes across
multiple opponent classes. It does not publish training rollouts, private team
files, checkpoints, or hidden-set training labels. Update the curated
`site/progress-source.json` after a run finishes, then regenerate
`site/progress.json` and commit both files to update the public summary. No
single opponent's win rate is treated as the project completion target.

Build the data locally after updating `PLAN.md`:

```bash
python3 scripts/build_training_progress.py
```
