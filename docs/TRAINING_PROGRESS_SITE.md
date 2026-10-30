# Training progress site

The static dashboard is in `site/`. GitHub Actions publishes it through GitHub
Pages when dashboard files are pushed to `master`.

For the first deployment, enable GitHub Pages for this repository and choose
**GitHub Actions** as the build and deployment source. The public URL is
<https://joconn819.github.io/poke-vgc-ai/>.

The dashboard intentionally reports only aggregate evaluation data and
experiment notes. It does not publish training rollouts, private team files,
checkpoints, or hidden-set training labels. Keep `PLAN.md` current when a new
run finishes, then regenerate and commit `site/progress.json` to update the
public summary.

Build the data locally after updating `PLAN.md`:

```bash
venv/bin/python scripts/build_training_progress.py
```
