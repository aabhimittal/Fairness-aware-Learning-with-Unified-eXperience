# Releasing

Releases are automated with **PyPI trusted publishing** — no API tokens.

## One-time setup (repo owner)

1. On [pypi.org](https://pypi.org) → your account → *Publishing* → add a
   **pending publisher** (or a publisher on the existing project):
    - PyPI project name: `flux-ranking`
    - Owner: `aabhimittal`
    - Repository: `Fairness-aware-Learning-with-Unified-eXperience`
    - Workflow name: `publish.yml`
    - Environment: `pypi`
2. On GitHub → repo *Settings → Environments* → create an environment named
   `pypi` (optionally add required reviewers as a release gate).

## Cutting a release

```bash
# 1. bump version in pyproject.toml AND flux/__init__.py, update CHANGELOG.md
# 2. merge to main, then:
git tag v0.3.0
git push origin v0.3.0
```

The tag push triggers `.github/workflows/publish.yml`, which builds the
sdist and wheel, runs `twine check`, and publishes via OIDC. Watch the run
under the repo's *Actions* tab.

## Docs deployment

`.github/workflows/docs.yml` builds this MkDocs site and pushes it to the
`gh-pages` branch on every push to `main`. One-time setup: repo *Settings →
Pages* → deploy from the `gh-pages` branch.
