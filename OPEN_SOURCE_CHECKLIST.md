# Open Source Release Checklist

Use this before making the repository public.

- [x] Replace legacy implementation-risk code with an independent AdaptiveStreamFL reference implementation.
- [x] Add `LICENSE` using the MIT License.
- [x] Keep large datasets outside Git.
- [x] Remove experiment-analysis scripts and hard-coded result templates from the public release.
- [x] Keep citation metadata in `CITATION.cff`.
- [ ] Run smoke tests in a fresh environment.
- [ ] Create a release tag and archive the exact source version used for public release.
- [ ] Optionally mirror the release to Zenodo for a citable software DOI.
