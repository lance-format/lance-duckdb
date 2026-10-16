---
name: lance-duckdb-update-lance-dependency
description: Update lance-duckdb to a specific Lance release or tag. Use when bumping Lance Rust dependencies in the lance-duckdb repository, including Cargo.toml, Cargo.lock, validation, branch creation, commit, push, and PR creation when requested.
---

# lance-duckdb Update Lance Dependency

## Scope

Use this skill in the `lance-format/lance-duckdb` repository when updating Lance Rust dependencies to a specific Lance version or tag.

Inputs can be a version (`7.2.0-beta.1`), a tag (`v7.2.0-beta.1`), a tag ref (`refs/tags/v7.2.0-beta.1`), or `latest`.

## Workflow

1. Confirm the worktree status with `git status --short`.
2. Resolve the target Lance version:

   - If the input is `latest`, empty, or omitted, run:

     ```bash
     python3 ci/check_lance_release.py
     ```

     Parse the JSON output. If `needs_update` is not `true`, stop without creating a PR. Otherwise use `latest_tag`.

   - If the input is explicit, use it directly.

3. Compute update metadata without changing files:

   ```bash
   python3 ci/update_lance_dependency.py "$TAG_OR_VERSION" --metadata-only
   ```

   Before making changes, check for an existing open PR with the emitted `pr_title`:

   ```bash
   gh pr list --search "\"$PR_TITLE\" in:title" --state open --limit 1 --json number,url,title
   ```

   If a matching open PR exists, stop and report it instead of creating a duplicate.

4. Run the deterministic update entrypoint:

   ```bash
   python3 ci/update_lance_dependency.py "$TAG_OR_VERSION"
   ```

   This updates the direct Lance Rust crate versions in `Cargo.toml`, removes repository-local Lance `[patch.crates-io]` entries if present, refreshes `Cargo.lock` with precise Cargo updates, and prints JSON metadata containing `branch_name`, `commit_message`, and `pr_title`.

5. Run validation:

   ```bash
   cargo fmt --all
   cargo check --manifest-path Cargo.toml
   cargo clippy --manifest-path Cargo.toml --all-targets
   ```

   If Cargo reports incompatible Arrow/DataFusion requirements, inspect the target Lance release requirements and update the pinned Arrow/DataFusion versions in `Cargo.toml`, then rerun `python3 ci/update_lance_dependency.py "$TAG_OR_VERSION"` and the validation commands. Fix real diagnostics and rerun validation until it succeeds.

6. Inspect `git status --short` and `git diff` to ensure only the Lance dependency update and required compatibility fixes are present.

7. If the task only asks to prepare local changes, stop here and report the changed files and validation result.

8. If the task asks to publish the update, create a branch using the printed `branch_name`, stage all relevant files, and commit using the printed `commit_message`. Do not amend or rewrite existing commits.

9. Push to `origin`. Before creating the PR, check that the current token has push permission:

   ```bash
   gh api repos/lance-format/lance-duckdb --jq .permissions.push
   ```

   If the remote branch already exists for the same generated branch name, delete the remote ref with `gh api -X DELETE repos/lance-format/lance-duckdb/git/refs/heads/$BRANCH_NAME`, then push. Do not force-push.

10. Create a PR targeting `main` with the printed `pr_title`. If there is no PR template, keep the body to two or three concise sentences: state the Lance dependency bump, note any required compatibility fixes, and link the triggering Lance tag or release.

11. Read back the remote PR title after creation. If it is not a Conventional Commit title, fix it immediately.

## GitHub Actions

This workflow is intentionally manual. When this skill is used from GitHub Actions, `TAG`, `GH_TOKEN`, and `GITHUB_TOKEN` may already be set. Resolve `latest` first when `TAG` is empty. Once an explicit tag or version is known, use:

```bash
python3 ci/update_lance_dependency.py "$TAG" --github-output "$GITHUB_OUTPUT"
```

Then use the emitted `branch_name`, `commit_message`, and `pr_title` values for branch, commit, and PR creation.
