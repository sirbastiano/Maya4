# GitHub Actions Secrets Setup

This document explains how to configure GitHub Actions secrets for the Maya4 repository.

## Required Secrets

### HF_TOKEN (HuggingFace Token)

**Purpose**: Allows GitHub Actions to download SAR datasets from HuggingFace Hub during CI/CD testing.

**Required for**: 
- Online data download tests
- Accessing private HuggingFace datasets (if applicable)
- Rate limit increases on HuggingFace API

**How to set up**:

1. **Get Your HuggingFace Token**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "GitHub Actions - Maya4")
   - Select access level:
     - **Read**: Sufficient for public datasets
     - **Write**: Required if uploading artifacts
   - Click "Generate token"
   - **Copy the token** (you won't see it again!)

2. **Add to GitHub Repository**
   - Navigate to https://github.com/sirbastiano/Maya4
   - Click **Settings** (top navigation)
   - In left sidebar: **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `HF_TOKEN`
   - Value: Paste your HuggingFace token
   - Click **Add secret**

3. **Verify Setup**
   - Push a commit to the `feature/dataloader-refactor` branch
   - Go to **Actions** tab in GitHub
   - Check that the "Download test data with online mode" step succeeds
   - Look for successful downloads in the logs

## Security Notes

- ✅ Secrets are encrypted and not exposed in logs
- ✅ Secrets are only available to workflows in your repository
- ✅ Read-only tokens minimize security risk
- ⚠️ Never commit tokens directly to code
- ⚠️ Rotate tokens periodically for security

## Optional: Organization-Level Secrets

If you have multiple repositories that need the same token:

1. Go to your organization settings: https://github.com/organizations/sirbastiano/settings/secrets/actions
2. Create an organization secret with the same name
3. Select which repositories can access it

## Troubleshooting

### Online tests fail with authentication errors
- Verify the `HF_TOKEN` secret exists in repository settings
- Check token hasn't expired on HuggingFace
- Ensure token has "Read" access

### Downloads are slow or rate-limited
- Consider using a token with higher rate limits
- Reduce `max_products` in test configuration
- Use organization-level tokens if available

### Token not recognized
- Ensure secret name is exactly `HF_TOKEN` (case-sensitive)
- Check workflow file uses `${{ secrets.HF_TOKEN }}`
- Verify secret is set at repository level, not environment level

## Workflow Usage

The token is automatically injected into the workflow as an environment variable:

```yaml
- name: Download test data with online mode
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    pytest tests/test_online_download.py -v --tb=short -m online
```

The `huggingface_hub` library automatically detects and uses the `HF_TOKEN` environment variable.

## Related Documentation

- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [HuggingFace Tokens](https://huggingface.co/docs/hub/security-tokens)
- [Maya4 CI/CD Workflow](../.github/workflows/tests.yml)
