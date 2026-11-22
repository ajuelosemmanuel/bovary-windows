# Security policy

## Scope and supported versions

Bovary is alpha software. Security fixes target the latest alpha source; older snapshots are retained for reproducibility and do not receive a separate maintenance commitment. The project is a local desktop/CLI/library workbench, not an authenticated multi-user inference service.

The native engine, model files and controlled code fixtures must come from sources you trust. Task validators are bounded application checks, not a general-purpose sandbox. Local execution is the default; commercial execution requires explicit provider, data and budget permissions.

## Reporting a vulnerability

Use this repository's **Security → Report a vulnerability** option when private reporting is available. Do not put API keys, private prompts/responses, model files or unsanitized logs into a public issue.

If private reporting is unavailable, open a minimal issue requesting a private reporting channel. Omit exploit details and sensitive data until a private channel is established. No response-time or remediation SLA is promised for this alpha.

Include the affected Bovary version or commit, Windows/Python versions, relevant configuration with secrets removed, expected and observed behavior, and a minimal synthetic reproduction where possible. State whether the issue involves local execution, a mocked provider or an explicitly authorized live-provider run.

## Before sharing diagnostics

Keep credentials in environment variables. Review paths and error messages before posting them. The ignored `work/` and `outputs/` directories can contain settings, model files, prompts, answers and raw evidence; they are outside the public source boundary. Do not attach them wholesale.
