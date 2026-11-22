# Bovary

**Local inference, on your terms.**

Bovary is the owner-selected name for this Windows-first inference workbench. The name makes no speed, cost or quality claim and does not imply that automatic commercial routing is qualified.

The distribution and Python implementation are `bovary`. The console commands are `bovary` and `bovary-settings`; `python -m bovary` uses the same CLI. The source checkout includes `Bovary.cmd` to open the settings window through its configured `.venv`. The package exposes Request, Result, Event, ExecutionError and create_workbench. Runtime configuration uses `BOVARY_HOME`, `BOVARY_MODEL`, `BOVARY_MODEL_MANIFEST`, `BOVARY_CPU_EXE` and `BOVARY_CUDA_EXE`. Native targets are `bovary-infer` and `bovary-quantize`, with `BOVARY_` CMake controls and streaming settings. See [configuration](configuration.md) and [reproducibility](reproducibility.md) for paths and commands.

This is an alpha interface. Historical research artifacts retain the identifiers recorded when they were produced; they are dated evidence, not current commands or runtime configuration. Checkpoints are separate user-supplied assets and are not renamed or redistributed by the package.

## Name screening, 14 November 2026

The [PyPI project JSON endpoint](https://pypi.org/pypi/bovary/json) returned HTTP 404 at check time. A read-only [GitHub repository-name query](https://api.github.com/search/repositories?q=bovary%20in%3Aname&per_page=15) returned HTTP 200 with 33 repositories, including books and unrelated sites. Bovary is therefore not a globally unique name.

Limited primary checks found [tiste/bovary](https://github.com/tiste/bovary) to be a Middleman/Ruby website for Champagne Bovary. Two other exact-name repositories returned no README and no language metadata. This limited inspection did not establish a directly competing inference package, and it does not rule out other uses. No package, domain, account or repository was registered. These dated observations do not reserve the name or establish trademark clearance.

The proposed public repository slug is `bovary`. The owner still chooses the hosting account and visibility and authorizes publication. Software licensing is documented separately in [licensing.md](licensing.md).
