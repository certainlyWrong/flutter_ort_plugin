#!/usr/bin/env python3
import subprocess
import pathlib
import sys


def run(cmd: list[str]) -> str:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    ).stdout.strip()


def current_branch() -> str:
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def commits(base: str, head: str) -> str:
    return run(["git", "log", "--pretty=format:- %s", f"{base}..{head}"])


def changed_files(base: str, head: str) -> str:
    return run(["git", "diff", "--name-status", f"{base}..{head}"])


def patch_summary(base: str, head: str, limit: int = 10) -> str:
    out = []
    files = run(["git", "diff", "--name-only", f"{base}..{head}"]).splitlines()
    for f in files[:limit]:
        p = run(["git", "diff", f"{base}..{head}", "--", f])
        if p:
            out.append(f"\n### {f}\n```diff\n{p[:1500]}\n```")
    return "\n".join(out)


def read_template():
    for f in [".pull_request_template.md", ".github/pull_request_template.md"]:
        if pathlib.Path(f).exists():
            return pathlib.Path(f).read_text()
    return "# Problema\n\n## Solução\n"


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "main"
    head = current_branch()

    commits_txt = commits(base, head)
    files_txt = changed_files(base, head)
    patches_txt = patch_summary(base, head)
    template = read_template()

    prompt = f"""
Você é um engenheiro experiente escrevendo a descrição de um Pull Request.

Preencha este **template de PR em português**:

---

{template}

---

Contexto:
- Repositório local: {pathlib.Path('.').resolve().name}
- Branch: `{head}` ← `{base}`

Commits:
{commits_txt}

Arquivos alterados:
{files_txt}

Patches (exemplo):
{patches_txt}

---

**Responda com o template preenchido**, sem comentários extras.
""".strip()

    pathlib.Path("pr_prompt.txt").write_text(prompt, encoding="utf-8")
    print("✅ Prompt salvo em pr_prompt.txt")


if __name__ == "__main__":
    main()
