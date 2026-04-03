# metis

cli second brain that pairs with obsidian.

ingest anything. search by meaning. chat with your knowledge. discover connections.

---

## setup

```bash
git clone https://github.com/yourusername/metis.git
cd metis

# option a: install globally (use metis from anywhere)
uv tool install -e .

# option b: install in virtual env
uv venv
source .venv/bin/activate
uv pip install -e "."

metis init
metis --install-completion  # enable tab completion for commands
exec $SHELL                # restart shell to apply
```

store your api keys securely:

```bash
metis secret set openai-key
metis secret set azure-key     # if using azure
metis secret set x-token       # optional, for full x/twitter extraction
```

edit `~/.metis/config.yaml` for vault path and provider settings.

---

## commands

run `metis --help` or `metis <command> --help` for all options.

### ingest

save anything to your vault — summarized, tagged, embedded, and linked back to the source.

```bash
metis ingest https://en.wikipedia.org/wiki/Metis_(mythology)
metis ingest ~/books/project-hail-mary.pdf
metis ingest lecture-notes.md
metis ingest paper.pdf --folder research/ai
metis ingest https://www.youtube.com/watch?v=abc123
metis ingest https://www.youtube.com/watch?v=abc123 --pick-lang
metis ingest https://arxiv.org/abs/2401.12345
```

supports pdfs, urls, markdown, arxiv papers, youtube videos, and x/twitter posts. `--folder` organizes into vault subfolders.

---

### search

```bash
metis search "what role did titans play in greek mythos"
```

semantic search. finds by meaning, not keywords.

---

### chat

```bash
metis chat "how does project hail mary handle the fermi paradox?"
metis chat "what does he say about nash equilibrium?" --note game_theory/intro
metis chat "question" --note game_theory/intro --save
metis chat "question" --expand
```

answers grounded in your vault with sources cited. `--note` scopes to a specific note. `--save` writes the Q&A into the note. `--expand` offers to search wikipedia when your vault doesn't have enough.

---

### link

```bash
metis link
metis link --write
metis link --verbose
```

discovers connections between notes. `--write` adds `[[wikilinks]]` to the files. `--verbose` explains why notes are connected.

---

### sync

```bash
metis sync
```

re-indexes the vault after you edit notes in obsidian.

---

## config

```yaml
# ~/.metis/config.yaml

vault_path: ~/obsidian/my-vault
output_folder: metis-ingested

# option a: regular openai
provider: openai

# option b: azure openai
provider: azure
azure_openai:
  endpoint: https://your-resource.openai.azure.com/
```

api keys are stored in your os keychain via `metis secret set`. also reads from environment variables or config file as fallback.
