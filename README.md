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
# to update global install
uv tool install -e . --force

# option b: install in virtual env (for development)
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
metis secret set               # interactive — pick which key
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
metis ingest https://www.youtube.com/watch?v=abc123 --lang fr
metis ingest https://arxiv.org/abs/2401.12345
```

supports pdfs, urls, markdown, arxiv papers, youtube videos, and x/twitter posts. `--folder` organizes into vault subfolders.

> interactive: `--pick-folder` (vault folders) / `--pick-lang` (transcript languages)

---

### search

```bash
metis search "what role did titans play in greek mythos"
```

semantic search. finds by meaning, not keywords. pick a result to chat about it.

---

### chat

```bash
metis chat "how does project hail mary handle the fermi paradox?"
metis chat "what does he say about nash equilibrium?" --note game_theory/intro
metis chat "question" --note game_theory/intro --save
metis chat "question" --expand
```

answers grounded in your vault with sources cited. `--note` scopes to a specific note and offers to save the Q&A. `--save` saves without prompting. `--expand` searches wikipedia when your vault doesn't have enough.

> interactive: `--pick` (choose vault note to ask about)

---

### link

```bash
metis link
metis link --write
metis link --verbose
```

discovers connections between notes. `--write` adds `[[wikilinks]]` to the files. `--verbose` explains why notes are connected.

> interactive: `--pick` (choose vault note to find connections for)

---

### sync

```bash
metis sync
```

re-indexes the vault after you edit notes in obsidian.

---

## classification & clustering

metis learns from your vault to help you organize.

**auto-categorization:** when you ingest without `--folder`, metis suggests a folder based on your vault's content. accept, override, or pick from menu. every choice improves future suggestions.

**vault health:**

```bash
metis health
metis health --misplaced
metis health --split hermes_folder
metis health --unique
```

checkup on your vault structure. shows folder alignment, suggests which notes might belong in a different folder, and proposes subfolder splits for large folders.

**folder descriptions:**

```bash
metis folders
metis folders --edit
```

list folders with their ML descriptions. `--edit` opens in your editor to refine how the classifier understands each folder.

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
