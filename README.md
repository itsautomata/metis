# metis

cli second brain that pairs with obsidian.

ingest anything. search by meaning. chat with your knowledge. discover connections.

---

## setup

```bash
git clone https://github.com/itsautomata/metis
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
metis secret set provider-key  # your one key (works for any provider via base_url)
metis secret set x-token       # optional, for full x/twitter extraction
metis secret set               # interactive, you pick which key you want to set
metis secret list              # show which keys are set (no values)
metis doctor                   # verify the whole setup: key, provider, models, index
metis models                   # show the chat + embedding models (and provider) in use
```

quick changes via `metis config vault <path>`. for everything else, edit `~/.metis/config.yaml`.

---

## commands

run `metis --help` or `metis <command> --help` for all options.

### ingest

save anything to your vault: summarized, tagged, embedded, and linked back to the source.

```bash
metis ingest https://en.wikipedia.org/wiki/Metis_(mythology)
metis ingest ~/books/project-hail-mary.pdf
metis ingest lecture-notes.md
metis ingest paper.pdf --folder research/ai
metis ingest https://www.youtube.com/watch?v=abc123 --lang fr
metis ingest https://arxiv.org/abs/2401.12345
metis ingest paper1.pdf paper2.pdf https://arxiv.org/abs/2402.00001
```

accepts one source or many at once. supports pdfs, urls, markdown, arxiv papers, youtube videos, and x/twitter posts. `--folder` organizes into vault subfolders.

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

### reindex

```bash
metis reindex
```

rebuilds the whole vector index from scratch. run it after changing your
`embedding_model`: the old vectors live in a different space, so metis refuses
search/link/health until you reindex.

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

```bash
metis config                       # show current settings
metis config vault ~/obsidian/my-vault
metis config folder metis-ingested
```

for anything not covered above, edit `~/.metis/config.yaml`:

```yaml
# ~/.metis/config.yaml

vault_path: ~/obsidian/my-vault
output_folder: metis-ingested

openai:
  # base_url points at any OpenAI-compatible provider.
  # leave empty for OpenAI; set it for OpenRouter, Ollama, a local server, etc.
  base_url: ""              # e.g. https://openrouter.ai/api/v1
  chat_model: gpt-4o
  embedding_model: text-embedding-3-small
```

api keys live in your os keychain via `metis secret set`, or a `METIS_*` env var for automation (`METIS_PROVIDER_KEY`, `METIS_EMBEDDING_KEY`, `METIS_X_BEARER`). keys never go in the config file. one key covers both chat and embeddings; set an `embedding-key` only if you split embeddings to a different provider. run `metis doctor` to check your setup: it verifies the key, the provider, the models, and the index in one pass.

> changing `embedding_model` re-spaces the whole index. metis will refuse until you run `metis reindex`.
>
> on a gateway like OpenRouter, embedding ids are vendor-prefixed; metis auto-adapts the default `text-embedding-3-small` to `openai/text-embedding-3-small`.
