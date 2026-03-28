# metis

cli second brain that pairs with obsidian.

ingest anything. search by meaning. chat with your knowledge. discover connections.

---

## setup

```bash
git clone https://github.com/yourusername/metis.git
cd metis
uv venv
source .venv/bin/activate
uv pip install -e "."
metis init
metis --install-completion  # enable tab completion for commands
exec $SHELL                # restart shell to apply
```

edit `~/.metis/config.yaml` with your vault path and api key.

---

## commands

run `metis --help` or `metis <command> --help` for all options.

### ingest

save anything to your vault: summarized, tagged, embedded, and linked back to the source.

```bash
metis ingest https://en.wikipedia.org/wiki/Metis_(mythology)
metis ingest ~/books/project-hail-mary.pdf
metis ingest lecture-notes.md
```

**organize into folders:**

```bash
metis ingest paper.pdf --folder research/ai
```

defaults to `output_folder` in config if not specified.

**youtube videos:**

```bash
metis ingest https://www.youtube.com/watch?v=abc123
metis ingest https://www.youtube.com/watch?v=abc123 --lang fr
metis ingest https://www.youtube.com/watch?v=abc123 --pick-lang
```

transcripts default to english. `--lang` picks a specific language. `--pick-lang` shows an interactive menu. if no transcript exists, metis asks if you want to save the link anyway.

**arxiv papers:**

```bash
metis ingest https://arxiv.org/abs/2401.12345
```

auto-detected. downloads the pdf, extracts text, grabs the paper title.

**supported sources:** pdfs, urls, markdown, arxiv papers, youtube videos.

---

### search

```bash
metis search "what role did titans play in greek mythos"
metis search "how does project hail mary handle the fermi paradox"
```

semantic search. finds by meaning, not keywords.

---

### chat

```bash
metis chat "how does project hail mary handle the fermi paradox?"
```

answers grounded in your vault with sources cited. retrieves, reasons, retrieves again if needed.

---

### link

```bash
metis link
metis link --write
```

discovers connections between notes. `--write` adds `[[wikilinks]]` to the files.

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
openai:
  api_key: sk-...

# option b: azure openai
provider: azure
azure_openai:
  endpoint: https://your-resource.openai.azure.com/
  api_key: your-key
```
