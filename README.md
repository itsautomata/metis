# metis

cli second brain that pairs with obsidian.

## setup

```bash
git clone https://github.com/yourusername/metis.git
cd metis
uv venv
source .venv/bin/activate
uv pip install -e "."
metis init
```

edit `~/.metis/config.yaml` with your vault path and api key.

## commands

### `metis ingest <source>`

```bash
metis ingest https://en.wikipedia.org/wiki/Metis_(mythology)
metis ingest ~/books/project-hail-mary.pdf
metis ingest lecture-notes.md
metis ingest paper.pdf --folder research/ai
```

extracts text, summarizes, tags, embeds, saves as a native obsidian note with a link back to the source. `--folder` lets you organize into vault subfolders. defaults to `output_folder` in `~/.metis/config.yaml` if not specified.

### `metis search <query>`

```bash
metis search "what role did titans play in greek mythos"
```

semantic search. finds by meaning, not keywords.

### `metis chat <question>`

```bash
metis chat "how does project hail mary handle the fermi paradox?"
```

answers grounded in your vault with sources cited.

### `metis link [note]`

```bash
metis link
metis link --write
```

finds connections between notes. `--write` adds `[[wikilinks]]` to the files.

### `metis sync`

```bash
metis sync
```

re-indexes the vault after you edit notes in obsidian.

