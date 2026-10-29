# Dataset card: Voz comment moderation (clean / offensive / hate)

> **Content warning:** the data contains offensive and hateful language.

The anonymised splits are in this folder: `train.csv`, `val.csv`, `test.csv`. They were produced with `python data/anonymize.py` (see below), so they contain no usernames or post links. The raw collected pool is not released.

## Source and collection

| Item | Value |
|---|---|
| Source | Public comments on the Voz forum (voz.vn) |
| Collected pool | about 500,000 comments |
| Comments in the released dataset | 18,046 (about 3.6% of the pool) |
| Posting period | March 2020 to September 2026 |
| Threads / distinct accounts | 12,668 threads, 9,135 accounts |
| Largest sub-forums | Điểm báo (5,027), 4 bánh (2,655), Chuyện trò linh tinh™ (2,609), from f17 with love (1,124), Android (668), PC Gaming (609) |
| Crawling tool and crawl date | **[TODO]** |

## Labels

- **hate**: insults, contempt or incitement against a *group* of people (nationality, region, religion, gender, sexual orientation, political view).
- **offensive**: profanity, insults or attacks on an individual or an object, not aimed at a group.
- **clean**: everything else, including slang used only for emphasis.

### Labelling procedure

1. All collected comments were pre-labelled by **Gemini** [TODO: model version and date], prompted with the rules above (no separate guideline document).
2. Each of the three authors verified the comments Gemini assigned to **one** class. Wrong labels were corrected or the comment was discarded.
3. Verification stopped when each class reached its target size (ratio about 50/30/20).

Each comment was therefore checked by **one** person, and **inter-annotator agreement was not measured**.

## Splits

| Split | clean | offensive | hate | Total |
|---|---|---|---|---|
| train | 7,220 | 4,330 | 2,886 | 14,436 |
| val | 902 | 542 | 361 | 1,805 |
| test | 903 | 541 | 361 | 1,805 |

About 59.4% of comments quote an earlier comment (`quoted_text`).

## Columns after `anonymize.py`

| Column | Meaning |
|---|---|
| `thread_id` | integer ID of the thread (replaces the URL) |
| `thread_title` | title of the thread |
| `category` | sub-forum |
| `posted_at` | month of posting (YYYY-MM) |
| `quoted_text` | cleaned text of the quoted comment, may be empty |
| `text` | cleaned comment text; links replaced by `<url>`; @mentions and usernames in nested "… said:" quote headers replaced by `@user` |
| `label` | clean / offensive / hate |
| `author_id` | only with `--keep-author-ids`: salted pseudonym, not reversible without the secret salt |

Masking links and usernames changes about 5% of comments slightly compared with the text the models were trained on.

## Known limitations

- Single forum; labels proposed by an LLM and checked by one verifier per class, so verifiers may have favoured the model's label and agreement is unknown.
- Keeping verified comments until each class was full may favour easy examples; the 50/30/20 ratio is a design choice, and hate is much rarer on the real forum.

## Licence and intended use

Data: **CC BY-NC 4.0**, for research only. Do not use it to identify, profile or target the authors of comments.
Check the Voz terms of use before redistributing. [TODO: confirm]
