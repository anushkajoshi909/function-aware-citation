# Citation Function Annotation Guidelines

## Scope
We will sample **50 papers**, taking **up to 2 citation-function instances per paper** (≈100 rows). Each row is annotated independently.

## Labels (single best label per row)
1. **Background** — gives context or general knowledge; not directly compared or used.
   - Cues: “as shown in…”, “following the definition in…”, “based on the work of…”
2. **Uses** — the cited work’s method/data/tool is **applied**.
   - Cues: “we use/apply/adopt…”, “built upon the dataset/tool by…”
3. **Comparison** — explicit performance or methodological **comparison**.
   - Cues: “outperforms/compared with/similar to/differs from…”
4. **Extension** — goes **beyond use** by modifying/generalizing/improving a prior method.
   - Cues: “we extend/generalize/improve upon…”
5. **Motivation/Future Work** — motivates a gap/limitation or outlines **future directions**.
   - Cues: “remains an open question”, “future work could…”, “limitations noted by…”
6. **Other/Mention** — neutral mention without a clear functional role above.

## Support (Yes/No)
“Supported = Yes” when the **retrieved top-1 passage** points to the **gold paper** and is **semantically appropriate** for the function. Otherwise “No”.

## Evidence
If **Supported = Yes**, copy a short quote (1–3 sentences) from the retrieved passage that justifies your decision.

## Decision Rules
- If torn between **Uses** and **Extension**: prefer **Uses** for straightforward application; choose **Extension** for substantive modifications/innovation.
- **Comparison** requires an explicit comparative statement.
- **Motivation/Future Work** is about research gaps, limitations, or future avenues — not general background.
- Prefer **Background** over **Other** when the citation clearly provides relevant conceptual context.
- One function per row.

## What to Annotate (per row)
- Confirm **Function** label.
- Mark **Supported** (TRUE/FALSE) for the retrieved top-1.
- Provide an **Evidence span** when Supported = TRUE.
- Add **Notes** for tricky cases.
