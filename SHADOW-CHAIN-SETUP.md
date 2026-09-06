# Shadow chain on GitHub Actions — setup (one-time, ~10 minutes)

The nightly job builds the pre-registered signal files, hashes and chains them, stamps them in Bitcoin
(OpenTimestamps), runs the integrity check, and stores everything in the private HF dataset
`deepakdsoni/emporion-data` under `shadow/` (`state.tar` = point-in-time stores + signals; `signals/` = the files).
No GPU, no server: GitHub's free scheduled runner, ~5 minutes a night.

1. **Rotate the HF token** (the old one passed through chat). Create a new *write* token at
   https://huggingface.co/settings/tokens and revoke the old one.
2. **Create a private GitHub repository** (e.g. `emporion`) and push this folder:
   ```bash
   cd Trading-DLM
   git remote add origin git@github.com:<you>/emporion.git    # or the https URL
   git push -u origin main
   ```
   (`runs/` is git-ignored: data lives on HF, code on GitHub.)
3. **Add the secret**: repository → Settings → Secrets and variables → Actions → New repository secret
   `HF_TOKEN` = the new token.
4. **First run by hand**: Actions → `shadow-chain` → Run workflow. Check the log ends with `chain ok` and that
   `shadow/signals/` on HF gained today's files and `.ots` proofs.
5. From then on it runs at 23:30 UTC daily. A missed night is recorded as a gap, never back-filled into the
   live chain. `scripts/integrity_check.py` upgrades and verifies the OpenTimestamps proofs as they mature
   (Bitcoin confirmation takes hours to a day).

What the chain produces: one out-of-sample print per day per signal. Certification (research/03 §10) needs
roughly 6–12 months of prints for a 20-day signal. `runs/signals-backfill/REPORT.md` shows the report format on a
rehearsal month (August 2026) — built after the fact, flagged BACKFILL inside every hash, never stamped, not evidence.

State today: live chain day 1 = 2026-09-06 (attested, Bitcoin block 965744); 2026-09-07 onward comes from Actions once
steps 1–4 are done. Local mirror of the state: `runs/signals/` (git-ignored).
