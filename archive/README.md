# CHANGES_LOG Archive

This directory contains archived entries from `CHANGES_LOG.md`.

## Archiving Policy

- Entries older than 3 months are moved from the main `CHANGES_LOG.md` to monthly archive files
- Archive files are named `CHANGES_YYYY-MM.md` (e.g., `CHANGES_2026-01.md` for January 2026)
- All history is preserved - nothing is deleted
- Main log stays focused on recent work for better readability

## Archive Files

Archive files will be created as entries age:
- `CHANGES_2026-01.md` - January 2026 entries
- `CHANGES_2026-02.md` - February 2026 entries
- `CHANGES_2026-03.md` - March 2026 entries (when entries become >3 months old)
- etc.

## Searching Archive

To search across all archived changes:
```bash
# Search all archive files
grep -r "search term" archive/

# Search specific month
grep "search term" archive/CHANGES_2026-01.md
```
