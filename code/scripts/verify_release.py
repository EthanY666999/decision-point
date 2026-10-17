"""Verify published files against the release manifest (stdlib only)."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
m=json.loads((root/'RELEASE_MANIFEST.json').read_text())
for name,item in m['files'].items():
 p=root/name
 if not p.is_file():raise SystemExit(f'Missing: {name}')
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=item['sha256'] or p.stat().st_size!=item['bytes']:raise SystemExit(f'Mismatch: {name}')
print(f'OK: {len(m["files"])} published files match SHA256/size.')
