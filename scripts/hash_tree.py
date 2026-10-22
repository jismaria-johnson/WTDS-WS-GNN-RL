"""SHA-256 of every file under one or more folders (read-only).

Usage: python hash_tree.py <out.txt> <folder> [<folder> ...]
Each line: <sha256>  <folder-name>/<relative path>
"""
import hashlib
import os
import sys

out, roots = sys.argv[1], sys.argv[2:]
rows = []
for root in roots:
    name = os.path.basename(os.path.normpath(root))
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            rows.append(f"{h.hexdigest()}  {name}/{rel}")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(rows) + "\n")
print(f"{len(rows)} files hashed -> {out}")
