"""Write a copy of a table notebook (03a/03b) with `challenge` set to the given hub."""
import json, re, sys
src, dst, hub = sys.argv[1:4]
nb = json.load(open(src))
cell = nb["cells"][1]
s = "".join(cell["source"])
s2 = re.sub(r"""^challenge = ['"]\w+['"]""", f'challenge = "{hub}"', s, count=1, flags=re.M)
assert s2 != s or f'"{hub}"' in s or f"'{hub}'" in s
cell["source"] = s2.splitlines(keepends=True)
json.dump(nb, open(dst, "w"), indent=1)
