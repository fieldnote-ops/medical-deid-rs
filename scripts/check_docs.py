"""Check local Markdown links and executable documentation entry points."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for path in root.rglob('*.md'):
    if '.git' in path.parts: continue
    text = path.read_text()
    if re.search(r'/(?:Users|home)/[A-Za-z0-9_-]+', text):
        raise ValueError(f'personal path in documentation: {path.relative_to(root)}')
    for link in re.findall(r'\]\(([^)]+)\)', text):
        if '://' in link or link.startswith('#'):continue
        target=(path.parent/link.split('#')[0]).resolve()
        if not target.exists():raise ValueError(f'broken local link: {path.name}: {link}')
for script in ['cargo.sh','deid.sh','env.sh','setup.sh','train_demo.sh','evaluate.sh','model.sh','package_release.sh']:
    assert (root/'scripts'/script).is_file(),script
print('Documentation links and command entry points passed')
