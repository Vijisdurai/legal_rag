from preprocessing.load_bns_json import load_bns_sections, get_ipc_to_bns_map
from retrieval.dual_corpus import load_combined_corpus

bns      = load_bns_sections()
combined = load_combined_corpus()
ipc_map  = get_ipc_to_bns_map()

ipc_only = [c for c in combined if c.get('corpus', 'ipc') == 'ipc']
bns_only = [c for c in combined if c.get('corpus', 'ipc') == 'bns']

print(f"BNS sections   : {len(bns)}")
print(f"Combined corpus: {len(combined)} (IPC={len(ipc_only)}, BNS={len(bns_only)})")
print(f"IPC-to-BNS map : {len(ipc_map)} entries")

for c in bns_only:
    if c['section_number'] == '103':
        print(f"BNS §103: {c['title']} | IPC eq: {c['ipc_equivalent']}")
        break

# Check new BNS-only sections
for c in bns_only:
    if c['section_number'] in ['111', '112', '113']:
        print(f"BNS §{c['section_number']}: {c['title'][:60]}")
