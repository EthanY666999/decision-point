import pathlib,os,json,hashlib,sys,time
R=pathlib.Path('/home/vipuser/golden-cert-20261102-r1/code/artifacts/deep_disk_20261103');m=json.load((R/'local_candidates.json').open());roots={p['label']:p['run'] for p in m['packages']}
for method in ['grpo','dapo','patr','tree_random']:roots['historical_v1_'+method]='/home/vipuser/patr-reproduction/runs/formal/'+method+'_seed42'
entries={}
for label,root in roots.items():
 if not pathlib.Path(root).exists():continue
 for parent,dirs,files in os.walk(root):
  for name in files:
   p=pathlib.Path(parent)/name
   if name.startswith('checkpoint') or name.endswith(('.pt','.safetensors','.ARCHIVED.json')) or p.is_symlink():continue
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
   entries[label+'/'+str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
if sys.argv[1]=='before':(R/'nonweight_before.json').write_text(json.dumps(entries,indent=2))
else:
 expected=json.load((R/'nonweight_before.json').open());assert entries==expected,'Nonweight historical evidence changed'
 (R/'integrity_after.json').write_text(json.dumps({'checked_unix':time.time(),'all_nonweight_files_unchanged':True,'file_count':len(entries),'bytes':sum(x['bytes'] for x in entries.values())},indent=2))
print(sys.argv[1],len(entries),sum(x['bytes'] for x in entries.values()),flush=True)
