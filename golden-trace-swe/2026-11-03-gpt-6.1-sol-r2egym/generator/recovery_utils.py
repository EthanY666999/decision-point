"""Narrow recovery eligibility; never retry ambiguous repository mutations."""
def transient_image_error(message):
    text=str(message).lower()
    permanent=('manifest unknown','pull access denied','insufficient_scope','unauthorized:','invalid reference format','404 client error')
    if any(x in text for x in permanent):return False
    return any(x in text for x in ('eof','connection reset','connection refused','timed out','timeout','temporary failure','tls handshake','502 bad gateway','503 service','500 server error','toomanyrequests','429 client error'))

def safe_image_restart(root,status):
    import json,pathlib
    root=pathlib.Path(root)
    error=status.get('error','')
    if status.get('state')!='attention_required' or 'images/create' not in error or not transient_image_error(error):return False
    return safe_preparing_resume(root)

def safe_preparing_resume(root):
    import json,pathlib
    root=pathlib.Path(root)
    pending=[]
    for p in sorted((root/'episodes').glob('*/episode.json')):
        e=json.loads(p.read_text())
        if e.get('phase')!='complete':pending.append(e)
    if len(pending)!=1:return False
    e=pending[0]
    return e.get('phase')=='preparing' and not e.get('steps') and not (root/'environment_states'/e['token']/'state.json').exists() and not (root/'teacher_pending.json').exists()
