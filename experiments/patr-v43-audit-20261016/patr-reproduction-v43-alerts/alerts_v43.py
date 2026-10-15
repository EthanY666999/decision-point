"""Report-only alerts. No stop signal or algorithm mutation."""
def advance(streaks, record):
    top=record.get('first_action_top_share')
    conditions={'first_action_above_90pct': top is not None and top>.9,
                'informative_groups_below_3':record['optimization']['accepted_groups']<3,
                'fence_loop_above_1pct':record.get('fence_loop_fraction',0)>.01}
    events=[]
    for key,condition in conditions.items():
        streaks[key]=streaks.get(key,0)+1 if condition else 0
        threshold=3 if key=='fence_loop_above_1pct' else 10
        if streaks[key]==threshold:events.append({'kind':key,'step':record['step'],'raw_batch':record['raw_batch'],'streak':threshold,'action':'report_only'})
    return events
