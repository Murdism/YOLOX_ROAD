import json
from pathlib import Path

ann_path = Path('datasets/road_uk/train/annotations/road_trainval_v1.0.json')
video = '2014-06-26-09-53-12_stereo_centre_02'
frame_key = '1570' #1721'

with ann_path.open() as f:
    d = json.load(f)

entry = d['db'].get(video)
if entry is None:
    print('VIDEO_NOT_FOUND')
    raise SystemExit

frames = entry.get('frames', {})
frame = frames.get(frame_key)
if frame is None:
    # handle zero-padded keys if any
    for k in frames.keys():
        if int(k) == int(frame_key):
            frame_key = k
            frame = frames[k]
            break

if frame is None:
    print('FRAME_NOT_FOUND')
    print('available sample keys:', list(frames.keys())[:20])
    raise SystemExit

all_agent = d['all_agent_labels']
all_action = d['all_action_labels']
all_loc = d['all_loc_labels']

print('video:', video)
print('frame_key:', frame_key)
print('split_ids:', entry.get('split_ids'))
print('num_annos:', len(frame.get('annos', {})))

for anno_id, obj in sorted(frame.get('annos', {}).items(), key=lambda x: x[0]):
    agent_ids = obj.get('agent_ids') or []
    action_ids = obj.get('action_ids') or []
    loc_ids = obj.get('loc_ids') or []

    agent_names = [all_agent[i] if 0 <= i < len(all_agent) else f'UNK_{i}' for i in agent_ids]
    action_names = [all_action[i] if 0 <= i < len(all_action) else f'UNK_{i}' for i in action_ids]
    loc_names = [all_loc[i] if 0 <= i < len(all_loc) else f'UNK_{i}' for i in loc_ids]

    print('---')
    print('anno_id:', anno_id)
    print('tube_uid:', obj.get('tube_uid'))
    print('box:', obj.get('box'))
    print('agent_ids:', agent_ids, '=>', agent_names)
    print('action_ids:', action_ids, '=>', action_names)
    print('loc_ids:', loc_ids, '=>', loc_names)
