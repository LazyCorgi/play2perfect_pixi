from argparse import Namespace
import json
from pathlib import Path
import pytest
from tools import beam_half
from tools import run_beam_full_curriculum as full
from tools.beam_curriculum_common import save

def test_artifact_checks_reject_missing_and_changed_files(tmp_path):
    (tmp_path/'releases').mkdir()
    p=tmp_path/'model.pth';p.write_bytes(b'weights')
    (tmp_path/'releases/beam_half.json').write_text(json.dumps({'files':{'model.pth':beam_half.sha256(p)}}))
    beam_half.verify(tmp_path)
    p.write_bytes(b'other')
    with pytest.raises(ValueError): beam_half.verify(tmp_path)
    p.unlink()
    with pytest.raises(ValueError): beam_half.verify(tmp_path)

def test_historical_first_stage_selection_and_later_selection(tmp_path):
    for epoch in (100,200,300,400,500):
        (tmp_path/f'last_beam_ep_{epoch}_rew_0.pth').touch()
    assert [int(p.name.split('_ep_')[1].split('_')[0]) for p in full.candidates(tmp_path,90)]==[100,400,500]
    assert [int(p.name.split('_ep_')[1].split('_')[0]) for p in full.candidates(tmp_path,80)]==[100,300,500]

@pytest.mark.parametrize('reject',[None,70])
def test_official_to060_to050_handoffs(tmp_path,monkeypatch,reject):
    monkeypatch.setattr(full,'ROOT',tmp_path)
    checkpoint=tmp_path/'official.pth';checkpoint.write_bytes(b'official')
    monkeypatch.setattr(full,'OFFICIAL_SHA256',full.digest(checkpoint))
    calls=[];previous=[str(checkpoint)]
    def runner(command):
        def value(flag):return command[command.index(flag)+1]
        run_id=value('--run-id');directory=tmp_path/'outputs/progressive_beam'/run_id;directory.mkdir()
        if '--worker' in command:
            percent=int(value('--percent'));calls.append(percent)
            assert value('--checkpoint')==previous[0]
            if percent==reject:
                save(directory/'status.json',dict(status='rejected_validation'));return 0
            model=directory/'selected.pth';model.write_bytes(str(percent).encode());previous[0]=str(model)
            save(directory/'accepted.json',dict(checkpoint=str(model),problem=f'beam_assembly_step1_scale_{percent:03d}',validation_rate=.95,heldout_rate=.95))
            save(directory/'status.json',dict(status='accepted',child_pid=None,selected_checkpoint=str(model)))
        else:
            calls.append('fine')
            assert Path(value('--source-accepted')).parent.name.endswith('scale060')
            save(directory/'status.json',dict(status='completed'))
            save(directory/'best_model.json',dict(checkpoint='final.pth',heldout_rate=.75))
        return 0
    state=full.run_course(Namespace(checkpoint=str(checkpoint),run_id='delivery',total_minutes=360),runner)
    assert calls==([90,80,70] if reject else [90,80,70,65,60,'fine'])
    assert state['status']==('stopped_at_stage' if reject else 'completed')
