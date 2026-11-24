import csv
import json
import pytest
from cloth_trainer.config import DataConfig, TrainConfig
from cloth_trainer.data import load_record, protocol_for, matches_protocol
from test_data import fixture_csv

RELATIVE=DataConfig.from_dict(dict(contact_rule="relative-035-v1",preprocess_version="abs-v3"))


def shifted_capture(tmp_path, sequence, origin):
    p = fixture_csv(tmp_path, i=sequence)
    mp = p.parent / 'metadata.json'
    meta = json.loads(mp.read_text())
    meta['sequence'] = sequence
    mp.write_text(json.dumps(meta))
    with p.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row['motor_x_pulses'] = str(float(row['motor_x_pulses']) + origin * 250)
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    return p


@pytest.mark.parametrize('origin,bounds', [(-24.348, (19.652, 119.652)), (26, (70, 170)), (0, (44, 144))])
def test_relative_crops_preserve_raw_coordinates_and_sources(tmp_path, origin, bounds):
    p = shifted_capture(tmp_path, 35, origin)
    before = (p.read_bytes(), (p.parent/'metadata.json').read_bytes())
    r = load_record(p, RELATIVE, signals=('by',))
    assert all(t.bounds == pytest.approx(bounds) for t in r.trials)
    assert r.trials[0].raw_x[0] == pytest.approx(origin)
    assert all(len(t.values) == 1043 for t in r.trials)
    assert r.diagnostics['contact_rule']['back_travel_mm'] == [11., 111.]
    assert before == (p.read_bytes(), (p.parent/'metadata.json').read_bytes())


def test_earlier_and_serialized_old_models_keep_fixed_x(tmp_path):
    early = load_record(shifted_capture(tmp_path/'early', 34, 0), RELATIVE, signals=('by',))
    assert [t.bounds for t in early.trials] == [(45, 142), (16, 114)]
    old = TrainConfig.from_dict({'data': {'preprocess_version': 'abs-v3'}})
    r = load_record(shifted_capture(tmp_path/'old', 35, 0), old.data, signals=('by',))
    assert [t.bounds for t in r.trials] == [(45, 142), (16, 114)]


def test_mixed_contact_lengths_have_valid_protocol(tmp_path):
    a = load_record(shifted_capture(tmp_path/'a', 34, 0), RELATIVE, signals=('by',))
    b = load_record(shifted_capture(tmp_path/'b', 36, 26), RELATIVE, signals=('by',))
    protocol = protocol_for([a, b], RELATIVE)
    assert matches_protocol(a, protocol) and matches_protocol(b, protocol)


def test_conflicting_sequence_is_not_silently_cropped(tmp_path):
    p = shifted_capture(tmp_path, 35, 0)
    mp = p.parent/'metadata.json'
    meta = json.loads(mp.read_text());meta['sequence'] = 34;mp.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='序号'):
        load_record(p, RELATIVE, signals=('by',))


def test_missing_outgoing_start_cannot_be_replaced_with_new_origin(tmp_path):
    p = shifted_capture(tmp_path, 36, 0)
    with p.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f);fields = reader.fieldnames
        rows = [r for r in reader if r['stage'] != 'step_003' or float(r['motor_x_pulses']) >= 10*250]
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields);writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError, match='155|起点|往返'):
        load_record(p, RELATIVE, signals=('by',))
