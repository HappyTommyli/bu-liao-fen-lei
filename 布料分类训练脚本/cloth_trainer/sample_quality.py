"""Reassess selected sensor channels without rewriting raw acquisition flags."""
import numpy as np


def selected_axis_rows(rows,axes,diagnostics):
    rows=list(rows)
    def number(value):
        try:return float(value)
        except (ValueError,TypeError):return float('nan')
    # Judge the actual feature channel; missing raw provenance is separately
    # handled by the loader's baseline verification, without changing dBy.
    values=np.array([[number(row.get(f'dB{axis}_uT',row.get(f'B{axis}_uT'))) for axis in axes] for row in rows],dtype=float).reshape(-1,len(axes))
    finite=np.isfinite(values).all(axis=1);spikes=np.zeros(len(rows),dtype=bool)
    if len(rows)>2:
        spikes[1:-1]=(np.linalg.norm(values[1:-1]-values[:-2],axis=1)>=1000)&(np.linalg.norm(values[1:-1]-values[2:],axis=1)>=1000)&(np.linalg.norm(values[2:]-values[:-2],axis=1)<=250)
    diagnostics.update(version='selected-axis-v1',axes=list(axes),original_invalid=0,restored_other_axis_spike=0,selected_axis_spikes=int(spikes.sum()),selected_axis_nonfinite=int((~finite).sum()))
    for i,source in enumerate(rows):
        row=dict(source)
        valid=str(row.get('sample_valid','1')).strip().lower() not in ('0','false','no')
        diagnostics['original_invalid']+=int(not valid)
        if not valid and row.get('invalid_reason')=='isolated_spike' and finite[i] and not spikes[i]:
            row['sample_valid']='1';row['invalid_reason']=''
            diagnostics['restored_other_axis_spike']+=1
        if not finite[i] or spikes[i]:
            row['sample_valid']='0'
            row['invalid_reason']='selected_axis_nonfinite' if not finite[i] else 'selected_axis_isolated_spike'
        yield row
