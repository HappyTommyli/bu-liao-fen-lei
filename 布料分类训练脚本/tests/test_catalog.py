from cloth_trainer.catalog import configurations


def test_repetitions_grouped_and_unrun_tests_not_zero():
    models={f'run{i}':dict(configuration_id='gru_by_mean',architecture='gru',signal='by',readout='mean',seed=42+i,
                        model_id=str(i),val_metrics={'macro_f1':.5+i*.1},test_metrics=None,status='completed') for i in range(3)}
    rows=configurations({'models':models})
    assert len(rows)==1 and len(rows[0]['keys'])==3
    assert abs(rows[0]['val_mean']-.6)<1e-8
    assert rows[0]['val_std']>0 and rows[0]['test_mean'] is None
