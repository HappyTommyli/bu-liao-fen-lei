from cloth_trainer.evaluation import recognition_summary


def test_matrix_unknowns_replay_and_directions_are_separate():
    rows=[dict(true=0,predicted=0,direction='out',training_overlap=False),dict(true=1,predicted=0,direction='back',training_overlap=False),
          dict(true=-1,predicted=1,direction='out',training_overlap=False),dict(true=1,predicted=1,direction='out',training_overlap=True)]
    result=recognition_summary(rows,['A','B'])
    assert result['new']['confusion']==[[1,0],[1,0]] and result['new']['total']==2
    assert result['replay']['confusion']==[[0,0],[0,1]]
    assert result['unknown_labels']==1
    assert result['new_directions']['out']['accuracy']==1 and result['new_directions']['back']['accuracy']==0
    assert recognition_summary([],['A','B'])['new']['accuracy'] is None
