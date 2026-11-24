"""Evaluation scopes: never pool different checkpoints or unknown truth labels."""
import numpy as np

def classification_metrics(rows,labels,checkpoint=5):
    matrix=np.zeros((len(labels),len(labels)),dtype=int)
    for r in rows:
        if r['true']>=0:
            predicted=r['predicted'] if 'predicted' in r else r['checkpoints'][checkpoint]['predicted']
            matrix[r['true'],predicted]+=1
    d=matrix.diagonal();recall=np.divide(d,matrix.sum(1),out=np.zeros(len(labels)),where=matrix.sum(1)>0)
    precision=np.divide(d,matrix.sum(0),out=np.zeros(len(labels)),where=matrix.sum(0)>0)
    f1=np.divide(2*recall*precision,recall+precision,out=np.zeros(len(labels)),where=recall+precision>0);total=int(matrix.sum())
    return dict(accuracy=float(d.sum()/total) if total else None,macro_f1=float(f1.mean()) if total else None,total=total,correct=int(d.sum()),confusion=matrix.tolist(),
                per_class=[dict(label=c,precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(matrix[i].sum())) for i,c in enumerate(labels)])


def recognition_summary(rows,labels):
    new=[r for r in rows if not r.get('training_overlap',False)];replay=[r for r in rows if r.get('training_overlap',False)]
    return dict(new=classification_metrics(new,labels),replay=classification_metrics(replay,labels),all=classification_metrics(rows,labels),unknown_labels=sum(r['true']<0 for r in rows),
                new_directions={d:classification_metrics([r for r in new if r['direction']==d],labels) for d in ('out','back')},
                replay_directions={d:classification_metrics([r for r in replay if r['direction']==d],labels) for d in ('out','back')})


def summarize_configs(meta):
    groups={}
    for result in meta['models'].values():groups.setdefault(result['configuration_id'],{}).setdefault(str(result['seed']),[]).append(result)
    summaries={}
    for key,seeds in groups.items():
        by_seed={}
        for seed,results in seeds.items():
            if len(results)!=5 or {r['fold'] for r in results}!=set(range(5)) or any(r['test_metrics'] is None for r in results):continue
            rows=[row for r in results for row in r['test_predictions']]
            if len({r['id'] for r in rows})!=len(rows):raise ValueError('折外样本重复，不能汇总')
            folds=[r['test_metrics'] for r in sorted(results,key=lambda r:r['fold'])];metrics=classification_metrics(rows,meta['labels'])
            confusion=np.array(metrics['confusion']);pairs=sorted([(int(confusion[i,j]),meta['labels'][i],meta['labels'][j]) for i in range(len(confusion)) for j in range(len(confusion)) if i!=j and confusion[i,j]],reverse=True)
            by_seed[seed]=dict(oof_metrics=metrics,fold_metrics=folds,fold_statistics=dict(accuracy_mean=float(np.mean([m['accuracy'] for m in folds])),accuracy_std=float(np.std([m['accuracy'] for m in folds])),f1_mean=float(np.mean([m['macro_f1'] for m in folds])),f1_std=float(np.std([m['macro_f1'] for m in folds]))),
                              direction_metrics={d:classification_metrics([r for r in rows if r['direction']==d],meta['labels']) for d in ('out','back')},
                              quality_metrics={q:classification_metrics([r for r in rows if r.get('quality','normal')==q],meta['labels']) for q in ('normal','suggest_review','suggest_recollect')},
                              main_confusions=[dict(count=n,true=a,predicted=b) for n,a,b in pairs[:15]],
                              train_val_gap=[dict(fold=r['fold'],accuracy_gap=r['train_metrics']['accuracy']-r['val_metrics']['accuracy'],f1_gap=r['train_metrics']['macro_f1']-r['val_metrics']['macro_f1']) for r in results])
        if by_seed:summaries[key]=dict(by_seed=by_seed,note='各随机种子分别汇总一次折外预测；不合并成新增独立样本。')
    return summaries
