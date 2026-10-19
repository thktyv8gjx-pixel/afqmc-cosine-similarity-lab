"""Calculate Count/TF-IDF cosine with frozen implementation on own toy text.

No AFQMC data, threshold search, labels, formal predictions or performance scores.
"""
import argparse,hashlib,json,math,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--input',type=Path,default=Path(__file__).with_name('synthetic_example.json'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.input.read_text(encoding='utf-8'));assert data['origin']=='author_constructed_for_delivery_validation'
    assert data['corpus']==['人工智能','人工数学'] and len(data['pairs'])==3
    sys.path.insert(0,str(a.root/'stage03a_internal_experiment/src'))
    from features import fit_vectorizers,normalized_pair_texts,paired_sparse_cosine
    count,tfidf,features,vocab=fit_vectorizers(data['corpus']);ids,left,right=normalized_pair_texts(data['pairs']);rows=[]
    for name,vectorizer in [('count_char2',count),('tfidf_char2',tfidf)]:
        score,zl,zr=paired_sparse_cosine(vectorizer.transform(left),vectorizer.transform(right))
        expected=[1.,1/3 if name=='count_char2' else 1/(1+2*(math.log(3/2)+1)**2),0.]
        for i,s in enumerate(score):
            assert math.isclose(float(s),expected[i],rel_tol=0,abs_tol=1e-14)
            rows.append({'example_id':ids[i],'method':name,'score':float(s),'left_zero':bool(zl[i]),'right_zero':bool(zr[i])})
    output={'passed':True,'origin':data['origin'],'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'corpus_documents':2,'feature_dimension':len(features),'scope':data['scope'],'formal_afqmc_model_used':False,'thresholds_or_performance_metrics_computed':False,'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes((json.dumps(output,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(output,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
