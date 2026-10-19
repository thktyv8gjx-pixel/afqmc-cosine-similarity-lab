"""Document gate: no raw Dev archive or model execution. Uses bundled Python.

Identity expectations stay in local_only/identity.json (never publish it).
Structural checks supplement mandatory human inspection of every rendered page.
"""
import hashlib,json,re,unicodedata
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from docx import Document
from pypdf import PdfReader
from PIL import Image,ImageChops
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;L=OUT/'local_only'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','r':'http://schemas.openxmlformats.org/package/2006/relationships'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def flat(t):return re.sub(r'\s|[\u00ad\u200b\ufeff]','',t)
def mathtext(x):return ''.join(x.xpath('.//m:t/text()',namespaces=NS))
def collision_free(r):
    expressions=[flat(mathtext(m)) for m in r.findall('.//m:oMath',NS)]
    return not any('cos(xi,yi)' in x for x in expressions)
def main():
    doc=next(L.glob('*_实验报告1_*.docx'));pdf=doc.with_suffix('.pdf');d=Document(doc);reader=PdfReader(pdf)
    export=read(L/'word_export_final.json');build=read(L/'build_record.json')
    assert export['source_sha256_before']==sha(doc)==export['source_sha256']==build['docx_sha256']
    assert sha(pdf)==export['pdf_sha256'] and len(reader.pages)==export['pages']
    assert export['engine']=='Microsoft Word ExportAsFixedFormat' and export['source_saved'] is False
    text='\n'.join(p.extract_text() for p in reader.pages);ft=flat(text)
    (L/'final_pdf_extracted.txt').write_text(text,encoding='utf-8')
    identity=read(L/'identity.json');cover=d.tables[0]
    for row,val in zip(cover.rows[2:7],identity['values']):assert row.cells[1].text==val and flat(val) in ft
    assert cover.rows[7].cells[1].text=='【待确认：实际提交日期】'
    assert '待填' not in ft and ft.count('待确认')==1 and '通信工程学院' in ft
    with ZipFile(ROOT/'实验报告模板.docx') as original,ZipFile(doc) as z:
        root=E.fromstring(z.read('word/document.xml'));core=E.fromstring(z.read('docProps/core.xml'))
        assert sha(ROOT/'实验报告模板.docx')==build['template_sha256']
        allowed={'word/document.xml','word/_rels/document.xml.rels','[Content_Types].xml','docProps/core.xml','docProps/app.xml'}
        protected=[p for p in original.namelist() if p not in allowed]
        assert all(original.read(p)==z.read(p) for p in protected)
        assert core.find('{http://purl.org/dc/elements/1.1/}creator').text==identity['values'][0]
        assert core.find('{http://purl.org/dc/elements/1.1/}title').text==identity['title']
        assert core.find('{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy').text==identity['values'][0]
        app=E.fromstring(z.read('docProps/app.xml'))
        assert int(app.find('{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Pages').text)==len(reader.pages)
        original_root=E.fromstring(original.read('word/document.xml'))
        assert E.tostring(root.find('.//w:sectPr',NS))==E.tostring(original_root.find('.//w:sectPr',NS))
        for p in root.findall('.//w:p',NS):
            t=''.join(p.xpath('.//w:t/text()',namespaces=NS))
            if t.strip() and not p.findall('.//m:oMath',NS) and not p.findall('.//w:drawing',NS):assert flat(t) in ft,t[:35]
        assert collision_free(root)
        old=next((ROOT/'stage04_final_report').glob('*.docx'))
        with ZipFile(old) as oz:assert not collision_free(E.fromstring(oz.read('word/document.xml'))),'Regression must expose original symbol collision'
        mathlist=[mathtext(m) for m in root.findall('.//m:oMath',NS)]
        assert any('xi(1)=ϕ(N(si(1)))' in x and 'xi(2)=ϕ(N(si(2)))' in x for x in mathlist)
        assert any('cos(xi(1),xi(2))' in x and '≥τ' in x for x in mathlist)
        assert any('yi∈{0,1}' in x for x in mathlist)
        assert any('z(s)=' in x and '>0' in x for x in mathlist)
        pdfmath=unicodedata.normalize('NFKC',ft)
        assert 'cos(xi,yi)' not in pdfmath
        assert 'cos(xi(1),xi(2))≥τ' in pdfmath
        for word in ['Acc','Specificity','BA']:
            runs=[r for r in root.findall('.//m:r',NS) if r.find('m:t',NS) is not None and r.find('m:t',NS).text==word]
            def upright(r):
                style=r.find('m:rPr/m:sty',NS)
                return r.find('m:rPr/m:nor',NS) is not None or (style is not None and style.get('{'+NS['m']+'}val')=='p')
            assert runs and all(upright(r) for r in runs),word
        assert len(root.findall('.//m:f',NS))>=10 and len(root.findall('.//m:sSubSup',NS))>=4
        xmltext=''.join(root.xpath('//w:t/text()|//m:t/text()',namespaces=NS))
        assert not re.search(r'\\(?:frac|operatorname|begin|hat|sqrt)',xmltext)
        rels=E.fromstring(z.read('word/_rels/document.xml.rels'))
        expectedimages={sha(p) for p in (L/'print_figures').glob('*.png')}
        actualimages={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
        assert len(expectedimages)==6 and expectedimages==actualimages
        urls=[r.get('Target') for r in rels if r.get('Type','').endswith('/hyperlink')]
        pdfurls={str(a.get_object().get('/A',{}).get('/URI','')) for page in reader.pages for a in page.get('/Annots',[])}
        assert len(urls)==4 and set(urls)<=pdfurls
    # Native text and six figure XObjects, not one raster image for every page.
    assert all(len(page.extract_text())>200 for page in reader.pages)
    assert sum(len(page.images) for page in reader.pages)==6
    assert 'Word' in str(reader.metadata)
    perpage=[]
    for i,page in enumerate(reader.pages,1):
        a=L/f'word_render/page-{i}.png';b=L/f'pdf_render/page-{i:02d}.png'
        ia,ib=Image.open(a),Image.open(b)
        assert ia.width>=1240 and ia.height>=1753 and ia.size==ib.size
        assert ImageChops.difference(ia.convert('RGB'),ib.convert('RGB')).getbbox() is None
        assert re.search(r'—\s*'+str(i)+r'\s*—',page.extract_text()),('page number',i)
        perpage.append({'page':i,'dpi':150,'width_px':ia.width,'height_px':ia.height,'word_render_sha256':sha(a),'pdf_render_sha256':sha(b),'word_pdf_pixel_equal':True,'searchable_characters':len(page.extract_text()),'image_count':len(page.images)})
    synthetic=read(L/'synthetic_scores.json');attempt=read(L/'synthetic_inference_attempt.json')
    assert synthetic['preregistration_sha256']==sha(L/'synthetic_preregistration.json')==attempt['preregistration_sha256']
    assert attempt['status']=='completed' and len(synthetic['rows'])==8
    assert attempt['started_utc']>read(L/'synthetic_preregistration.json')['registered_utc']
    for x in synthetic['rows']:
        assert f'{x["score"]:.6f}' in ft and x['prediction']==int(x['score']>=x['threshold'])
    # Preserve all original files, including earlier audit reports and full local draft.
    for relative,before in read(L/'baseline_private.json')['files'].items():assert sha(ROOT/relative)==before['sha256'],relative
    result={'passed':True,'pages':len(reader.pages),'docx_bytes':doc.stat().st_size,'pdf_bytes':pdf.stat().st_size,'docx_sha256':sha(doc),'pdf_sha256':sha(pdf),'template_sha256':build['template_sha256'],'preserved_template_parts':len(protected),'authorized_metadata_exception':'core.xml: title/author/timestamps; app.xml: remove stale template statistics, update actual pages','native_omml_count':len(root.findall('.//m:oMath',NS)),'native_display_equations':len(root.findall('.//m:oMathPara',NS)),'symbol_collision_regression_old_fails_new_passes':True,'vector_sentence_side_correspondence':True,'five_identity_fields_verified_locally':True,'date_pending':True,'institution_original_retained':True,'all_normal_paragraphs_searchable_in_pdf':True,'all_image_hashes_match':True,'pdf_figure_images':6,'hyperlinks_verified':len(urls),'synthetic_registered_pairs':4,'no_raw_dev_read':True,'pages_evidence':perpage}
    (OUT/'document_gate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='pages_evidence'},ensure_ascii=False))
if __name__=='__main__':main()
