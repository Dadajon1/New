import io,os,sys,json,time,shutil,urllib.request
from PIL import Image
from pypdf import PdfReader,PdfWriter
os.chdir('/var/www/faceswap-api')
P=json.load(open('/root/sb_payload.json'))
ref=PdfReader('books/jobs_book.pdf').pages[0]; W=float(ref.mediabox.width); H=float(ref.mediabox.height)
w=PdfWriter()
for i,u in enumerate(P["order"]):
    d=urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"curl/8.5.0"}),timeout=90).read()
    im=Image.open(io.BytesIO(d)).convert("RGB").resize((2080,2080),Image.LANCZOS)
    b=io.BytesIO(); im.save(b,"PDF",resolution=2080*72.0/W,quality=93); w.add_page(PdfReader(io.BytesIO(b.getvalue())).pages[0])
    print("page",i+1,im.size,flush=True)
with open('books/sports_book.pdf','wb') as f: w.write(f)
for lang,pages in P["json"].items():
    json.dump(pages,open(f'json/sports_book_{lang}.json','w'),ensure_ascii=False,indent=1)
c=open('constants.py','rb').read().decode('utf-8')
if '"sports_book"' not in c:
    shutil.copy2('constants.py','/var/www/releases/constants.py.'+time.strftime('%Y%m%d-%H%M'))
    nl='\r\n' if '\r\n' in c else '\n'
    lines=c.split(nl); k=[i for i,l in enumerate(lines) if l.strip().startswith('"jobs_book": [')][0]
    ind=lines[k][:len(lines[k])-len(lines[k].lstrip())]
    lines.insert(k+1, ind+'"sports_book": ['+','.join(str(x) for x in range(1,30,2))+'],')
    open('constants.py','wb').write(nl.join(lines).encode('utf-8'))
print("BUILT",len(PdfReader('books/sports_book.pdf').pages),W,H)
