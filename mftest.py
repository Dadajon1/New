import asyncio, os, sys, base64, io, time
os.chdir('/var/www/faceswap-api'); sys.path.insert(0, '/var/www/faceswap-api')
from dotenv import load_dotenv; load_dotenv()
import rp_handler as r
from pdf2image import convert_from_path
from PIL import Image
PAGES = [int(x) for x in sys.argv[2].split(',')]  # 1-based
SRC = sys.argv[1]
async def main():
    src = base64.b64encode(open(SRC, 'rb').read()).decode()
    ssl = r.make_ssl_context()
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async def one(p):
            page = convert_from_path('books/magic_flower.pdf', dpi=300, first_page=p, last_page=p)[0]
            orig = page.copy()
            idx, b, err = await r.process_one_page(session=s, idx=p-1, page=page, source_b64=src,
                prompt="Clear face, distortion free image and even eyes", model_type="CARTOON_LITE",
                face_strength=0.9, style_strength=0.6, ignore_indices=set(), ssl_ctx=ssl,
                masked_pdf_path=__import__('pathlib').Path('masked/magic_flower.pdf'), grow_mask=10,
                is_magic_flower=True)
            out = Image.open(io.BytesIO(b)).convert('RGB').resize((600, 600))
            o = orig.convert('RGB').resize((600, 600))
            S = Image.new('RGB', (1200, 600)); S.paste(o, (0, 0)); S.paste(out, (600, 0))
            S.save(f'/tmp/mf_p{p}.jpg', quality=85); print('DONE', p, err, flush=True)
        await asyncio.gather(*(one(p) for p in PAGES))
asyncio.run(main())
