import os
import ssl
import base64
import asyncio
import aiohttp
import gc
from io import BytesIO
from pathlib import Path
from PIL import Image
import json
from pdf2image import convert_from_path, pdfinfo_from_path
from dotenv import load_dotenv
from typing import Literal
import random
from limiter import rate_limiter
from openai import AsyncOpenAI
import io
from utils import get_logger, load_image_from_local, save_final_pdf, job_context,upload_to_r2_async
from constants import ignore_index_map, GIRL_BOOK_FIX_PROMPT, GIRL_BOOK_FIX_PROMPT_OWN_HAIR, GPT_FIX_PAGES, HAIR_CHECK_MODEL, HAIR_CHECK_PROMPT
from constants import RESULT_DOWNLOAD_BACKOFF,RESULT_DOWNLOAD_RETRIES,PAYLOAD_JPEG_QUALITY,PAGE_BATCH_SIZE,MAX_CONCURRENT_PAGES,GPT_IMAGE_QUALITY,GPT_IMAGE_SIZE,GPT_MAX_RETRIES,GPT_MODEL,GPT_RETRY_BACKOFF
from helper import make_ssl_context,get_api_semaphore,download_with_retry,get_pdf_path,load_masked_page_b64
from constants import NON_RETRYABLE,RETRYABLE
import uuid
import random


logger = get_logger()
load_dotenv()
async def faceswap_runpod(
    session: aiohttp.ClientSession,
    source_b64: str,
    target_b64: str,
    masked_b64: str | None,
    prompt: str,
    face_strength: float,
    style_strength: float,
    refine: bool = False,
    grow_mask: int = 0,
    seed: int = 242134,
    output_format: str = "jpeg",
    ssl_ctx: ssl.SSLContext | None = None,
    poll_interval: float = 4.0,
    max_poll_seconds: float = 600.0,
) -> tuple[str, str | None]:
    seed = random.randint(0, 2**32 - 1)
    api_key     = os.getenv("RUNPOD_API_KEY")
    endpoint_id = os.getenv("RUNPOD_REFINE_ID") if refine else os.getenv("RUNPOD_NON_REFINE_ID")

    job_prefix = f"jobs/{uuid.uuid4()}"
    try:
        upload_coros = [
            upload_to_r2_async(base64.b64decode(source_b64), key=f"{job_prefix}/source.jpg"),
            upload_to_r2_async(base64.b64decode(target_b64), key=f"{job_prefix}/target.jpg"),
        ]
        if masked_b64:
            upload_coros.append(
                upload_to_r2_async(base64.b64decode(masked_b64), key=f"{job_prefix}/masked.jpg")
            )

        upload_results = await asyncio.gather(*upload_coros)
        source_url = upload_results[0]
        target_url = upload_results[1]
        masked_url = upload_results[2] if masked_b64 else None

    except Exception as e:
        return target_b64, f"R2 upload failed: {e}"

    submit_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "input": {
            "source_img": source_url,
            "target_img": target_url,
            "masked_img": masked_url,
            "prompt": prompt,
            "face_strength": face_strength,
            "steps": 9,
            "cfg": 1.6,
            "style_strength": style_strength,
            "refine": refine,
            "grow_mask": grow_mask,
            "seed": seed,
            "output_format": output_format,
            "output_quality": 100,
            "request_id": uuid.uuid4().hex,
        }
    }

    job_id: str | None = None
    last_error: str | None = None

    for attempt in range(1, RESULT_DOWNLOAD_RETRIES + 1):
        try:
            await asyncio.sleep(random.uniform(0.5, 2.0))
            await rate_limiter.acquire()

            async with get_api_semaphore():
                async with session.post(
                    submit_url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    body = await resp.read()
                    status = resp.status

            if status in (200, 201, 202):
                data = json.loads(body)
                job_id = data.get("id")
                if job_id:
                    break
                last_error = "RunPod submit: no job id in response"
            else:
                text = body.decode("utf-8", errors="replace")[:200]
                if status in NON_RETRYABLE or status == 400:
                    return target_b64, f"RunPod submit HTTP {status}: {text}"
                last_error = f"RunPod submit HTTP {status}: {text}"
                logger.warning("Retryable submit error", extra={"status": status, "attempt": attempt})

        except aiohttp.ClientSSLError as e:
            return target_b64, f"SSL error: {e}"
        except aiohttp.ClientConnectorError as e:
            return target_b64, f"Connection error: {e}"
        except asyncio.TimeoutError:
            last_error = f"Submit timed out (attempt {attempt})"
        except Exception as e:
            last_error = f"Unexpected submit error: {e}"

        if attempt < RESULT_DOWNLOAD_RETRIES:
            await asyncio.sleep(RESULT_DOWNLOAD_BACKOFF * (2 ** (attempt - 1)))

    if not job_id:
        return target_b64, f"Submit failed after {RESULT_DOWNLOAD_RETRIES} attempts. Last error: {last_error}"

    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    elapsed = 0.0
    last_sdata: dict = {}

    while elapsed < max_poll_seconds:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

        try:
            async with session.get(
                status_url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as sresp:
                sbody = await sresp.read()
                if sresp.status == 404:
                    return target_b64, f"RunPod: job {job_id} not found"
        except (aiohttp.ClientError, asyncio.TimeoutError):
            continue

        try:
            sdata = json.loads(sbody)
        except json.JSONDecodeError:
            continue

        last_sdata = sdata
        job_status = sdata.get("status")
        output = sdata.get("output")
        output_dict = output if isinstance(output, dict) else {}

        # Execution timeout / faulty GPU (status == FAILED, error == "executionTimeout exceeded")
        if job_status == "FAILED":
            error_text = sdata.get("error")

            if error_text == "executionTimeout exceeded":
                worker_id = sdata.get("workerId") or output_dict.get("x-worker-id")
                logger.warning(
                    "RunPod execution timeout / faulty GPU",
                    extra={"job_id": job_id, "worker_id": worker_id},
                )
                if worker_id:
                    asyncio.ensure_future(_delete_runpod_worker(session, worker_id, api_key))
                return target_b64, f"RunPod: job {job_id} execution timeout on worker {worker_id}"

            err = error_text or output or sdata
            logger.error("RunPod job failed", extra={"job_id": job_id, "error": str(err)})
            return target_b64, f"RunPod FAILED: {err}"

        # Legacy/alternate timeout status, kept as a fallback
        if job_status == "TIMED_OUT":
            worker_id = sdata.get("workerId") or output_dict.get("x-worker-id")
            logger.warning("RunPod job timed out", extra={"job_id": job_id, "worker_id": worker_id})
            if worker_id:
                asyncio.ensure_future(_delete_runpod_worker(session, worker_id, api_key))
            return target_b64, f"RunPod: job {job_id} timed out on worker {worker_id}"

        if job_status == "CANCELLED":
            return target_b64, f"RunPod: job {job_id} was cancelled"

        if job_status == "COMPLETED":
            logger.info("RunPod job completed", extra={"job_id": job_id, "output": str(output)[:200]})

            if not output:
                return target_b64, "RunPod: COMPLETED but no output field"
            if not isinstance(output, dict):
                return target_b64, f"RunPod: unexpected output type {type(output).__name__}: {str(output)[:200]}"

            # Model init failed (output.status == "failed", output.status_code == 503)
            if output.get("status") == "failed":
                status_code = output.get("status_code", "?")
                error_msg   = output.get("error", "unknown error")
                logger.error("RunPod handler error", extra={"status": status_code, "error": error_msg})
                if status_code == 503:
                    worker_id = output.get("x-worker-id")
                    if worker_id:
                        asyncio.ensure_future(_delete_runpod_worker(session, worker_id, api_key))
                return target_b64, f"RunPod handler error {status_code}: {error_msg}"

            image_url = output.get("image")
            if not image_url:
                return target_b64, f"RunPod: no 'image' key in output: {str(output)[:200]}"

            content = await download_with_retry(session, image_url, ssl_ctx)
            if content:
                return base64.b64encode(content).decode("utf-8"), None
            return target_b64, "RunPod: output image URL download failed"

        # job_status is IN_QUEUE / IN_PROGRESS / unknown -> keep polling

    # Poll loop exhausted: job never reached a terminal status (true "hang") 
    hung_worker_id = last_sdata.get("workerId") if isinstance(last_sdata, dict) else None
    logger.warning(
        "RunPod job exceeded max_poll_seconds without completing",
        extra={"job_id": job_id, "worker_id": hung_worker_id, "last_status": last_sdata.get("status")},
    )

    asyncio.ensure_future(_cancel_runpod_job(session, endpoint_id, job_id, api_key))

    if hung_worker_id:
        asyncio.ensure_future(_delete_runpod_worker(session, hung_worker_id, api_key))

    return target_b64, f"RunPod: job {job_id} did not complete within {max_poll_seconds}s"


async def _cancel_runpod_job(
    session: aiohttp.ClientSession,
    endpoint_id: str,
    job_id: str,
    api_key: str,
) -> None:
    """Best-effort cancel of a hung RunPod job so it stops holding a worker."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/cancel/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.post(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            body = await resp.text()
            if resp.status in (200, 201, 202):
                logger.info("Cancelled hung RunPod job", extra={"job_id": job_id})
            else:
                logger.warning(
                    "Cancel hung job unexpected status",
                    extra={"job_id": job_id, "status": resp.status, "body": body[:200]},
                )
    except Exception as e:
        logger.warning("Cancel hung job failed", extra={"job_id": job_id, "error": str(e)})

async def _delete_runpod_worker(
    session: aiohttp.ClientSession,
    worker_id: str,
    api_key: str,
) -> None:
    """Best-effort deletion of a broken RunPod worker pod."""
    url = f"https://rest.runpod.io/v1/pods/{worker_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.delete(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 204:
                logger.info("Deleted broken RunPod worker", extra={"worker_id": worker_id})
            else:
                body = await resp.text()
                logger.warning("Worker delete unexpected status", extra={"status": resp.status, "body": body[:200]})
    except Exception as e:
        logger.warning("Worker delete failed", extra={"worker_id": worker_id, "error": str(e)})
   
def _log_gpt_usage(response, stage: str) -> None:
    """Log OpenAI image token usage so per-book GPT cost can be measured."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        data = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        logger.info("GPT usage", extra={"stage": stage, "usage": data})
    except Exception as e:
        logger.warning(f"Could not log GPT usage: {e}")


async def faceswap_gpt(
    source_b64: str,
    target_b64: str,
    prompt: str,
) -> tuple[str, str | None]:
    """
    Face-swap using OpenAI's gpt-image-2 edit endpoint.
    Returns (result_b64, error_str | None).
    Falls back to target_b64 on any failure.
    """
    api_key = os.getenv("GPT_API")
    if not api_key:
        return target_b64, "GPT_API env var not set"

    def _compress_b64_to_bytes(b64: str, max_size=(1024, 1024), quality=98) -> io.BytesIO:
        raw = base64.b64decode(b64)
        with Image.open(BytesIO(raw)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(max_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            buf.seek(0)
            return buf

    last_error: str | None = None
    client = AsyncOpenAI(api_key=api_key)

    for attempt in range(GPT_MAX_RETRIES):
        try:
            await asyncio.sleep(random.uniform(0.2, 0.8))

            async with get_api_semaphore():
                # Re-compress on every attempt so BytesIO cursors are fresh
                cartoon_buf = await asyncio.get_running_loop().run_in_executor(
                    None, _compress_b64_to_bytes, target_b64
                )
                face_buf = await asyncio.get_running_loop().run_in_executor(
                    None, _compress_b64_to_bytes, source_b64
                )

                response = await client.images.edit(
                    model=GPT_MODEL,
                    image=[
                        ("base.jpg",  cartoon_buf, "image/jpeg"),   # target first
                        ("face.jpg",  face_buf,    "image/jpeg"),   # source face second
                    ],
                    prompt=prompt,
                    size=GPT_IMAGE_SIZE,
                    quality=GPT_IMAGE_QUALITY,
                    output_format="jpeg",
                    n=1,
                )

            result_b64 = response.data[0].b64_json
            _log_gpt_usage(response, "faceswap_gpt")
            if result_b64:
                return result_b64, None
            
            if result_b64 == target_b64:
                logger.warning("Didnt swap page")

            last_error = "gpt-image-2 returned empty b64_json"

        except Exception as e:
            last_error = f"gpt-image-2 attempt {attempt + 1} error: {e}"
            logger.warning(last_error, extra={"attempt": attempt + 1})
            backoff = GPT_RETRY_BACKOFF * (2 ** attempt)
            await asyncio.sleep(backoff)

    return target_b64, f"All {GPT_MAX_RETRIES} GPT attempts failed. Last: {last_error}"


async def detect_hair_length(source_b64: str) -> str:
    """One cheap vision call per book: 'short' or 'long'. Defaults to 'short' (template hair) on any failure."""
    api_key = os.getenv("GPT_API")
    if not api_key:
        return "short"
    try:
        with Image.open(BytesIO(base64.b64decode(source_b64))) as img:
            img = img.convert("RGB")
            img.thumbnail((512, 512))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=HAIR_CHECK_MODEL,
            max_tokens=3,
            temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": HAIR_CHECK_PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
            ]}],
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        return "long" if answer.startswith("long") else "short"
    except Exception as e:
        logger.warning(f"Hair length check failed, using template hair: {e}")
        return "short"


def _scene_distance(a_b64: str, b_b64: str) -> float:
    """Mean abs difference of 32x32 grayscale thumbnails (0..255). Small = same scene/composition."""
    def _thumb(b64: str):
        with Image.open(BytesIO(base64.b64decode(b64))) as im:
            return list(im.convert("L").resize((32, 32)).getdata())
    try:
        a, b = _thumb(a_b64), _thumb(b_b64)
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)
    except Exception:
        return 999.0


async def fix_face_artifacts_gpt(
    image_b64: str,
    prompt: str,
    reference_b64: str | None = None,
    child_b64: str | None = None,
) -> tuple[str, str | None]:
    """
    Post-processing correction pass using OpenAI's gpt-image-2 edit endpoint.
    Used only for "_girl" books to fix hair/neck deformation, facial-structure
    irregularities, and hand issues, and to enhance overall image quality.
    Returns (result_b64, error_str | None).
    Falls back to the original image_b64 on any failure.
    """
    api_key = os.getenv("GPT_API")
    if not api_key:
        return image_b64, "GPT_API env var not set"

    def _compress_b64_to_bytes(b64: str, max_size=(1024, 1024), quality=98) -> io.BytesIO:
        raw = base64.b64decode(b64)
        with Image.open(BytesIO(raw)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(max_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            buf.seek(0)
            return buf

    last_error: str | None = None
    client = AsyncOpenAI(api_key=api_key)
    
    for attempt in range(GPT_MAX_RETRIES):
        try:
            await asyncio.sleep(random.uniform(0.2, 0.8))

            async with get_api_semaphore():
                image_buf = await asyncio.get_running_loop().run_in_executor(
                    None, _compress_b64_to_bytes, image_b64
                )
                images = [("image.jpg", image_buf, "image/jpeg")]
                if reference_b64:
                    # Second image = the original book illustration (before face swap):
                    # GPT restores clothing, headwear and hairstyle from it.
                    ref_buf = await asyncio.get_running_loop().run_in_executor(
                        None, _compress_b64_to_bytes, reference_b64
                    )
                    images.append(("original.jpg", ref_buf, "image/jpeg"))
                if reference_b64 and child_b64:
                    # Third image = the customer's photo, used only to decide the hair rule.
                    child_buf = await asyncio.get_running_loop().run_in_executor(
                        None, _compress_b64_to_bytes, child_b64
                    )
                    images.append(("child.jpg", child_buf, "image/jpeg"))

                response = await client.images.edit(
                    model=GPT_MODEL,
                    image=images,
                    prompt=prompt,
                    size=GPT_IMAGE_SIZE,
                    quality=GPT_IMAGE_QUALITY,
                    output_format="jpeg",
                    n=1,
                )

            result_b64 = response.data[0].b64_json
            _log_gpt_usage(response, "fix_face_artifacts_gpt")
            if result_b64:
                return result_b64, None

            last_error = "gpt-image-2 fix pass returned empty b64_json"
            
        except Exception as e:
            last_error = f"gpt-image-2 fix attempt {attempt + 1} error: {e}"
            logger.warning(last_error, extra={"attempt": attempt + 1})
            backoff = GPT_RETRY_BACKOFF * (2 ** attempt)
            await asyncio.sleep(backoff)

    return image_b64, f"All {GPT_MAX_RETRIES} GPT fix attempts failed. Last: {last_error}"


async def process_one_page(
    session: aiohttp.ClientSession,
    idx: int,
    page: Image.Image,
    source_b64: str,
    prompt: str,
    model_type: str,
    face_strength: float,
    style_strength: float,
    ignore_indices: set[int],
    ssl_ctx: ssl.SSLContext,
    masked_pdf_path: Path | None = None,
    grow_mask: int = 0,
    refine: bool = False,
    seed: int = 242134,
    is_regenerate: bool = False,
    is_magic_flower: bool = False,
    is_girl_book: bool = False,
    gpt_fix_pages: set[int] | None = None,
    girl_fix_prompt: str = GIRL_BOOK_FIX_PROMPT,
    child_b64_for_fix: str | None = None,
) -> tuple[int, bytes, str | None]:

    buf = BytesIO()
    page.save(buf, format="JPEG", quality=PAYLOAD_JPEG_QUALITY, optimize=True)
    buf.seek(0)
    original_page_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    page.close()
    del page
    gc.collect()

    if idx in ignore_indices:
        logger.info("Page skipped", extra={"page": idx + 1})
        return idx, base64.b64decode(original_page_b64), None

    MAGIC_FLOWER_GPT_SKIP: set[int] = set()  # page 18 (deer) now goes through GPT too: it kept short hair and a bent neck
    masked_b64: str | None = None
    if model_type == "CARTOON_LITE" and masked_pdf_path is not None:
        loop = asyncio.get_running_loop()
        masked_b64 = await loop.run_in_executor(
            None, load_masked_page_b64, masked_pdf_path, idx
        )

    logger.info("Processing page", extra={"page": idx + 1})

    final_b64: str | None = None
    error: str | None = None
    stage1_b64: str | None = None

    if is_girl_book:
        # Girl-book pipeline: RunPod swap only, then GPT fix-artifacts pass.
        logger.info("→ Girl-book: Stage 1 RunPod", extra={"page": idx + 1})
        stage1_b64, stage1_error = await faceswap_runpod(
            session=session,
            source_b64=source_b64,
            target_b64=original_page_b64,
            masked_b64=masked_b64,
            prompt=prompt,
            face_strength=face_strength,
            style_strength=style_strength,
            ssl_ctx=ssl_ctx,
            refine=refine,
            grow_mask=grow_mask,
            seed=seed,
        )

        runpod_swapped = (
            not stage1_error
            and stage1_b64 is not None
            and stage1_b64 != original_page_b64
        )

        if not runpod_swapped:
            logger.warning(
                "Girl-book: RunPod did not swap, skipping fix pass",
                extra={"page": idx + 1, "stage1_error": stage1_error},
            )
            final_b64 = stage1_b64 if stage1_b64 else original_page_b64
            error = stage1_error
        elif gpt_fix_pages is not None and (idx + 1) not in gpt_fix_pages:
            # Page is not on the owner's GPT-fix list for this book: keep the RunPod result.
            logger.info("Girl-book: GPT fix skipped (page not in GPT_FIX_PAGES)", extra={"page": idx + 1})
            final_b64 = stage1_b64
            error = None
        else:
            logger.info("→ Girl-book: Stage 2 GPT fix pass", extra={"page": idx + 1})
            fixed_b64, fix_error = await fix_face_artifacts_gpt(
                image_b64=stage1_b64,
                prompt=girl_fix_prompt,
                reference_b64=original_page_b64,
                child_b64=child_b64_for_fix,
            )

            # Guard: with the customer photo as image 3, GPT sometimes copies the whole photo
            # (room, pose, clothes) instead of the book page. If the result looks more like the
            # customer photo than the book page, redo the fix without the photo (template hair).
            if not fix_error and fixed_b64 and child_b64_for_fix:
                d_child = _scene_distance(fixed_b64, child_b64_for_fix)
                d_page = _scene_distance(fixed_b64, original_page_b64)
                if d_child < d_page:
                    logger.warning(
                        "Girl-book: GPT copied the customer photo, retrying without it",
                        extra={"page": idx + 1, "d_child": round(d_child, 1), "d_page": round(d_page, 1)},
                    )
                    fixed_b64, fix_error = await fix_face_artifacts_gpt(
                        image_b64=stage1_b64,
                        prompt=GIRL_BOOK_FIX_PROMPT,
                        reference_b64=original_page_b64,
                        child_b64=None,
                    )

            fix_applied = (
                not fix_error
                and fixed_b64 is not None
                and fixed_b64 != stage1_b64
            )

            if fix_applied:
                logger.info("→ Girl-book: fix pass succeeded", extra={"page": idx + 1})
                final_b64 = fixed_b64
                error = None
            else:
                logger.warning(
                    "Girl-book: fix pass failed, keeping RunPod result",
                    extra={"page": idx + 1, "fix_error": fix_error},
                )
                final_b64 = stage1_b64
                error = fix_error

    elif is_magic_flower and not is_regenerate:

        logger.info("→ Stage 1: RunPod", extra={"page": idx + 1})
        stage1_b64, stage1_error = await faceswap_runpod(
            session=session,
            source_b64=source_b64,
            target_b64=original_page_b64,
            masked_b64=masked_b64,
            prompt=prompt,
            face_strength=face_strength,
            style_strength=style_strength,
            ssl_ctx=ssl_ctx,
            refine=refine,
            grow_mask=grow_mask,
            seed=seed
        )

        propso_swapped = (
            not stage1_error
            and stage1_b64 is not None
            and stage1_b64 != original_page_b64
        )

        if not propso_swapped:
            logger.warning(
                "Propsolab did not swap, skipping GPT stage",
                extra={"page": idx + 1, "stage1_error": stage1_error},
            )
            final_b64 = stage1_b64 if stage1_b64 else original_page_b64
            error = stage1_error
        elif propso_swapped and idx in MAGIC_FLOWER_GPT_SKIP:
            logger.info(
                "→ Stage 2: GPT skipped",
                extra={"page": idx + 1},
            )
            final_b64 = stage1_b64
            error = None
        else:
            logger.info("→ Stage 2: GPT", extra={"page": idx + 1})
            enhance_prompt = (
                "TOP PRIORITY — HEAD SIZE: the girl's head in the FIRST image is usually TOO BIG. "
                "Shrink the EXISTING head of the FIRST image uniformly (like scaling a photo layer) until the whole head (face + hair) is the same size, width and height as the girl's head in the SECOND image, "
                "at the same position on the shoulders. Head-to-body ratio must equal the SECOND image (a small child's head, about one quarter to one fifth of her height). "
                "Never keep an oversized head; never make it bigger than in the SECOND image. Body, arms and dress stay the same size. "
                "Use the FIRST image as the base image. "
                "FACE IDENTITY IS LOCKED: do NOT redraw, regenerate, restyle or beautify the face. The face must be the SAME face as in the FIRST image, only smaller: "
                "identical eye shape, eye size ratio, eye color, eyebrows, nose, mouth, lips, chin, face shape, cheeks, skin details and expression. "
                "Never take the face from the SECOND image — from the SECOND image take ONLY the head size, neck and hair. "
                "Do not modify facial identity in any way. "
                "Replace ONLY the hair with the hairstyle from the SECOND image. "
                "Completely remove the original hair from the first image before applying the new hairstyle. "
                "The new hair must attach naturally to the head with correct hairline and skull structure. "
                "No duplicate hair, floating hair, extra buns, detached strands or merged hairstyles. "
                "Preserve original pose, dress, body, background, lighting and art style from the first image. "
                "HEAD SIZE (repeat): final head size must match the SECOND image, not the first. "
                "Neck straight and natural as in the second image, correct length, no bend, twist or tilt; shoulders as in the second image. "
                "The hair must be long, exactly as in the second image, never short. "
                "SKIN TONE: natural, soft, light skin matching the child's real skin tone and the color grading of the original page; no orange, red, tanned, sunburnt, dark or over-saturated face, no harsh contrast or burnt shadows; face and neck the same tone as the arms. "
                "Generate a clean, high-quality image with sharp details, no blur, no deformation, no artifacts and no pixelation."
                " Remove the  dots, lines, scratches, and any other artifacts from face "
                "Ensure the face is smooth and free of any blemishes."
                "remove distortion in any part of the image, especially hand and face"
            )

            gpt_b64, gpt_error = await faceswap_gpt(
                source_b64=original_page_b64,        # Face from Stage 1
                target_b64=stage1_b64,                # Hair + everything from original book
                prompt=enhance_prompt,
            )

            gpt_swapped = (
                not gpt_error
                and gpt_b64 is not None
                and gpt_b64 != stage1_b64
                and gpt_b64 != original_page_b64
            )

            if gpt_swapped:
                logger.info("→ Stage 2: GPT succeeded", extra={"page": idx + 1})
                final_b64 = gpt_b64
                error = None
            else:
                logger.warning(
                    "GPT did not swap, keeping stage1 result",
                    extra={"page": idx + 1, "gpt_error": gpt_error, "gpt_swapped": gpt_swapped},
                )
                final_b64 = stage1_b64
                error = gpt_error

    else:

        if model_type == "GPT":
            final_b64, error = await faceswap_gpt(original_page_b64, source_b64, prompt)
        else:
            final_b64, error = await faceswap_runpod(
                session=session,
                source_b64=source_b64,
                target_b64=original_page_b64,
                masked_b64=masked_b64,
                prompt=prompt,
                face_strength=face_strength,
                style_strength=style_strength,
                ssl_ctx=ssl_ctx,
                refine=refine,
                grow_mask=grow_mask,
                seed=seed
            )

    if not final_b64 or (final_b64 == original_page_b64) or error:
        logger.warning(
            "Final fallback to original page",
            extra={"page": idx + 1, "error": error},
        )
        final_b64 = (
            stage1_b64
            if stage1_b64 and stage1_b64 != original_page_b64
            else original_page_b64
        )

    result_bytes = base64.b64decode(final_b64)

    del final_b64, original_page_b64, masked_b64
    if 'stage1_b64' in locals() and stage1_b64:
        del stage1_b64
    gc.collect()

    return idx, result_bytes, error
    


async def process_data(input_data: dict) -> dict:
    job_id = input_data.get("request_id", "no-id")
    job_context.set(job_id)
    
    logger.info("Job started", extra={"pdf": input_data.get("storybook_pdf")})

    pdf_name = (input_data.get("storybook_pdf") or "").strip()
    if not pdf_name:
        return {"status": "error", "message": "storybook_pdf missing"}
    if not pdf_name.lower().endswith(".pdf"):
        pdf_name += ".pdf"

    source_image   = input_data.get("source_image", "")
    face_strength  = float(input_data.get("face_strength", 0.74))
    style_strength = float(input_data.get("style_strength", 0.74))
    max_pages      = int(input_data.get("n_images", 100))
    model_type     = input_data.get("model_type", "CARTOON_LITE")
    refine         = input_data.get("refine", False)
    
    # if any(name in pdf_name for name in ["jobs_girl", "travel_book_girl"]):
    #     refine = True

    pdf_key        = os.path.splitext(os.path.basename(pdf_name))[0]
    ignore_indices = set(ignore_index_map.get(pdf_key, []))
    is_magic_flower = "magic_flower" in pdf_key.lower()

    _fix_pages = GPT_FIX_PAGES.get(pdf_key)
    gpt_fix_pages: set[int] | None = set(_fix_pages) if _fix_pages is not None else None
    if gpt_fix_pages is not None:
        logger.info("GPT fix limited to listed pages", extra={"pdf_key": pdf_key, "pages": sorted(gpt_fix_pages)})

    is_girl_book = False
    if "_girl" in pdf_key.lower() or "girl_book" in pdf_key.lower() and not is_magic_flower:
        is_girl_book = True


    masked_pdf_path: Path | None = None
    if model_type == "CARTOON_LITE" or model_type == "REALISTIC":
        _raw = (input_data.get("masked_pdf_path") or "").strip()
        if _raw:
            _p = Path(_raw)
            if _p.exists():
                masked_pdf_path = _p
                logger.info("Masked PDF found", extra={"masked_pdf": str(masked_pdf_path)})
            else:
                logger.warning(
                    "masked_pdf_path provided but file not found",
                    extra={"masked_pdf_path": _raw},
                )
        else:
            logger.info(
                "No masked_pdf_path provided, CARTOON_LITE will run without masks",
                extra={"pdf_key": pdf_key},
            )

    grow_mask = 10  # default
 
    if model_type=="GPT":
        prompt = (
            "Redraw the character's facial features to match the reference portrait exactly. "
            "Swap the hair properly and match with given image"
            "Just focus on face and hair"
            "Preserve the original pose. "
            "Match the skin tone, eye shape, nose, cheek and mouth from the reference. "
            "Add happy vibe to the face if you want "
            "Blend the jawline and neck naturally with the existing illustration style. "
            "Do not alter any background elements, clothing, or body proportions."
        )
        if "magic_flower" in pdf_name:
            prompt = (
            "Redraw the character's facial features to match the reference portrait exactly. "
            "Keep the hair exactly same as given image"
            "Just focus on face and hair"
            "Preserve the original pose. "
            "Match the skin tone, eye shape, nose, cheek and mouth from the reference. "
            "Add happy vibe to the face if you want "
            "Blend the jawline and neck naturally with the existing illustration style. "
            "Do not alter any background elements, clothing, or body proportions."
        )
            
        face_strength = 0.78
        style_strength = 0.78
        grow_mask = 0

        logger.info("Magic Flower special parameters applied", extra={"face_strength": face_strength, "style_strength": style_strength})
    else:
        prompt = "Clear face, distortion free image and even eyes"
        grow_mask = 10

    logger.info("Prompt set", extra={"prompt": prompt[:150] + "..." if len(prompt) > 150 else prompt})

    try:
        pdf_path = get_pdf_path(pdf_name)
    except FileNotFoundError as e:
        logger.error(str(e))
        return {"status": "error", "message": str(e)}

    loop = asyncio.get_running_loop()

    try:
        info = await loop.run_in_executor(None, pdfinfo_from_path, str(pdf_path))
        total_pages = min(int(info["Pages"]), max_pages)
        logger.info("PDF info", extra={"total_pages": total_pages})
    except Exception as e:
        logger.error(f"PDF info error: {e}")
        return {"status": "error", "message": f"PDF info error: {e}"}

    try:
        source_bytes = load_image_from_local(source_image)
        source_b64 = base64.b64encode(source_bytes).decode("utf-8")
        del source_bytes
    except Exception as e:
        logger.error(f"Source image error: {e}")
        return {"status": "error", "message": f"Source image error: {e}"}

    # Girl books: decide the hair rule ONCE for the whole book.
    girl_fix_prompt = GIRL_BOOK_FIX_PROMPT
    child_b64_for_fix: str | None = None
    if is_girl_book:
        hair = await detect_hair_length(source_b64)
        logger.info("Girl-book hair check", extra={"hair": hair})
        if hair == "long":
            girl_fix_prompt = GIRL_BOOK_FIX_PROMPT_OWN_HAIR
            child_b64_for_fix = source_b64

    ssl_ctx = make_ssl_context()
    results: dict[int, bytes] = {}
    swap_failure_pages: list[int] = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            ssl=ssl_ctx,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
        ),
        timeout=aiohttp.ClientTimeout(total=300),
    ) as session:

        async def bounded_process(idx: int, page: Image.Image):
            async with semaphore:
                return await process_one_page(
                    session=session,
                    idx=idx,
                    page=page,
                    source_b64=source_b64,
                    prompt=prompt,
                    model_type=model_type,
                    face_strength=face_strength,
                    style_strength=style_strength,
                    ignore_indices=ignore_indices,
                    ssl_ctx=ssl_ctx,
                    masked_pdf_path=masked_pdf_path,
                    grow_mask=grow_mask,
                    refine=refine,
                    seed=242134,          # you can make this dynamic if needed
                    is_magic_flower=is_magic_flower,
                    is_girl_book=is_girl_book,
                    gpt_fix_pages=gpt_fix_pages,
                    girl_fix_prompt=girl_fix_prompt,
                    child_b64_for_fix=child_b64_for_fix,
                )

        for batch_start in range(0, total_pages, PAGE_BATCH_SIZE):
            batch_end = min(batch_start + PAGE_BATCH_SIZE, total_pages)
            logger.info(
                "Decoding batch",
                extra={"batch_start": batch_start + 1, "batch_end": batch_end},
            )

            batch_pages: list[Image.Image] = await loop.run_in_executor(
                None,
                lambda s=batch_start, e=batch_end: convert_from_path(
                    str(pdf_path),
                    dpi=300,
                    first_page=s + 1,
                    last_page=e,
                    thread_count=2,
                ),
            )

            tasks = [
                bounded_process(batch_start + i, page)
                for i, page in enumerate(batch_pages)
            ]
            batch_pages.clear()
            del batch_pages

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in batch_results:
                if isinstance(res, Exception):
                    logger.error(f"Task exception: {res}")
                    continue
                idx, img_bytes, error = res
                if idx in results:
                    logger.error("DUPLICATE PAGE DETECTED", extra={
                        "page": idx + 1,
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                    })
                results[idx] = img_bytes
                if error:
                    swap_failure_pages.append(idx + 1)

            gc.collect()
            logger.info(
                "Batch complete",
                extra={
                    "batch_end": batch_end,
                    "processed_so_far": len(results),
                    "total": total_pages,
                },
            )

    missing = [i + 1 for i in range(total_pages) if i not in results]
    if missing:
        logger.warning("Missing pages", extra={"missing_pages": missing})
    if swap_failure_pages:
        logger.warning(
            "Some swaps failed, originals kept",
            extra={"swap_failures": swap_failure_pages},
        )

    pdf_buffer = BytesIO()
    final_pdf_path = None

    try:
        ordered_bytes = [results[i] for i in range(total_pages) if i in results]
        if not ordered_bytes:
            return {"status": "error", "message": "No pages processed successfully"}

        pil_images = [Image.open(BytesIO(b)) for b in ordered_bytes]

        pil_images[0].save(
            pdf_buffer,
            format="PDF",
            save_all=True,
            append_images=pil_images[1:],
            quality=100,
        )

        final_pdf_path = save_final_pdf(pdf_buffer.getvalue(), job_id + "_swapped")
        if not final_pdf_path:
            raise ValueError("Failed to save final PDF")

    finally:
        pdf_buffer.close()
        for img in pil_images:
            img.close()
        del pil_images, ordered_bytes, results
        gc.collect()

    logger.info(
        "Job finished",
        extra={
            "page_count": total_pages,
            "missing": missing,
            "swap_failures": swap_failure_pages,
            "output": final_pdf_path,
        },
    )

    return {
        "status": "success",
        "pdf_path": final_pdf_path,
        "page_count": total_pages,
        "missing_pages": missing,
        "swap_failures": swap_failure_pages,
    }