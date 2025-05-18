import aioboto3  # type: ignore
import contextvars
import json
import threading
import logging
from typing import Optional
import inspect
import os
import pathlib
from opentelemetry.instrumentation.utils import suppress_instrumentation

from pydantic_core import to_jsonable_python

logger = logging.getLogger(__name__)
request_id = contextvars.ContextVar("request_id", default="UNKNOWN")
cnt_lock = threading.Lock()
counts: dict[str, int] = {}


async def trace_output(
    data: str | list | dict | None, name_hint: Optional[str] = None
) -> None:
    out_dir = os.environ.get("RESON_TRACE", "")
    if not out_dir:
        return

    req_id = request_id.get()

    if name_hint:
        name = name_hint
    else:
        caller = inspect.stack()[1]
        name = f"{caller.filename}:{caller.function}:{caller.lineno}"

    with cnt_lock:
        count = counts.get(req_id, 0)
        counts[req_id] = count + 1

    serialized = ""
    if isinstance(data, str):
        serialized = data
    elif isinstance(data, (dict, list)):
        serialized = json.dumps(data, indent=4, default=to_jsonable_python)
    elif data is None:
        serialized = "(none)"
    else:
        raise ValueError(f"Unsupported tracing data type: {type(data)}")

    if out_dir.startswith("s3"):
        bucket = "bismuth-traces"
        if ":" in out_dir:
            bucket = out_dir.split(":")[1]

        with suppress_instrumentation():
            try:
                session = aioboto3.Session()
                async with session.client("s3") as s3:
                    await s3.put_object(
                        Bucket=bucket,
                        Key=f"{req_id}/{str(count).zfill(3)}_{name}.txt",
                        Body=serialized.encode("utf-8"),
                    )
            except:
                logger.warning("Failed to upload trace file to s3", exc_info=True)
    else:
        out_dir_p = pathlib.Path(out_dir) / req_id
        out_dir_p.mkdir(parents=True, exist_ok=True)
        fn = out_dir_p / f"{str(count).zfill(3)}_{name}.txt"

        with open(fn, "w") as f:
            f.write(serialized)