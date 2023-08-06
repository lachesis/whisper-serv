#!/usr/bin/env python
import asyncio
import argparse
import contextlib
import io
import math
import html
import json
import logging
import os
import random
import hashlib
import re
import sys
import time
import urllib
import tempfile
import shutil

from contextlib import contextmanager
from typing import Iterable, List, Union
from pprint import pprint

from quart import Quart, render_template, request, make_response, jsonify, abort
from werkzeug.exceptions import HTTPException, TooManyRequests, BadRequest, InternalServerError

from ffmpeg.asyncio import FFmpeg

logger = logging.getLogger(__name__)

# define the webapp
debug = os.environ.get('ENV') == "debug"
app = Quart(__name__, static_url_path='')
logging.basicConfig(level=logging.INFO)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 * 10
app.config['RESPONSE_TIMEOUT'] = 3600 * 2 + 300

WHISPERCPP_BASE_PATH = os.getenv('WHISPERCPP_BASE_PATH') or '/app/whisper.cpp'

async def is_running(proc):
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), 1e-6)
    return proc.returncode is None

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        #"error": e.name,
        "error": e.description,
    })
    response.content_type = "application/json"
    return response

@contextmanager
def context_timer(name='timer'):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        tim = end - start
        if tim < 1:
            logger.info("Timer %s took %3.0f ms", name, tim*1000)
        else:
            logger.info("Timer %s took %6.2f sec", name, tim)

def str_to_bool(s):
    return bool(s and s.lower() in ['1', 't', 'true', 'yes', 'y'])

@app.route("/transcribe", methods=["POST"])
async def transcribe():
    file = next((await request.files).values())
    if not file:
        raise BadRequest("must supply a file")
    _, ext = os.path.splitext(file.filename)
    if not ext or len(ext) > 5:
        raise BadRequest("must supply a file with a reasonable extension")

    fmt = request.args.get('f') or 'stream'

    out_txt = out_srt = None

    # get a temp dir
    tempd = tempfile.mkdtemp()
    assert tempd

    proc = None
    def clean_up():
        nonlocal proc
        if os.path.exists(tempd):
            shutil.rmtree(tempd)  # remove the tempdir
        if proc:
            proc.kill()

    try:
        # write the posted file to the tempdir
        in_fn = 'input' + ext
        in_fn = os.path.join(tempd, in_fn)
        with open(in_fn, 'wb') as out:
            shutil.copyfileobj(file, out)

        # run ffmpeg on the file
        audio_fn = os.path.join(tempd, 'audio16.wav')
        ffmpeg = (
            FFmpeg()
            .option("y")
            .input(in_fn)
            .output(
                audio_fn,
                acodec='pcm_s16le',
                ac=1,
                ar=16000,
            )
        )
        await ffmpeg.execute()

        # run whisper.cpp on the file on a threadpool
        # stream the raw output from whisper.cpp back to the user TODO

        WHISPERCPP_BIN_PATH = os.path.join(WHISPERCPP_BASE_PATH, 'main')
        WHISPERCPP_MODEL_PATH = os.path.join(WHISPERCPP_BASE_PATH, 'models', 'ggml-large.bin')

        out_fn = os.path.join(tempd, 'out')
        proc = await asyncio.create_subprocess_exec(
            WHISPERCPP_BIN_PATH,
            '-m', WHISPERCPP_MODEL_PATH,
            '--threads', '4',
            '--beam-size', '4',
            '--output-srt',
            '--output-txt',
            '-of', out_fn,
            '-f', audio_fn,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
    except Exception:
        clean_up()
        raise

    if fmt == 'stream':
        async def async_generator():
            try:
                while True:
                    buf = await proc.stdout.readline()
                    if buf:
                        yield buf
                    elif not await is_running(proc):
                        print("Process completed! return code: %r" % proc.returncode)
                        # read and yield the output files
                        for filetype in ['srt', 'txt']:
                            if os.path.exists(out_fn + '.' + filetype):
                                yield ("__OUTPUT_FILE__\tout.{}\n".format(filetype)).encode()
                                with open(out_fn + '.' + filetype) as inp:
                                    yield inp.read().encode()
                        return # all done
            except Exception:
                logger.exception("inner loop err")
                raise
            finally:
                clean_up()  # delete the temp dir
        return async_generator(), 200

    elif fmt in ['txt', 'srt']:
        # wait for the process to end
        try:
            await asyncio.wait_for(proc.wait(), 3600 * 2)
            with open(out_fn + '.' + fmt) as inp:
                return inp.read(), 200
        finally:
            clean_up()  # delete the temp dir

    else:
        raise BadRequest("Unsupported format: %s", fmt)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=debug, use_reloader=not debug, port=8080)

