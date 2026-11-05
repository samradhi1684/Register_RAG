"""Finds which request option makes Ollama return 500. Tries a ladder of /api/chat requests.
Usage: python scripts/ollama_probe.py --model qwen2.5:3b [--host http://127.0.0.1:11434]"""
import argparse, json, time, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen2.5:3b"); ap.add_argument("--host", default="http://127.0.0.1:11434")
ap.add_argument("--timeout", type=int, default=90)
a = ap.parse_args()
MSG = [{"role": "system", "content": "Reply with one word."}, {"role": "user", "content": "Say OK"}]
LADDER = [
    ("A  no options (like `ollama run`)", {}, {}),
    ("B  temperature=0",               {"temperature": 0}, {}),
    ("C  seed=0",                      {"seed": 0}, {}),
    ("D  num_predict=64",              {"num_predict": 64}, {}),
    ("E  num_ctx=4096",                {"num_ctx": 4096}, {}),
    ("F  num_ctx=8192",                {"num_ctx": 8192}, {}),
    ("G  keep_alive=30m",              {}, {"keep_alive": "30m"}),
    ("H  everything the harness sends", {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 64}, {"keep_alive": "30m"}),
]
for name, opts, extra in LADDER:
    body = {"model": a.model, "stream": False, "messages": MSG, **extra}
    if opts:
        body["options"] = opts
    req = urllib.request.Request(f"{a.host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=a.timeout))
        print(f"OK    {name:36s} {time.time()-t:5.1f}s  -> {r['message']['content'][:30]!r}")
    except urllib.error.HTTPError as e:
        print(f"FAIL  {name:36s} HTTP {e.code}  {e.read()[:140]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name:36s} {type(e).__name__}: {str(e)[:100]}")
