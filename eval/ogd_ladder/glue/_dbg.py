import json, os
_f = open("/tmp/claude-0/glue_dbg_%d.jsonl" % os.getpid(), "w") if os.environ.get("GLUE_DEBUG") else None
def log(msg, act=None):
    if _f:
        _f.write(json.dumps({"msg": msg, "act": act}, ensure_ascii=False) + "\n"); _f.flush()
