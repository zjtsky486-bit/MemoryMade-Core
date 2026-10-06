"""Bounded, process-local cache; callers never share mutable observations."""
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from threading import Lock


class RecognitionCache:
    def __init__(self, maxsize=32):
        self.maxsize = maxsize
        self._values = OrderedDict()
        self._pending = {}
        self._lock = Lock()

    def get(self, key, compute):
        with self._lock:
            if key in self._values:
                self._values.move_to_end(key)
                return deepcopy(self._values[key])
            future = self._pending.get(key)
            leader = future is None
            if leader:
                future = Future()
                self._pending[key] = future
        if not leader:
            return deepcopy(future.result())
        try:
            result = compute()
            # Retry transient detector/detail/core failures on the next upload.
            description = result.get('description', {})
            reusable = (result.get('ok') and not result.get('errors')
                        and not description.get('fallback_reason')
                        and description.get('detail_review', {}).get('status') != 'incomplete')
            snapshot = deepcopy(result)
            with self._lock:
                if reusable:
                    self._values[key] = snapshot
                    while len(self._values) > self.maxsize:
                        self._values.popitem(last=False)
                future.set_result(snapshot)
                self._pending.pop(key, None)
            return deepcopy(snapshot)
        except BaseException as exc:
            with self._lock:
                future.set_exception(exc)
                self._pending.pop(key, None)
            raise
