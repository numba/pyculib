import os

class Environment:

    def __init__(self):
        """Set config flags based on environment variables."""

        self._environ = os.environ
        WARNINGS = Environment._readenv("PYCULIB_WARNINGS", int, 0)

        globals()['WARNINGS'] = WARNINGS

    @staticmethod
    def _readenv(name, ctor, default):
        value = os.environ.get(name)
        if value is None:
            return default() if callable(default) else default
        try:
            return ctor(value)
        except Exception:
            warnings.warn("environ %s defined but failed to parse '%s'" %
                          (name, res), RuntimeWarning)
            return default

_env = Environment()

