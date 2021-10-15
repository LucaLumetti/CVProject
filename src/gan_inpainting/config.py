import json

class Config():
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._cfg = json.loads(f.read())

    def __str__(self):
        return json.dumps(self._cfg)

    def __getattr__(self, name):
        return self._cfg[name]
