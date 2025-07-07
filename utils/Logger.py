# -----------------------------
# Logger Setup
# -----------------------------
class Logger:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for f in self.files:
            f.write(message)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()
