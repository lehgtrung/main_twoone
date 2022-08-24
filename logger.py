

class Logger:
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        with open(self.path, 'a') as f:
            f.write(msg)