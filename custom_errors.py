

class FileExists(Exception):
    def __init__(self, filename):
        self.value = filename
    def __str__(self):
    	string = """Exception: %s already exists, pass 'force=True' as argument to overwrite file or delete the file."""%(self.value)
        return repr(string)