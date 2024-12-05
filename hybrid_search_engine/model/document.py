import os


class Document:
    def __init__(self,  id = None, content = "", title = "", metadata:dict = None):
        # set PYTHONHASHSEED=0 to make hash deterministic
        os.environ["PYTHONHASHSEED"] = "0"
        self.id = id if id is not None else hash(content)
        self.title = title
        self.content = content
        self.metadata = metadata or {}

        assert self.content is not None and self.content != "", "Document content cannot be empty"

    def get_searchable_text(self):
        return (self.title + "\n" + self.content).strip()

    def __repr__(self):
        return f"Document({self.id}, {self.title}, {self.content})"

    def __str__(self):
        return f"Document({self.id}, {self.title}, {self.content})"

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(
        self,
    ):
        return hash(self.id)