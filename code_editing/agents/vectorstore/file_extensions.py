extensions = [
    "py",
    "c",
    "cpp",
    "h",
    "hpp",
    "java",
    "ts",
    "json",
    "yaml",
    "yml",
    "sh",
    # "md",
    # "rst",
    # "txt",
    "go",
    "rb",
    "kt",
]


def filter_docs(docs):
    """Filter out the documents that are not relevant for the vector store."""
    exclude = [".git", "boost", "Python24", "third_party", "AppServer/lib"]
    for doc in docs:
        src = doc.metadata["source"]
        src = src.replace("\\", "/")
        if any([ex in src for ex in exclude]):
            continue
        yield doc
