from config import DATA_DIR
from rag import build_index_from_json_folder

if __name__ == "__main__":
    build_index_from_json_folder(str(DATA_DIR))