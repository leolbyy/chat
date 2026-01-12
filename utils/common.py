import os


def get_base_dir():
    if 'CHAT_BASE_DIR' in os.environ:
        return os.environ.get('CHAT_BASE_DIR')
    home_dir = os.path.expanduser('~')
    base_dir = os.path.join(home_dir, 'learn', 'chat')
    return base_dir
    