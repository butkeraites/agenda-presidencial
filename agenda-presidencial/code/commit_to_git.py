"""Commit + push dos CSVs em ``data/`` para o repositório git remoto.

Backup off-host barato: o último ``df.csv`` em git é a fonte canônica
para ``bootstrap.py`` reconstruir o banco em outra máquina.

Pula silenciosamente se:
- ``GIT_TOKEN`` ou ``GIT_REMOTE`` não estão configurados
- não há mudanças em ``data/``

O container precisa de ``.git/`` bind-mountado para enxergar a história.
"""

import os
import subprocess
import sys
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE, '..', '..'))
PATH_DATA = 'agenda-presidencial/code/data'

GIT_TOKEN = os.environ.get('GIT_TOKEN', '').strip()
GIT_REMOTE = os.environ.get('GIT_REMOTE', '').strip()
GIT_USER_EMAIL = os.environ.get('GIT_USER_EMAIL', 'agenda-bot@local').strip()
GIT_USER_NAME = os.environ.get('GIT_USER_NAME', 'agenda-bot').strip()
GIT_BRANCH = os.environ.get('GIT_BRANCH', 'master').strip()


def run(cmd, **kwargs):
    print('$ ' + ' '.join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, **kwargs)


def commit_and_push():
    if not GIT_TOKEN or not GIT_REMOTE:
        print('GIT_TOKEN/GIT_REMOTE ausentes; pulando commit/push (backup off-host desativado).')
        return 0

    status = subprocess.run(
        ['git', 'status', '--porcelain', '--', PATH_DATA],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    if not status.stdout.strip():
        print('Sem mudanças em data/; nada a commitar.')
        return 0

    run(['git', 'config', 'user.email', GIT_USER_EMAIL])
    run(['git', 'config', 'user.name', GIT_USER_NAME])
    # Marcar o repo como safe — o UID dentro do container pode diferir do dono do bind mount.
    run(['git', 'config', '--global', '--add', 'safe.directory', REPO_ROOT])

    run(['git', 'add', PATH_DATA])
    mensagem = f'Atualizacao automatica dos dados ({datetime.now().strftime("%Y-%m-%d %H:%M")})'
    run(['git', 'commit', '-m', mensagem])

    url_com_token = f'https://x-access-token:{GIT_TOKEN}@{GIT_REMOTE}'
    run(['git', 'push', url_com_token, f'HEAD:{GIT_BRANCH}'])
    print('Push concluído.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(commit_and_push())
    except subprocess.CalledProcessError as e:
        print(f'Comando git falhou (exit={e.returncode})', file=sys.stderr)
        sys.exit(1)
