"""Snapshot do Postgres via ``pg_dump`` para ``data/backups/`` no host.

Gzip + rotação. Roda dentro do container ``scheduler``; ``pg_dump`` é
fornecido pelo pacote ``postgresql-client`` instalado na imagem.

Snapshots vivem no bind-mount ``./data/backups/`` — sobrevivem mesmo se
o volume Postgres for apagado.
"""

import gzip
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = Path(BASE) / 'data' / 'backups'

DB_HOST = os.environ.get('PGHOST', 'db')
DB_PORT = os.environ.get('PGPORT', '5432')
DB_USER = os.environ.get('POSTGRES_USER', 'agenda')
DB_NAME = os.environ.get('POSTGRES_DB', 'agenda')
DB_PASSWORD = os.environ.get('POSTGRES_PASSWORD', '')

RETENCAO_DIAS = int(os.environ.get('AGENDA_SNAPSHOT_RETENTION_DAYS', '30'))


def snapshot():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    destino = BACKUP_DIR / f'agenda-{timestamp}.sql.gz'

    env = os.environ.copy()
    env['PGPASSWORD'] = DB_PASSWORD

    print(f'Snapshot: pg_dump -> {destino}')
    with gzip.open(destino, 'wb') as f:
        subprocess.run(
            [
                'pg_dump',
                '-h', DB_HOST,
                '-p', DB_PORT,
                '-U', DB_USER,
                '-d', DB_NAME,
                '--no-owner',
                '--no-privileges',
            ],
            stdout=f,
            env=env,
            check=True,
        )

    if RETENCAO_DIAS > 0:
        corte = datetime.now() - timedelta(days=RETENCAO_DIAS)
        for arquivo in BACKUP_DIR.glob('agenda-*.sql.gz'):
            if datetime.fromtimestamp(arquivo.stat().st_mtime) < corte:
                arquivo.unlink()
                print(f'Snapshot removido (retenção {RETENCAO_DIAS}d): {arquivo.name}')


if __name__ == '__main__':
    try:
        snapshot()
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f'pg_dump falhou (exit={e.returncode})', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Snapshot falhou: {e}', file=sys.stderr)
        sys.exit(1)
