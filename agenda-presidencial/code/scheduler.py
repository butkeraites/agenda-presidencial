"""Loop principal do container ``scheduler``.

Encadeia os cinco passos do pipeline e dorme até o próximo ciclo. Falha
em um passo é registrada mas não derruba o loop — o próximo ciclo tenta
de novo. Logs vão para stdout (Docker captura).

Configuração: ``AGENDA_SCRAPE_INTERVAL_HOURS`` (default 24).
"""

import os
import subprocess
import sys
import time
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
INTERVALO_HORAS = float(os.environ.get('AGENDA_SCRAPE_INTERVAL_HOURS', '24'))

PASSOS = [
    ('bootstrap',     'bootstrap.py'),
    ('scrape',        'main.py'),
    ('processar',     'gerar_dados.py'),
    ('snapshot',      'snapshot.py'),
    ('commit_to_git', 'commit_to_git.py'),
]


def agora():
    return datetime.now().isoformat(timespec='seconds')


def rodar_passo(nome, script):
    print(f'[{agora()}] >>> {nome} ({script})', flush=True)
    try:
        subprocess.run([sys.executable, os.path.join(BASE, script)], check=True)
        print(f'[{agora()}] <<< {nome} OK', flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f'[{agora()}] <<< {nome} FAIL exit={e.returncode}', flush=True)
        return False


def main():
    intervalo_segundos = INTERVALO_HORAS * 3600
    while True:
        print(f'[{agora()}] === Iniciando ciclo (intervalo={INTERVALO_HORAS}h) ===', flush=True)
        for nome, script in PASSOS:
            rodar_passo(nome, script)
        print(f'[{agora()}] === Ciclo concluído. Próximo em {INTERVALO_HORAS}h. ===', flush=True)
        time.sleep(intervalo_segundos)


if __name__ == '__main__':
    main()
