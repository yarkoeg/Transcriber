#!/bin/zsh
# Двойной щелчок в Finder: запускает сервер и открывает приложение в браузере.
# Закрыть окно Терминала или нажать Ctrl+C — сервер остановится.

cd "${0:A:h}" || exit 1
PORT=${PORT:-8000}
URL="http://localhost:$PORT"

if curl -fs "$URL/api/health" >/dev/null; then
  echo "Сервер уже работает — открываю $URL"
  open "$URL"
  exit 0
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Не найдено окружение .venv в $(pwd)."
  echo "Создайте его по инструкции из README.md и запустите ярлык снова."
  exit 1
fi

# python -m uvicorn, а не .venv/bin/uvicorn: шебанг скрипта ломается при переименовании папки.
.venv/bin/python -m uvicorn backend.app:app --port "$PORT" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' INT TERM HUP EXIT

for _ in {1..120}; do
  if curl -fs "$URL/api/health" >/dev/null; then
    open "$URL"
    echo
    echo "Сервер работает на $URL. Чтобы остановить — закройте это окно или нажмите Ctrl+C."
    break
  fi
  kill -0 $SERVER 2>/dev/null || break
  sleep 0.5
done

wait $SERVER
STATUS=$?
if (( STATUS != 0 && STATUS < 128 )); then
  echo
  echo "Сервер не запустился (код $STATUS). Если порт $PORT занят другой программой — закройте её и запустите ярлык снова."
fi
exit $STATUS
