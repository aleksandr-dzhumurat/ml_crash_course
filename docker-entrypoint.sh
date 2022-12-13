#!/usr/bin/env bash

set -o errexit      # make your script exit when a command fails.
set -o nounset      # exit when your script tries to use undeclared variables.

case "$1" in
  notebook)
    jupyter notebook src/jupyter_notebooks --ip 0.0.0.0 --port 8888 --allow-root --no-browser  --NotebookApp.token='' --NotebookApp.password=''
    ;;
  train)
    python3 src/train.py
    ;;
  serve)
    flask --app src/service run --host 0.0.0.0
    ;;
  labelstudio)
    label-studio
    ;;
  *)
    exec "$@"
esac