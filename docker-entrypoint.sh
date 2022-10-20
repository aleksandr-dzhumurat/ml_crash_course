#!/usr/bin/env bash

set -o errexit      # make your script exit when a command fails.
set -o nounset      # exit when your script tries to use undeclared variables.

case "$1" in
  notebook)
    jupyter notebook experiments --ip 0.0.0.0 --port 8888 --allow-root --no-browser
    ;;
  train)
    python3 -c 'import pandas as pd; print(pd.__version__)'
    ;;
  labelstudio)
    label-studio
    ;;
  *)
    exec "$@"
esac