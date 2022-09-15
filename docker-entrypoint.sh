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
  load)
    python3 /srv/src/tools/load_data.py -i /srv/data/Uploaded_Content_with_OCR_2022_09_09.csv
    ;;
  *)
    exec "$@"
esac