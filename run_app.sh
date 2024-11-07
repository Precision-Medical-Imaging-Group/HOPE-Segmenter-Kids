find /tmp/gradio -mindepth 1 -not -name 'tmp*' -exec rm -rf {} +
source ~/Code/ISBI2024-BraTS-GOAT/.mlcubes/bin/activate
python app.py