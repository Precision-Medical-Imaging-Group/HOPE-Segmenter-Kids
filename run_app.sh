source ~/Code/ISBI2024-BraTS-GOAT/.mlcubes/bin/activate
python app.py &
ngrok http --host-header=rewrite --domain=optimal-stinkbug-in.ngrok-free.app 7860