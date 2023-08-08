function activate_venv(){
    chmod +x runner.py 
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
}

function friday(){
    python runner.py $@
}
